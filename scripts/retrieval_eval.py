"""Score conversation memory search against the gold queries in fixtures/sales.

Chunks every interaction, embeds the chunks, then answers each semantic gold
query the way the pgvector SQL will: filter by company and viewer permission,
rank by cosine similarity, keep the top k.

    hit@k     a top-k chunk contains at least one relevant utterance
    recall@k  share of relevant utterances covered by the top-k chunks
    MRR       1 / rank of the first chunk with a relevant utterance
    leaks     top-k chunks from interactions the viewer must not see (must be 0)

Usage:
    python scripts/retrieval_eval.py --model BAAI/bge-m3 --k 5 --max-chars 300 --overlap 1
    python scripts/retrieval_eval.py --rerank BAAI/bge-reranker-v2-m3 --pool 10
    python scripts/retrieval_eval.py --no-company-filter   # search all of the viewer's customers
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.features.memory.chunking import chunk_utterances, with_context  # noqa: E402
from app.providers.text_embedding.sentence_transformers import (  # noqa: E402
    BGE_M3,
    SentenceTransformerEmbedder,
)
from app.schemas.memory import StoredUtterance  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sales"


def load_scenarios(paths: list[Path]) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def build_chunks(scenarios: list[dict], max_chars: int, overlap: int, context: bool) -> list[dict]:
    chunks = []
    for scenario in scenarios:
        for interaction in scenario["interactions"]:
            utterances = [StoredUtterance.model_validate(u) for u in interaction["utterances"]]
            for chunk in chunk_utterances(utterances, max_chars=max_chars, overlap=overlap):
                chunks.append({
                    "company_id": scenario["company"]["id"],
                    "interaction_id": interaction["id"],
                    "owner_user_id": interaction["owner_user_id"],
                    "visibility": interaction["visibility"],
                    "text": chunk.text,
                    "embed_text": with_context(
                        chunk.text,
                        scenario["company"]["name"],
                        interaction["kind"],
                        interaction["occurred_at"],
                        interaction["title"],
                    ) if context else chunk.text,
                    "utterance_ids": set(chunk.utterance_ids),
                })
    return chunks


def user_teams(scenarios: list[dict]) -> dict[int, int]:
    teams = {}
    for scenario in scenarios:
        for user in scenario["users"]:
            teams[user["user_id"]] = user["team_id"]
    return teams


def visible(chunk: dict, viewer: int, teams: dict[int, int]) -> bool:
    if chunk["owner_user_id"] == viewer:
        return True
    return chunk["visibility"] == "team" and teams[chunk["owner_user_id"]] == teams[viewer]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def memory_search(chunks, vectors, embedder, company_filter):
    """Rank in Python exactly as the pgvector query does (filter, then cosine)."""

    def search(query: dict, teams: dict[int, int], limit: int) -> list[dict]:
        viewer = query["viewer_user_id"]
        candidates = [
            i for i, c in enumerate(chunks)
            if visible(c, viewer, teams)
            and (not company_filter or c["company_id"] == query["company_id"])
        ]
        q = embedder.embed_query(query["question"])
        ranked = sorted(candidates, key=lambda i: dot(q, vectors[i]), reverse=True)
        return [chunks[i] for i in ranked[:limit]]

    return search


PG_SEARCH = """
SELECT company_id, interaction_id, content, embed_text, utterance_ids
FROM memory_sandbox.memory_chunks
WHERE embedding_model = %(model)s
  AND (%(company)s::text IS NULL OR company_id = %(company)s)
  AND (owner_user_id = %(viewer)s OR (visibility = 'team' AND team_id = %(team)s))
ORDER BY embedding <=> %(q)s::vector
LIMIT %(limit)s
"""


def pg_search(conn, embedder, company_filter):
    """Run the real pgvector query against the sandbox filled by load_memory_sandbox.py."""

    def search(query: dict, teams: dict[int, int], limit: int) -> list[dict]:
        viewer = query["viewer_user_id"]
        params = {
            "model": embedder.name,
            "company": query["company_id"] if company_filter else None,
            "viewer": viewer,
            "team": teams[viewer],
            "q": to_vector_literal(embedder.embed_query(query["question"])),
            "limit": limit,
        }
        rows = conn.execute(PG_SEARCH, params).fetchall()
        return [
            {"company_id": r[0], "interaction_id": r[1], "text": r[2], "embed_text": r[3],
             "utterance_ids": set(r[4])}
            for r in rows
        ]

    return search


def to_vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in vector) + "]"


def evaluate(scenarios, search, k, reranker=None, pool=10) -> list[dict]:
    teams = user_teams(scenarios)
    results = []
    for scenario in scenarios:
        for query in scenario["gold_queries"]:
            if query["kind"] != "semantic":
                continue
            top = search(query, teams, max(k, pool) if reranker is not None else k)
            if reranker is not None and top:
                # Re-read the best vector matches against the question and reorder them.
                scores = reranker.predict([(query["question"], c["embed_text"]) for c in top])
                top = [c for _, c in sorted(zip(scores, top), key=lambda p: -p[0])]
            top = top[:k]

            relevant = set(query["relevant_utterance_ids"])
            forbidden = set(query.get("forbidden_interaction_ids", []))
            covered = set().union(*(c["utterance_ids"] for c in top)) & relevant
            first = next((r for r, c in enumerate(top, 1) if c["utterance_ids"] & relevant), None)
            results.append({
                "scenario_id": scenario["scenario_id"],
                "query_id": query["id"],
                "question": query["question"],
                "viewer_user_id": query["viewer_user_id"],
                "expects_nothing": not relevant,
                "hit": first is not None,
                "recall": len(covered) / len(relevant) if relevant else None,
                "rr": 1 / first if first else 0.0,
                "leaks": sum(c["interaction_id"] in forbidden for c in top),
                "other_company": sum(c["company_id"] != query["company_id"] for c in top),
                "top1": top[0]["text"] if top else None,
            })
    return results


def summarize(results: list[dict]) -> dict:
    scored = [r for r in results if not r["expects_nothing"]]
    n = len(scored)
    return {
        "queries": n,
        "hit@k": sum(r["hit"] for r in scored) / n,
        "recall@k": sum(r["recall"] for r in scored) / n,
        "mrr": sum(r["rr"] for r in scored) / n,
        "leaks": sum(r["leaks"] for r in results),
        "other_company_chunks": sum(r["other_company"] for r in results),
        "no_access_queries_ok": sum(
            r["leaks"] == 0 for r in results if r["expects_nothing"]
        ),
        "no_access_queries": sum(r["expects_nothing"] for r in results),
    }


def print_report(summary, results, k) -> None:
    print(f"\n질문 {summary['queries']}개  hit@{k} {summary['hit@k']:.1%}  "
          f"recall@{k} {summary['recall@k']:.1%}  MRR {summary['mrr']:.3f}")
    print(f"권한 위반 {summary['leaks']}건  다른 고객사 청크 {summary['other_company_chunks']}건  "
          f"접근 불가 질문 {summary['no_access_queries_ok']}/{summary['no_access_queries']} 통과")

    by_scenario = defaultdict(list)
    for r in results:
        if not r["expects_nothing"]:
            by_scenario[r["scenario_id"]].append(r)
    print("\n시나리오별 hit / recall")
    for sid, rows in sorted(by_scenario.items()):
        hit = sum(r["hit"] for r in rows) / len(rows)
        recall = sum(r["recall"] for r in rows) / len(rows)
        print(f"  {sid}  hit {hit:5.0%}  recall {recall:5.0%}  ({len(rows)}개)")

    misses = [r for r in results if not r["expects_nothing"] and not r["hit"]]
    if misses:
        print(f"\n상위 {k}개 안에 정답이 없는 질문")
        for r in misses:
            top1 = (r["top1"] or "").replace("\n", " / ")[:90]
            print(f"  {r['query_id']} {r['question']}\n      1위: {top1}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixtures", nargs="*", type=Path, help="scenario JSON files (default: all)")
    parser.add_argument("--model", default=BGE_M3)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--no-company-filter", action="store_true")
    parser.add_argument("--no-context", action="store_true",
                        help="embed chunks without the company/date/title header")
    parser.add_argument("--rerank", metavar="MODEL",
                        help="cross-encoder that reorders the top --pool vector matches, e.g. BAAI/bge-reranker-v2-m3")
    parser.add_argument("--pool", type=int, default=10)
    parser.add_argument("--backend", choices=["memory", "pg"], default="memory",
                        help="pg searches memory_sandbox via MEMORY_DB_URL (fill it with load_memory_sandbox.py)")
    parser.add_argument("--out", type=Path, help="write per-query results as JSON")
    args = parser.parse_args()

    scenarios = load_scenarios(args.fixtures or sorted(FIXTURES.glob("*.json")))
    embedder = SentenceTransformerEmbedder(args.model)
    company_filter = not args.no_company_filter

    if args.backend == "pg":
        conn = connect()
        search = pg_search(conn, embedder, company_filter)
        print(f"{embedder.name}: pgvector memory_sandbox 검색")
    else:
        chunks = build_chunks(scenarios, args.max_chars, args.overlap, not args.no_context)
        started = time.perf_counter()
        vectors = embedder.embed_documents([c["embed_text"] for c in chunks])
        print(f"{embedder.name}: 청크 {len(chunks)}개, {len(vectors[0])}차원, "
              f"임베딩 {time.perf_counter() - started:.1f}초 "
              f"(max_chars={args.max_chars}, overlap={args.overlap}, context={not args.no_context})")
        search = memory_search(chunks, vectors, embedder, company_filter)

    reranker = None
    if args.rerank:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(args.rerank)
    results = evaluate(scenarios, search, args.k, reranker, args.pool)
    summary = summarize(results)
    print_report(summary, results, args.k)

    if args.out:
        rows = [{key: value for key, value in r.items()} for r in results]
        args.out.write_text(
            json.dumps({"config": vars(args) | {"fixtures": None, "out": None},
                        "summary": summary, "results": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def connect():
    import os

    import psycopg

    from app.core.config import load_environment

    load_environment()
    dsn = os.environ.get("MEMORY_DB_URL")
    if not dsn:
        sys.exit("MEMORY_DB_URL이 없습니다. .env에 postgresql://user:pass@localhost:15432/db 형식으로 적어 주세요.")
    return psycopg.connect(dsn)


if __name__ == "__main__":
    main()
