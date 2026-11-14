"""Compare embedding models and chunking settings on the fixtures/sales gold queries.

Every configuration is scored the same way as retrieval_eval.py (company and
permission filter, then cosine similarity), with k=3:

    hit@1     the first chunk contains a relevant utterance
    hit@3     one of the top 3 chunks does
    recall@3  share of relevant utterances covered by the top 3 chunks
    MRR       1 / rank of the first relevant chunk (0 if not in the top 3)
    all@1     hit@1 when searching every customer the viewer can see (no company filter)
    orig@1    hit@1 on the original questions only
    para@1    hit@1 on the reworded questions (paraphrase_of), which avoid the transcript's words
    open@1    hit@1 over every chunk of every customer, ignoring filters (hardest, measurement only)
    random@1  hit@1 of picking one candidate at random; larger chunks raise it, so read
              scores against it when comparing chunk sizes

Usage:
    python scripts/compare_embeddings.py models --out data/eval/models.json
    python scripts/compare_embeddings.py chunking --model BAAI/bge-m3 --out data/eval/chunking.json
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers.text_embedding import sentence_transformers as st  # noqa: E402
from retrieval_eval import (  # noqa: E402
    FIXTURES,
    build_chunks,
    dot,
    evaluate,
    load_scenarios,
    memory_search,
)

MODELS = [
    "BAAI/bge-m3",
    "nlpai-lab/KURE-v1",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-small",
    "Qwen/Qwen3-Embedding-0.6B",
    "jhgan/ko-sroberta-multitask",
]


def score(results: list[dict]) -> dict:
    scored = [r for r in results if not r["expects_nothing"]]
    n = len(scored)
    return {
        "hit@1": sum(r["rr"] == 1 for r in scored) / n,
        "hit@3": sum(r["hit"] for r in scored) / n,
        "recall@3": sum(r["recall"] for r in scored) / n,
        "mrr": sum(r["rr"] for r in scored) / n,
        "leaks": sum(r["leaks"] for r in results),
        "orig@1": hit1([r for r in scored if not r["paraphrase"]]),
        "para@1": hit1([r for r in scored if r["paraphrase"]]),
    }


def hit1(rows: list[dict]) -> float | None:
    return sum(r["rr"] == 1 for r in rows) / len(rows) if rows else None


def random_hit1(scenarios, chunks) -> float:
    """Expected hit@1 of a random pick among the chunks the company filter leaves."""
    rates = []
    for scenario in scenarios:
        candidates = [c for c in chunks if c["company_id"] == scenario["company"]["id"]]
        for query in scenario["gold_queries"]:
            relevant = set(query.get("relevant_utterance_ids", []))
            if query["kind"] != "semantic" or not relevant:
                continue
            visible = [c for c in candidates if c["interaction_id"] not in
                       set(query.get("forbidden_interaction_ids", []))]
            rates.append(sum(bool(c["utterance_ids"] & relevant) for c in visible) / len(visible))
    return sum(rates) / len(rates)


def open_search(chunks, vectors, embedder):
    """Rank every chunk with no company or permission filter, to stress the embedding."""

    def search(query: dict, teams: dict[int, int], limit: int) -> list[dict]:
        q = embedder.embed_query(query["question"])
        ranked = sorted(range(len(chunks)), key=lambda i: dot(q, vectors[i]), reverse=True)
        return [chunks[i] for i in ranked[:limit]]

    return search


def run(scenarios, embedder, max_chars, overlap, context) -> dict:
    chunks = build_chunks(scenarios, max_chars, overlap, context)
    started = time.perf_counter()
    vectors = embedder.embed_documents([c["embed_text"] for c in chunks])
    embed_seconds = time.perf_counter() - started

    questions = [q["question"] for s in scenarios for q in s["gold_queries"] if q["kind"] == "semantic"]
    started = time.perf_counter()
    for question in questions:
        embedder.embed_query(question)
    query_ms = (time.perf_counter() - started) / len(questions) * 1000

    company = score(evaluate(scenarios, memory_search(chunks, vectors, embedder, True), 3))
    everyone = score(evaluate(scenarios, memory_search(chunks, vectors, embedder, False), 3))
    unfiltered = score(evaluate(scenarios, open_search(chunks, vectors, embedder), 3))
    return {
        "chunks": len(chunks),
        "avg_chars": sum(len(c["text"]) for c in chunks) / len(chunks),
        "dims": len(vectors[0]),
        "embed_seconds": embed_seconds,
        "query_ms": query_ms,
        **company,
        "all@1": everyone["hit@1"],
        "all_leaks": everyone["leaks"],
        "open@1": unfiltered["hit@1"],
        "open@3": unfiltered["hit@3"],
        "random@1": random_hit1(scenarios, chunks),
    }


def release(model: str) -> None:
    # Keep only one model in memory while comparing several.
    st.load_model.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def compare_models(scenarios, models, max_chars, overlap) -> list[dict]:
    rows = []
    for model in models:
        embedder = st.SentenceTransformerEmbedder(model)
        started = time.perf_counter()
        try:
            st.load_model(model)
        except Exception as exc:  # report and keep going with the next model
            print(f"  {model}: 불러오기 실패 ({exc})")
            continue
        load_seconds = time.perf_counter() - started
        row = {"model": model, "load_seconds": load_seconds,
               **run(scenarios, embedder, max_chars, overlap, True)}
        rows.append(row)
        print(f"  {model}: hit@1 {row['hit@1']:.1%}  hit@3 {row['hit@3']:.1%}  MRR {row['mrr']:.3f}")
        release(model)
    return rows


def compare_chunking(scenarios, model) -> list[dict]:
    embedder = st.SentenceTransformerEmbedder(model)
    rows = []
    for context in (True, False):
        for max_chars in (150, 300, 500, 800):
            for overlap in (0, 1, 2):
                row = {"max_chars": max_chars, "overlap": overlap, "context": context,
                       **run(scenarios, embedder, max_chars, overlap, context)}
                rows.append(row)
                print(f"  chars={max_chars} overlap={overlap} context={context}: "
                      f"hit@1 {row['hit@1']:.1%}  recall@3 {row['recall@3']:.1%}")
    return rows


def print_models(rows: list[dict]) -> None:
    print("\n| 모델 | 차원 | hit@1 | orig@1 | para@1 | hit@3 | recall@3 | MRR | open@1 | 질문 1개(ms) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (-r["mrr"], -r["hit@1"])):
        print(f"| {r['model']} | {r['dims']} | {r['hit@1']:.1%} | {r['orig@1']:.1%} | {r['para@1']:.1%} "
              f"| {r['hit@3']:.1%} | {r['recall@3']:.1%} | {r['mrr']:.3f} | {r['open@1']:.1%} "
              f"| {r['query_ms']:.0f} |")


def print_chunking(rows: list[dict]) -> None:
    print("\n| 머리말 | 최대 글자 | 겹침 | 청크 수 | 평균 글자 | random@1 | hit@1 | hit@1-random | para@1 | open@1 | recall@3 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {'O' if r['context'] else 'X'} | {r['max_chars']} | {r['overlap']} | {r['chunks']} "
              f"| {r['avg_chars']:.0f} | {r['random@1']:.1%} | {r['hit@1']:.1%} "
              f"| {r['hit@1'] - r['random@1']:+.1%} | {r['para@1']:.1%} | {r['open@1']:.1%} "
              f"| {r['recall@3']:.1%} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["models", "chunking"])
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--model", default=st.BGE_M3, help="model for chunking mode")
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    scenarios = load_scenarios(sorted(FIXTURES.glob("*.json")))
    if args.mode == "models":
        rows = compare_models(scenarios, args.models, args.max_chars, args.overlap)
        print_models(rows)
    else:
        rows = compare_chunking(scenarios, args.model)
        print_chunking(rows)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
