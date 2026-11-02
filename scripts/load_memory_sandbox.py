"""Fill the memory_sandbox schema with fixtures/sales, chunks and embeddings.

Create the schema first with scripts/sql/01_sandbox_schema.sql. Each run empties
the memory_sandbox tables and loads them again; no other schema is touched.
The connection string comes from MEMORY_DB_URL (.env), e.g. through an SSH tunnel:

    ssh -N -L 15432:localhost:5432 <server>
    MEMORY_DB_URL=postgresql://<user>:<password>@localhost:15432/<db>

Usage:
    python scripts/load_memory_sandbox.py --model BAAI/bge-m3 --max-chars 300 --overlap 1
    python scripts/retrieval_eval.py --backend pg --k 3
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers.text_embedding.sentence_transformers import (  # noqa: E402
    BGE_M3,
    SentenceTransformerEmbedder,
)
from retrieval_eval import (  # noqa: E402
    FIXTURES,
    build_chunks,
    connect,
    load_scenarios,
    to_vector_literal,
    user_teams,
)

TABLES = ["memory_chunks", "facts", "utterances", "interactions", "contacts", "companies", "app_users"]


def load_source(cur, scenarios: list[dict]) -> None:
    for user_id, team_id in sorted(user_teams(scenarios).items()):
        cur.execute("INSERT INTO memory_sandbox.app_users VALUES (%s, %s)", (user_id, team_id))
    for s in scenarios:
        company = s["company"]
        cur.execute(
            "INSERT INTO memory_sandbox.companies VALUES (%s, %s, %s)",
            (company["id"], company["name"], company.get("industry")),
        )
        for c in s["contacts"]:
            cur.execute(
                "INSERT INTO memory_sandbox.contacts VALUES (%s, %s, %s, %s, %s)",
                (c["id"], company["id"], c["name"], c.get("title"), c.get("role_in_deal")),
            )
        for i in s["interactions"]:
            cur.execute(
                "INSERT INTO memory_sandbox.interactions VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (i["id"], company["id"], i["owner_user_id"], i["visibility"], i["kind"],
                 i["occurred_at"], i["title"]),
            )
            for seq, u in enumerate(i["utterances"]):
                cur.execute(
                    "INSERT INTO memory_sandbox.utterances VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (u["id"], i["id"], seq, u["start"], u["end"], u["speaker"], u["role"],
                     u["contact_id"], u["text"]),
                )
        for f in s["gold_facts"]:
            cur.execute(
                "INSERT INTO memory_sandbox.facts VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (f["id"], f["interaction_id"], f["type"], f["said_by"], f["content"],
                 f["source_utterance_ids"], f.get("owner"), f.get("due_at"), f.get("due_text"),
                 f.get("status"), f.get("supersedes")),
            )


def load_chunks(cur, scenarios, embedder, max_chars, overlap, context) -> int:
    teams = user_teams(scenarios)
    occurred = {i["id"]: i["occurred_at"] for s in scenarios for i in s["interactions"]}
    chunks = build_chunks(scenarios, max_chars, overlap, context)
    vectors = embedder.embed_documents([c["embed_text"] for c in chunks])
    for chunk, vector in zip(chunks, vectors):
        utterance_ids = sorted(chunk["utterance_ids"])
        cur.execute(
            """
            INSERT INTO memory_sandbox.memory_chunks
                (company_id, interaction_id, owner_user_id, team_id, visibility, occurred_at,
                 content, embed_text, utterance_ids, start_sec, end_sec, embedding, embedding_model)
            SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, min(start_sec), max(end_sec), %s::vector, %s
            FROM memory_sandbox.utterances WHERE id = ANY(%s)
            """,
            (chunk["company_id"], chunk["interaction_id"], chunk["owner_user_id"],
             teams[chunk["owner_user_id"]], chunk["visibility"], occurred[chunk["interaction_id"]],
             chunk["text"], chunk["embed_text"], utterance_ids, to_vector_literal(vector),
             embedder.name, utterance_ids),
        )
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=BGE_M3)
    parser.add_argument("--max-chars", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--no-context", action="store_true")
    args = parser.parse_args()

    scenarios = load_scenarios(sorted(FIXTURES.glob("*.json")))
    embedder = SentenceTransformerEmbedder(args.model)
    started = time.perf_counter()
    with connect() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE " + ", ".join(f"memory_sandbox.{t}" for t in TABLES))
        load_source(cur, scenarios)
        count = load_chunks(cur, scenarios, embedder, args.max_chars, args.overlap, not args.no_context)
    print(f"memory_sandbox: 시나리오 {len(scenarios)}개, 청크 {count}개 적재 "
          f"({embedder.name}, {time.perf_counter() - started:.1f}초)")


if __name__ == "__main__":
    main()
