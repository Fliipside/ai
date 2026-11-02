"""Write a SQL file that loads fixtures/sales, chunks and embeddings into schema `memory`.

Create the schema first with scripts/sql/01_memory_schema.sql. The generated SQL runs
in one transaction and replaces only the fixture rows (companies in fixtures/sales and
everything under them); other rows in `memory` and other schemas are untouched.
No DB password is needed here: run the file in DBeaver, or inside the DB container.

Usage:
    python scripts/load_memory_fixtures.py --out data/memory/load_fixtures.sql
    # then either open the file in DBeaver and run it (Option+X), or
    ssh <server> 'docker exec -i <db-container> psql -U <user> -d <db> -v ON_ERROR_STOP=1' \\
        < data/memory/load_fixtures.sql
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
    load_scenarios,
    to_vector_literal,
    user_teams,
)


def lit(value) -> str:
    """SQL literal for the generated script."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "ARRAY[" + ", ".join(lit(v) for v in value) + "]::text[]"
    return "'" + str(value).replace("'", "''") + "'"


def row(table: str, values: list) -> str:
    return f"INSERT INTO memory.{table} VALUES ({', '.join(lit(v) for v in values)});"


def source_sql(scenarios: list[dict]) -> list[str]:
    lines = []
    for user_id, team_id in sorted(user_teams(scenarios).items()):
        lines.append(
            f"INSERT INTO memory.app_users VALUES ({user_id}, {team_id}) "
            "ON CONFLICT (user_id) DO UPDATE SET team_id = EXCLUDED.team_id;"
        )
    for s in scenarios:
        company = s["company"]
        lines.append(row("companies", [company["id"], company["name"], company.get("industry")]))
        for c in s["contacts"]:
            lines.append(row("contacts", [c["id"], company["id"], c["name"], c.get("title"),
                                          c.get("role_in_deal")]))
        for i in s["interactions"]:
            lines.append(row("interactions", [i["id"], company["id"], i["owner_user_id"],
                                              i["visibility"], i["kind"], i["occurred_at"], i["title"]]))
            for seq, u in enumerate(i["utterances"]):
                lines.append(row("utterances", [u["id"], i["id"], seq, u["start"], u["end"],
                                                u["speaker"], u["role"], u["contact_id"], u["text"]]))
        for f in s["gold_facts"]:
            lines.append(row("facts", [f["id"], f["interaction_id"], f["type"], f["said_by"],
                                       f["content"], f["source_utterance_ids"], f.get("owner"),
                                       f.get("due_at"), f.get("due_text"), f.get("status"),
                                       f.get("supersedes")]))
    return lines


def chunk_sql(scenarios, embedder, max_chars, overlap, context) -> list[str]:
    teams = user_teams(scenarios)
    occurred = {i["id"]: i["occurred_at"] for s in scenarios for i in s["interactions"]}
    chunks = build_chunks(scenarios, max_chars, overlap, context)
    vectors = embedder.embed_documents([c["embed_text"] for c in chunks])
    lines = []
    for c, vector in zip(chunks, vectors):
        values = ", ".join([
            lit(c["company_id"]), lit(c["interaction_id"]), lit(c["owner_user_id"]),
            lit(teams[c["owner_user_id"]]), lit(c["visibility"]), lit(occurred[c["interaction_id"]]),
            lit(c["text"]), lit(c["embed_text"]), lit(sorted(c["utterance_ids"])),
            lit(c["start"]), lit(c["end"]), f"'{to_vector_literal(vector)}'::vector", lit(embedder.name),
        ])
        lines.append(
            "INSERT INTO memory.memory_chunks (company_id, interaction_id, owner_user_id, team_id, "
            "visibility, occurred_at, content, embed_text, utterance_ids, start_sec, end_sec, "
            f"embedding, embedding_model) VALUES ({values});"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=BGE_M3)
    parser.add_argument("--max-chars", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("data/memory/load_fixtures.sql"))
    args = parser.parse_args()

    scenarios = load_scenarios(sorted(FIXTURES.glob("*.json")))
    company_ids = lit([s["company"]["id"] for s in scenarios])
    embedder = SentenceTransformerEmbedder(args.model)
    started = time.perf_counter()
    chunks = chunk_sql(scenarios, embedder, args.max_chars, args.overlap, not args.no_context)

    sql = [
        "-- Generated by scripts/load_memory_fixtures.py. Replaces fixture rows only.",
        "BEGIN;",
        # interactions cascade to utterances, facts and memory_chunks
        f"DELETE FROM memory.interactions WHERE company_id = ANY({company_ids});",
        f"DELETE FROM memory.contacts WHERE company_id = ANY({company_ids});",
        f"DELETE FROM memory.companies WHERE id = ANY({company_ids});",
        *source_sql(scenarios),
        *chunks,
        "COMMIT;",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sql) + "\n", encoding="utf-8")
    print(f"{args.out}: 시나리오 {len(scenarios)}개, 청크 {len(chunks)}개 "
          f"({embedder.name}, {time.perf_counter() - started:.1f}초)")


if __name__ == "__main__":
    main()
