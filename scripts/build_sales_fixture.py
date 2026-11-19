import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "sales"
FACT_TYPES = {"requirement", "price_reaction", "promise", "purchase_timing", "buying_signal", "decision"}
ROLES_IN_DEAL = {"champion", "influencer", "decision_maker", "procurement"}
SECONDS_PER_CHAR = 0.18
GAP = 0.6


class DraftError(ValueError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise DraftError(message)


def build(draft: dict) -> dict:
    sid = draft["scenario_id"]
    check(len(sid) == 4 and sid.startswith("S") and sid[1:].isdigit(), f"scenario_id 형식: {sid}")
    full = lambda local: f"{sid}-{local}"  # noqa: E731

    contacts = []
    for n, c in enumerate(draft["contacts"], start=1):
        check(c["role_in_deal"] in ROLES_IN_DEAL, f"role_in_deal: {c}")
        contacts.append({"id": full(f"C{n}"), "name": c["name"], "title": c["title"], "role_in_deal": c["role_in_deal"]})
    contact_ids = {f"C{n}" for n in range(1, len(contacts) + 1)}
    users = {u["user_id"]: u["team_id"] for u in draft["users"]}
    check(users.get(draft["seller"]["user_id"]) == draft["seller"]["team_id"], "seller가 users에 없음")

    interactions, utterance_ids = [], {}
    previous_time = ""
    for n, raw in enumerate(draft["interactions"], start=1):
        iid = f"I{n}"
        check(raw["kind"] in {"meeting", "call"}, f"{iid} kind")
        check(raw["visibility"] in {"private", "team"}, f"{iid} visibility")
        check(raw["owner_user_id"] in users, f"{iid} owner_user_id가 users에 없음")
        check(raw["occurred_at"] > previous_time, f"{iid} 날짜가 이전 대화보다 빠름")
        previous_time = raw["occurred_at"]
        labels, utterances, participants, clock = {"me": "SPEAKER_00"}, [], [], 2.0
        for u, (who, text) in enumerate(raw["lines"]):
            check(who == "me" or who == "unknown" or who in contact_ids, f"{iid} 화자 {who}")
            check(bool(text.strip()), f"{iid}-U{u:02d} 빈 발화")
            if who == "unknown":
                # Each unknown voice is a different person unless written as unknown again later.
                key = "unknown"
            else:
                key = who
            if key not in labels:
                labels[key] = f"SPEAKER_{len(labels):02d}"
            if who in contact_ids and who not in participants:
                participants.append(who)
            role = "me" if who == "me" else "unknown" if who == "unknown" else "customer"
            end = round(clock + max(1.0, len(text) * SECONDS_PER_CHAR), 1)
            utterances.append({
                "id": full(f"{iid}-U{u:02d}"), "start": clock, "end": end, "speaker": labels[key],
                "role": role, "text": text, "contact_id": full(who) if role == "customer" else None,
            })
            utterance_ids[f"{iid}-U{u:02d}"] = (iid, who)
            clock = round(end + GAP, 1)
        interactions.append({
            "id": full(iid), "kind": raw["kind"], "occurred_at": raw["occurred_at"], "title": raw["title"],
            "owner_user_id": raw["owner_user_id"], "visibility": raw["visibility"],
            "participant_contact_ids": [full(c) for c in participants], "utterances": utterances,
        })
    interaction_ids = {f"I{n}" for n in range(1, len(interactions) + 1)}
    order = {f"I{n}": n for n in range(1, len(interactions) + 1)}

    facts = []
    fact_ids = {f"F{n:02d}" for n in range(1, len(draft["facts"]) + 1)}
    for n, raw in enumerate(draft["facts"], start=1):
        fid = f"F{n:02d}"
        check(raw["type"] in FACT_TYPES, f"{fid} type")
        check(raw["src"] and all(s in utterance_ids for s in raw["src"]), f"{fid} src가 없는 발화: {raw['src']}")
        conversation = {utterance_ids[s][0] for s in raw["src"]}
        check(len(conversation) == 1, f"{fid} 근거가 여러 대화에 걸침")
        [iid] = conversation
        said_by = raw["said_by"]
        check(said_by == "me" or said_by in contact_ids, f"{fid} said_by")
        fact = {"id": full(fid), "interaction_id": full(iid), "type": raw["type"],
                "said_by": "me" if said_by == "me" else full(said_by), "content": raw["content"],
                "source_utterance_ids": [full(s) for s in raw["src"]]}
        if raw.get("supersedes"):
            check(raw["supersedes"] in fact_ids and raw["supersedes"] < fid, f"{fid} supersedes")
            fact["supersedes"] = full(raw["supersedes"])
        if raw["type"] == "promise":
            check(raw.get("owner") in {"me", "customer"}, f"{fid} owner")
            resolved = raw.get("resolved_in")
            check(resolved is None or (resolved in interaction_ids and order[resolved] > order[iid]),
                  f"{fid} resolved_in은 약속 이후 대화여야 함")
            fact.update({"owner": raw["owner"], "due_at": raw.get("due_at"), "due_text": raw.get("due_text"),
                         "status": "done" if resolved else "open",
                         "resolved_in": full(resolved) if resolved else None})
        facts.append(fact)

    def visible(viewer: int, iid: str) -> bool:
        raw = draft["interactions"][order[iid] - 1]
        if raw["owner_user_id"] == viewer:
            return True
        return raw["visibility"] == "team" and users[raw["owner_user_id"]] == users.get(viewer)

    queries, paraphrases = [], []
    for n, raw in enumerate(draft["queries"], start=1):
        qid = f"Q{n:02d}"
        viewer = raw["viewer_user_id"]
        check(viewer in users, f"{qid} viewer_user_id가 users에 없음")
        query = {"id": full(qid), "question": raw["question"], "kind": raw["kind"],
                 "company_id": full("CO1"), "as_of": raw.get("as_of", draft["as_of"]), "viewer_user_id": viewer}
        if raw["kind"] == "semantic":
            evidence = raw.get("utterances", [])
            check(all(e in utterance_ids for e in evidence), f"{qid} 없는 발화: {evidence}")
            check(all(visible(viewer, utterance_ids[e][0]) for e in evidence), f"{qid} 볼 권한 없는 근거")
            query["relevant_utterance_ids"] = [full(e) for e in evidence]
            forbidden = raw.get("forbidden", [])
            check(all(f in interaction_ids and not visible(viewer, f) for f in forbidden), f"{qid} forbidden")
            check(bool(evidence or forbidden), f"{qid} 근거도 금지 대화도 없음")
            if forbidden:
                query["forbidden_interaction_ids"] = [full(f) for f in forbidden]
        else:
            check(raw["kind"] == "structured", f"{qid} kind")
            check(raw.get("facts") and all(f in fact_ids for f in raw["facts"]), f"{qid} facts")
            query["relevant_fact_ids"] = [full(f) for f in raw["facts"]]
        query["answer"] = raw["answer"]
        queries.append(query)
        for p, text in enumerate(raw.get("paraphrases", []), start=1):
            check(text != raw["question"], f"{qid} 바꿔 말한 질문이 원래와 같음")
            paraphrase = {"id": full(f"{qid}-P{p}"), "question": text, "paraphrase_of": full(qid)}
            paraphrase.update({k: v for k, v in query.items() if k not in ("id", "question")})
            paraphrases.append(paraphrase)

    return {
        "scenario_id": sid, "description": draft["description"], "fictional": True, "as_of": draft["as_of"],
        "deal_status": draft["deal_status"], "seller": draft["seller"], "users": draft["users"],
        "company": {"id": full("CO1"), **draft["company"]}, "contacts": contacts, "interactions": interactions,
        "gold_facts": facts, "gold_queries": queries + paraphrases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="초안 JSON을 fixtures/sales 형식으로 변환")
    parser.add_argument("drafts", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    failed = False
    for path in args.drafts:
        try:
            draft = json.loads(path.read_text(encoding="utf-8"))
            check(draft["deal_status"] in {"open", "won", "lost"}, "deal_status")
            fixture = build(draft)
        except (DraftError, KeyError, TypeError, ValueError) as exc:
            print(f"✗ {path.name}: {type(exc).__name__}: {exc}")
            failed = True
            continue
        target = args.out / f"{draft['scenario_id'].lower()}_{draft['slug']}.json"
        target.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"✓ {target.name}  대화 {len(fixture['interactions'])}, "
              f"정답 {len(fixture['gold_facts'])}, 질문 {len(fixture['gold_queries'])}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
