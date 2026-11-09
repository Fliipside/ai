import json
from pathlib import Path

import pytest

from app.schemas.voice import Utterance

FIXTURES = sorted((Path(__file__).parent.parent / "fixtures" / "sales").glob("*.json"))
FACT_TYPES = {"requirement", "price_reaction", "promise", "purchase_timing", "buying_signal", "decision"}


@pytest.fixture(params=FIXTURES, ids=lambda p: p.name)
def scenario(request) -> dict:
    return json.loads(request.param.read_text(encoding="utf-8"))


def test_fixtures_exist() -> None:
    assert FIXTURES


def test_utterances_match_voice_schema(scenario) -> None:
    contacts = {c["id"] for c in scenario["contacts"]}
    for interaction in scenario["interactions"]:
        participants = set(interaction["participant_contact_ids"])
        assert participants <= contacts
        previous_end = 0.0
        for raw in interaction["utterances"]:
            utterance = Utterance.model_validate(raw)
            assert previous_end <= utterance.start < utterance.end
            previous_end = utterance.end
            if utterance.role in {"me", "unknown"}:
                assert raw["contact_id"] is None
            else:
                assert raw["contact_id"] in participants


def test_ids_are_unique(scenario) -> None:
    ids = [scenario["company"]["id"]]
    ids += [c["id"] for c in scenario["contacts"]]
    for interaction in scenario["interactions"]:
        ids.append(interaction["id"])
        ids += [u["id"] for u in interaction["utterances"]]
    ids += [f["id"] for f in scenario["gold_facts"]]
    ids += [q["id"] for q in scenario["gold_queries"]]

    assert len(ids) == len(set(ids))


def test_facts_point_to_their_interaction(scenario) -> None:
    interactions = {i["id"]: {u["id"] for u in i["utterances"]} for i in scenario["interactions"]}
    contacts = {c["id"] for c in scenario["contacts"]}
    facts = {f["id"] for f in scenario["gold_facts"]}
    for fact in scenario["gold_facts"]:
        assert fact["type"] in FACT_TYPES
        assert fact["said_by"] == "me" or fact["said_by"] in contacts
        assert fact["source_utterance_ids"]
        assert set(fact["source_utterance_ids"]) <= interactions[fact["interaction_id"]]
        assert fact.get("supersedes") is None or fact["supersedes"] in facts
        if fact["type"] == "promise":
            assert fact["owner"] in {"me", "customer"}
            assert fact["status"] in {"open", "done"}
            assert (fact["status"] == "done") == (fact["resolved_in"] in interactions)


def test_queries_point_to_existing_evidence(scenario) -> None:
    utterances = {u["id"] for i in scenario["interactions"] for u in i["utterances"]}
    facts = {f["id"] for f in scenario["gold_facts"]}
    for query in scenario["gold_queries"]:
        assert query["company_id"] == scenario["company"]["id"]
        if query["kind"] == "semantic":
            # An empty answer is only expected when permissions hide the evidence.
            assert query["relevant_utterance_ids"] or query.get("forbidden_interaction_ids")
            assert set(query["relevant_utterance_ids"]) <= utterances
        else:
            assert query["relevant_fact_ids"]
            assert set(query["relevant_fact_ids"]) <= facts


def test_paraphrases_share_the_original_answer(scenario) -> None:
    queries = {q["id"]: q for q in scenario["gold_queries"]}
    for query in scenario["gold_queries"]:
        if "paraphrase_of" not in query:
            continue
        original = queries[query["paraphrase_of"]]
        assert "paraphrase_of" not in original
        assert query["question"] != original["question"]
        for key in ("kind", "company_id", "viewer_user_id", "relevant_utterance_ids",
                    "forbidden_interaction_ids"):
            assert query.get(key) == original.get(key), key


def can_view(interaction: dict, viewer: dict, teams: dict[int, int]) -> bool:
    if interaction["owner_user_id"] == viewer["user_id"]:
        return True
    return interaction["visibility"] == "team" and teams[interaction["owner_user_id"]] == viewer["team_id"]


def test_query_evidence_respects_permissions(scenario) -> None:
    assert scenario["deal_status"] in {"open", "won", "lost"}
    teams = {u["user_id"]: u["team_id"] for u in scenario["users"]}
    assert teams[scenario["seller"]["user_id"]] == scenario["seller"]["team_id"]
    interactions = {i["id"]: i for i in scenario["interactions"]}
    utterance_to_interaction = {
        u["id"]: i["id"] for i in scenario["interactions"] for u in i["utterances"]
    }
    for interaction in interactions.values():
        assert interaction["owner_user_id"] in teams
        assert interaction["visibility"] in {"private", "team"}
    for query in scenario["gold_queries"]:
        viewer = {"user_id": query["viewer_user_id"], "team_id": teams[query["viewer_user_id"]]}
        for uid in query.get("relevant_utterance_ids", []):
            assert can_view(interactions[utterance_to_interaction[uid]], viewer, teams)
        for iid in query.get("forbidden_interaction_ids", []):
            assert not can_view(interactions[iid], viewer, teams)
