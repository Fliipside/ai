# [테스트] 평가 스크립트의 채점 규칙 테스트
# 정답 짝짓기, 화자 일치율, 사람 인식률, 권한 위반 집계, 보고서 HTML이 의도대로 계산되는지 확인합니다.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import compare_embeddings  # noqa: E402
import load_memory_fixtures  # noqa: E402
import memory_eval  # noqa: E402
import retrieval_eval  # noqa: E402
import voice_benchmark  # noqa: E402

from app.schemas.memory import CompanyMemory, MemoryFact, Person  # noqa: E402
from app.schemas.voice import SpeakerTurn  # noqa: E402


def fact(fact_id: str, sources: list[str], kind: str = "promise", **fields) -> MemoryFact:
    return MemoryFact(id=fact_id, conversation_id="I1", type=kind, said_by=None, content=fact_id,
                      source_utterance_ids=sources, **fields)


def gold_fact(fact_id: str, sources: list[str], kind: str = "promise", **fields) -> dict:
    return {"id": fact_id, "interaction_id": "I1", "type": kind, "said_by": "me", "content": fact_id,
            "source_utterance_ids": sources, **fields}


def test_match_facts_pairs_as_many_same_type_facts_as_possible() -> None:
    predicted = [fact("p1", ["u1", "u2"]), fact("p2", ["u1"])]
    gold = [gold_fact("g1", ["u1", "u2"]), gold_fact("g2", ["u2"])]

    # Taking the best overlap first (p1-g1) would leave p2 and g2 with nothing in common.
    assert memory_eval.match_facts(predicted, gold) == {"p1": "g2", "p2": "g1"}


def test_match_facts_prefers_overlap_and_same_type_and_skips_unrelated_facts() -> None:
    predicted = [fact("p1", ["u1"]), fact("p2", ["u1", "u2"]), fact("p3", ["u9"]),
                 fact("p4", ["u5"], kind="decision")]
    gold = [gold_fact("g1", ["u1", "u2"]), gold_fact("g2", ["u1"]), gold_fact("g3", ["u5"]),
            gold_fact("g4", ["u5"], kind="decision")]

    assert memory_eval.match_facts(predicted, gold) == {"p1": "g2", "p2": "g1", "p4": "g4"}
    assert memory_eval.match_facts([], gold) == {}
    assert memory_eval.match_facts(predicted, []) == {}


SCENARIO = {
    "users": [{"user_id": 1, "team_id": 1}, {"user_id": 2, "team_id": 2}],
    "contacts": [{"id": "C1", "name": "박지훈", "title": "대리"}, {"id": "C2", "name": "최은영", "title": "팀장"}],
    "interactions": [{
        "id": "I1", "owner_user_id": 1, "visibility": "private",
        "utterances": [
            {"speaker": "SPEAKER_00", "role": "me", "contact_id": None},
            {"speaker": "A", "role": "customer", "contact_id": "C1"},
            {"speaker": "B", "role": "customer", "contact_id": "C2"},
        ],
    }],
    "gold_facts": [gold_fact("G1", ["u1"], owner="me", due_at="2026-09-10", status="open")],
    "gold_queries": [{"id": "Q1", "kind": "structured", "question": "남은 할 일?", "viewer_user_id": 1,
                      "relevant_fact_ids": ["G1"]}],
}


def memory_with(people: list[Person], facts: list[MemoryFact]) -> CompanyMemory:
    return CompanyMemory(company="누리정밀", conversations=[], people=people, facts=facts)


def test_customer_the_model_left_out_counts_against_recognition() -> None:
    people = [Person(id="P1", name="박지훈", title="대리", role_in_deal="unknown", appearances=[("I1", "A")])]
    contacts = memory_eval.speaker_contacts(SCENARIO)

    result = memory_eval.score_people(SCENARIO, memory_with(people, []), contacts)

    # people_seen counts the speakers an extracted person covers; the rate is out of seen + missing.
    assert (result["people_seen"], result["people_full"], result["people_missing"]) == (1, 1, 1)


def test_open_promise_answer_needs_the_same_owner_and_due_date_not_only_the_same_evidence() -> None:
    def score(due_at: str) -> dict:
        promise = fact("I1-F01", ["u1"], owner="me", due_at=due_at, status="open")
        [result] = memory_eval.score_queries(SCENARIO, memory_with([], [promise]), {"I1-F01": "G1"})
        return result

    assert (score("2026-09-10")["evidence_exact"], score("2026-09-10")["exact"]) == (True, True)
    assert (score("2026-09-30")["evidence_exact"], score("2026-09-30")["exact"]) == (True, False)


def test_cache_key_changes_with_the_gateway_and_the_schema_fields(tmp_path) -> None:
    class Gateway:
        name = "gateway:model"

        def __init__(self, base_url: str) -> None:
            self.base_url = base_url
            self.calls = 0

        def generate(self, system, prompt, schema):
            self.calls += 1
            return schema(id="x", name=None, title=None, role_in_deal="unknown", appearances=[])

    class Other(Person):
        extra: str | None = None

    Other.__name__ = "Person"  # same name, other fields
    first, second = Gateway("https://a.example/v1"), Gateway("https://b.example/v1")

    memory_eval.CachedLLM(first, tmp_path).generate("s", "p", Person)
    memory_eval.CachedLLM(first, tmp_path).generate("s", "p", Person)
    memory_eval.CachedLLM(second, tmp_path).generate("s", "p", Person)
    memory_eval.CachedLLM(first, tmp_path).generate("s", "p", Other)

    assert (first.calls, second.calls) == (2, 1)


def test_report_shows_conversation_text_as_text() -> None:
    attack = "</script><script>alert(1)</script><b>"
    promise = fact("I1-F01", ["u1"], owner="me", status="open")
    promise.content = attack
    memory = memory_with([Person(id="P1", name=attack, title=None, role_in_deal="unknown", appearances=[])],
                         [promise])
    memory.company = attack

    html = memory_eval.render_html(memory, {"nodes": [{"id": "n", "title": attack}], "edges": []},
                                   "2026-11-01", "정답 데이터")

    assert attack not in html
    assert html.count("</script>") == 2  # the vis-network tag and the page's own script
    assert "\\u003c/script>" in html  # still the same text once the browser parses the JSON


def test_agreement_does_not_depend_on_the_order_of_overlapping_turns() -> None:
    turns = [SpeakerTurn(start=0, end=2, speaker="A"), SpeakerTurn(start=1, end=2, speaker="B")]

    assert voice_benchmark.to_frames(turns, 200) == voice_benchmark.to_frames(turns[::-1], 200)
    assert voice_benchmark.agreement(turns, turns[::-1], 2.0) == 1.0
    # Only the second where one speaker talks is compared.
    other = [SpeakerTurn(start=0, end=0.5, speaker="X"), SpeakerTurn(start=1, end=2, speaker="X")]
    assert voice_benchmark.agreement(turns, other, 2.0) == 0.5


def chunk(interaction_id: str, owner: int, visibility: str, utterance_ids: set[str]) -> dict:
    return {"company_id": "CO1", "interaction_id": interaction_id, "owner_user_id": owner,
            "visibility": visibility, "utterance_ids": utterance_ids, "text": "", "embed_text": ""}


RETRIEVAL = [{
    "scenario_id": "S001", "company": {"id": "CO1"},
    "users": [{"user_id": 1, "team_id": 1}, {"user_id": 2, "team_id": 2}],
    "interactions": [
        {"id": "I1", "owner_user_id": 1, "visibility": "private"},
        {"id": "I2", "owner_user_id": 2, "visibility": "private"},
    ],
    # No forbidden_interaction_ids: I2 is still not viewer 1's to see.
    "gold_queries": [{"id": "Q1", "kind": "semantic", "question": "q", "viewer_user_id": 1,
                      "company_id": "CO1", "relevant_utterance_ids": ["u1"]}],
}]
CHUNKS = [chunk("I1", 1, "private", {"u1"}), chunk("I2", 2, "private", {"u2"})]


def test_leaks_are_counted_from_who_may_see_the_interaction() -> None:
    def leaky_search(query, teams, limit):
        return CHUNKS

    [result] = retrieval_eval.evaluate(RETRIEVAL, leaky_search, 5)

    assert result["leaks"] == 1


def test_random_baseline_uses_only_chunks_the_viewer_may_see() -> None:
    assert compare_embeddings.random_hit1(RETRIEVAL, CHUNKS) == 1.0


def test_failed_model_does_not_stop_the_comparison(monkeypatch) -> None:
    released = []

    def run(scenarios, embedder, max_chars, overlap, context):
        if embedder.model == "bad":
            raise RuntimeError("query embedding failed")
        return {"hit@1": 1.0, "hit@3": 1.0, "mrr": 1.0}

    monkeypatch.setattr(compare_embeddings.st, "load_model", lambda model: None)
    monkeypatch.setattr(compare_embeddings, "run", run)
    monkeypatch.setattr(compare_embeddings, "release", released.append)

    rows = compare_embeddings.compare_models([], ["bad", "good"], 500, 1)

    assert [row["model"] for row in rows] == ["good"]
    assert released == ["bad", "good"]


def test_fixture_load_never_changes_an_existing_user() -> None:
    sql = "\n".join(load_memory_fixtures.source_sql([{
        "users": [{"user_id": 1, "team_id": 1}], "company": {"id": "CO1", "name": "누리정밀"},
        "contacts": [], "interactions": [], "gold_facts": [],
    }]))

    assert "DO UPDATE" not in sql
    assert "ON CONFLICT (user_id) DO NOTHING" in sql
    assert "RAISE EXCEPTION" in sql  # a user in another team stops the load


def test_missing_model_chunks_stop_the_db_evaluation() -> None:
    import pytest

    retrieval_eval.require_chunks("BAAI/bge-m3", 3)
    with pytest.raises(SystemExit):
        retrieval_eval.require_chunks("BAAI/bge-m3", 0)
