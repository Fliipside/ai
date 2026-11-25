# [테스트] 관계도·관계 타임라인 API 테스트
# 정보 추출로 쌓은 기억을 그대로 보내 대화별 사건, 처음 나온 사람, 남은 약속·기한 지남, 관계도 노드·연결선,
# 권한 때문에 일부 대화만 보낸 경우를 확인합니다.

import pytest
from fastapi.testclient import TestClient

from app.features.memory.extraction import MemoryBuilder
from app.main import create_app
from app.schemas.memory import Conversation
from tests.test_memory_extract_api import FIRST, JOB1, JOB2, REPLIES, SECOND, ScriptedLLM


def stored(*conversations: dict) -> dict:
    """The memory backend would have stored after extracting these conversations."""
    replies = REPLIES[:len(conversations)]
    memory = MemoryBuilder(ScriptedLLM(replies)).build(
        "누리정밀", [Conversation.model_validate(c) for c in conversations])
    return {
        "conversations": [{k: v for k, v in c.items() if k != "utterances"} for c in conversations],
        "people": [p.model_dump(mode="json") for p in memory.people],
        "facts": [f.model_dump(mode="json") for f in memory.facts],
    }


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def post(client: TestClient, path: str, memory: dict, as_of: str | None = "2026-11-01") -> dict:
    body = {"company": "누리정밀", "memory": memory} | ({"as_of": as_of} if as_of else {})
    response = client.post(f"/api/v1/memory/{path}", json=body)
    assert response.status_code == 200, response.text
    return response.json()["data"]


def test_timeline_shows_each_conversation_in_order(client) -> None:
    data = post(client, "timeline", stored(FIRST, SECOND))

    first, second = data["entries"]
    assert (first["conversation_id"], second["conversation_id"]) == (JOB1, JOB2)
    assert first["participants"] == [{"id": f"{JOB1}-P1", "label": "박지훈 대리", "role_in_deal": "champion",
                                      "first_seen": True, "inferred": False}]
    assert second["participants"][0]["first_seen"] is False
    assert [(e["type"], e["fact_type"], e["said_by"]) for e in first["events"]] == [
        ("promise", "promise", "나"), ("fact", "requirement", "박지훈 대리")]
    assert first["events"][0]["due_at"] == "2026-09-10"
    changed, done = second["events"]
    assert (changed["type"], changed["content"], changed["previous_content"]) == (
        "changed", "5개 라인으로 늘린다.", "3개 라인에 적용한다.")
    assert (done["type"], done["content"]) == ("promise_done", "자료를 보낸다.")
    # The only promise was kept, so nothing is left to do.
    assert data["upcoming"] == []


@pytest.mark.parametrize(("as_of", "overdue"), [("2026-09-09", False), ("2026-11-01", True)])
def test_open_promise_is_overdue_after_its_due_date(client, as_of, overdue) -> None:
    data = post(client, "timeline", stored(FIRST), as_of=as_of)

    [promise] = data["upcoming"]
    assert (promise["content"], promise["owner"], promise["due_at"]) == ("자료를 보낸다.", "me", "2026-09-10")
    assert promise["overdue"] is overdue
    assert data["as_of"] == as_of


def test_as_of_defaults_to_today(client) -> None:
    data = post(client, "timeline", stored(FIRST), as_of=None)

    assert len(data["as_of"]) == 10 and data["as_of"] > "2026-09-10"
    assert data["upcoming"][0]["overdue"] is True


def test_graph_nodes_and_edges(client) -> None:
    data = post(client, "graph", stored(FIRST, SECOND))

    nodes = {n["id"]: n for n in data["nodes"]}
    edges = {(e["from"], e["to"], e["type"]) for e in data["edges"]}
    person, promise, old, new = (nodes[i] for i in (f"{JOB1}-P1", f"{JOB1}-F01", f"{JOB1}-F02", f"{JOB2}-F01"))
    assert (person["type"], person["role_in_deal"]) == ("person", "champion")
    assert (promise["type"], promise["fact_type"], promise["status"], promise["group"]) == (
        "fact", "promise", "done", "promise")
    assert (old["group"], new["title"]) == ("superseded", "5개 라인으로 늘린다.")
    assert nodes[JOB1]["type"] == "conversation"
    assert {(f"{JOB1}-P1", "company", "member_of"), (f"{JOB1}-P1", JOB2, "attended"),
            (new["id"], old["id"], "changed"), (promise["id"], JOB2, "resolved_in")} <= edges


def test_graph_marks_overdue_promise(client) -> None:
    nodes = {n["id"]: n for n in post(client, "graph", stored(FIRST))["nodes"]}

    assert nodes[f"{JOB1}-F01"]["group"] == "overdue"


def test_hidden_conversations_leave_no_dangling_links(client) -> None:
    # The viewer may see only the first conversation: backend sends that one with the whole memory.
    memory = stored(FIRST, SECOND)
    memory["conversations"] = memory["conversations"][:1]

    graph = post(client, "graph", memory)
    timeline = post(client, "timeline", memory)

    node_ids = {n["id"] for n in graph["nodes"]}
    assert JOB2 not in node_ids and f"{JOB2}-F01" not in node_ids
    assert all(e["from"] in node_ids and e["to"] in node_ids for e in graph["edges"])
    # The newer fact that replaced it is hidden, so the earlier one is shown as current.
    assert next(n for n in graph["nodes"] if n["id"] == f"{JOB1}-F02")["group"] == "requirement"
    assert [e["conversation_id"] for e in timeline["entries"]] == [JOB1]
    assert all(e["type"] != "promise_done" for e in timeline["entries"][0]["events"])
    # It was kept in the hidden conversation, so to this viewer it is still open.
    promise = next(n for n in graph["nodes"] if n["id"] == f"{JOB1}-F01")
    assert (promise["status"], promise["group"]) == ("open", "overdue")
    assert [u["fact_id"] for u in timeline["upcoming"]] == [f"{JOB1}-F01"]


def test_guessed_link_is_marked_for_a_check(client) -> None:
    memory = stored(FIRST, SECOND)
    memory["people"][0]["inferred_in"] = [JOB2]

    graph = post(client, "graph", memory)
    timeline = post(client, "timeline", memory)

    person = next(n for n in graph["nodes"] if n["id"] == f"{JOB1}-P1")
    assert person["inferred_in"] == [JOB2] and "추정 연결" in person["label"]
    assert [e["participants"][0]["inferred"] for e in timeline["entries"]] == [False, True]


@pytest.mark.parametrize("kind", ["conversations", "people", "facts"])
def test_ids_that_clash_with_graph_nodes_are_rejected(client, kind) -> None:
    # Every id becomes a node id, next to the fixed "company" and "me" nodes.
    memory = stored(FIRST)
    memory[kind][0]["id"] = "me"
    response = client.post("/api/v1/memory/graph", json={"company": "누리정밀", "memory": memory})

    assert response.status_code == 422


def test_ids_must_differ_across_kinds(client) -> None:
    memory = stored(FIRST)
    memory["facts"][0]["id"] = memory["people"][0]["id"]
    response = client.post("/api/v1/memory/graph", json={"company": "누리정밀", "memory": memory})

    assert response.status_code == 422


def test_empty_memory(client) -> None:
    data = post(client, "timeline", {"conversations": [], "people": [], "facts": []})

    assert data["entries"] == [] and data["upcoming"] == []
    assert [n["id"] for n in post(client, "graph", {})["nodes"]] == ["company", "me"]


@pytest.mark.parametrize("body", [{"memory": {}}, {"company": "누리정밀"}, {"company": "", "memory": {}}])
def test_invalid_request(client, body) -> None:
    assert client.post("/api/v1/memory/graph", json=body).status_code == 422


def test_time_without_offset_is_korean_time(client) -> None:
    # One conversation sent without an offset used to break sorting with a 500.
    memory = stored(FIRST, SECOND)
    memory["conversations"][0]["occurred_at"] = "2026-09-07T10:00:00"
    memory["conversations"][1]["occurred_at"] = "2026-09-07T02:00:00Z"  # 11:00 in Korea

    entries = post(client, "timeline", memory)["entries"]

    assert [e["conversation_id"] for e in entries] == [JOB1, JOB2]
    assert entries[0]["occurred_at"] == "2026-09-07T10:00:00+09:00"


def test_due_date_that_is_not_a_date_counts_as_none(client) -> None:
    memory = stored(FIRST)
    promise = memory["facts"][0]
    later = promise | {"id": "later", "due_at": "2026-12-01"}
    memory["facts"] = [promise | {"due_at": "아무때나"}, later]

    upcoming = post(client, "timeline", memory)["upcoming"]

    assert [(p["fact_id"], p["overdue"]) for p in upcoming] == [("later", False), (promise["id"], False)]
