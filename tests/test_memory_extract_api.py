# [테스트] 핵심 정보 추출 API 테스트
# 가짜 LLM으로 backend 흐름(결과 저장 → 다음 대화와 함께 다시 보냄), 짧은 id 변환, 중복·잘못된 요청, LLM 오류를 확인합니다.

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import memory as memory_api
from app.api.v1.memory import get_extraction_service
from app.core.errors import AppError
from app.features.memory.extraction import MemoryBuilder
from app.features.memory.service import ExtractionService
from app.main import create_app
from app.schemas.memory import Conversation, PersonLinks

# Real ids look like analysis job uuids; the LLM only ever sees P1, F1, U1.
JOB1 = "0b7c6a52-3f8e-4c1d-9a55-2e7f1d0c9b11"
JOB2 = "9d1e4f20-6b3a-4e7c-8f02-5a9c3b7e6d44"


def conversation(job: str, day: int, lines: list[tuple[str, str, str]]) -> dict:
    return {
        "id": job, "kind": "meeting", "occurred_at": f"2026-09-{day:02d}T10:00:00+09:00", "title": f"미팅 {day}일",
        "utterances": [
            {"id": f"{job}-{n:04d}", "start": n * 5.0, "end": n * 5.0 + 4, "speaker": speaker, "role": role,
             "text": text}
            for n, (speaker, role, text) in enumerate(lines, start=1)
        ],
    }


FIRST = conversation(JOB1, 7, [
    ("me", "me", "자료는 목요일까지 보내드릴게요."),
    ("A", "customer", "생산관리팀 박지훈 대리입니다. 3개 라인에 쓰려고요."),
])
SECOND = conversation(JOB2, 14, [
    ("me", "me", "보내드린 자료 보셨죠?"),
    ("A", "customer", "네, 박 대리입니다. 라인은 5개로 늘리기로 했어요."),
])


def fact(type_: str, speaker: str, content: str, sources: list[str], **fields) -> dict:
    return {"type": type_, "speaker": speaker, "content": content, "source_utterance_ids": sources,
            "owner": None, "due_text": None, "due_date": None, "supersedes_fact_id": None, **fields}


def person(speaker: str, **fields) -> dict:
    return {"speaker": speaker, "known_person_id": None, "name": None, "surname": None, "title": None,
            "role_in_deal": "unknown", **fields}


REPLIES = [
    {
        "people": [person("A", name="박지훈", title="대리", role_in_deal="champion")],
        "facts": [
            fact("promise", "me", "자료를 보낸다.", ["U1"], owner="me", due_text="목요일까지", due_date="2026-09-10"),
            fact("requirement", "A", "3개 라인에 적용한다.", ["U2"]),
        ],
        "resolved_promise_ids": [],
    },
    {
        "people": [person("A", surname="박", title="대리", known_person_id="P1")],
        "facts": [fact("requirement", "A", "5개 라인으로 늘린다.", ["U2"], supersedes_fact_id="F2")],
        "resolved_promise_ids": ["F1"],
    },
]


class ScriptedLLM:
    name = "fake:llm"

    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.link_prompts: list[str] = []

    def generate(self, system, prompt, schema):
        if schema is PersonLinks:
            # The matching call adds nothing here; the extraction replies already link the speakers.
            self.link_prompts.append(prompt)
            return PersonLinks(links=[])
        self.prompts.append(prompt)
        return schema.model_validate(self.replies.pop(0))


class FailingLLM:
    name = "fake:failing"

    def generate(self, system, prompt, schema):
        raise AppError(status_code=502, code="LLM_BAD_RESPONSE", message="LLM 응답이 형식에 맞지 않습니다.")


def client_with(llm) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_extraction_service] = lambda: ExtractionService(llm)
    return TestClient(app)


class Backend:
    """Stores what /extract returns the way backend would: upsert people and facts by id."""

    def __init__(self) -> None:
        self.conversations: list[dict] = []
        self.people: dict[str, dict] = {}
        self.facts: dict[str, dict] = {}

    def memory(self) -> dict:
        return {"conversations": self.conversations, "people": list(self.people.values()),
                "facts": list(self.facts.values())}

    def save(self, conversation: dict, data: dict) -> None:
        self.conversations.append({k: v for k, v in conversation.items() if k != "utterances"})
        self.people.update({p["id"]: p for p in data["people"]})
        self.facts.update({f["id"]: f for f in data["facts"]})


def test_first_conversation_returns_new_people_and_facts_with_real_ids() -> None:
    llm = ScriptedLLM(REPLIES)

    response = client_with(llm).post("/api/v1/memory/extract", json={"company": "누리정밀", "conversation": FIRST})

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["model"] == "fake:llm"
    [p] = data["people"]
    assert (p["id"], p["name"], p["appearances"]) == (f"{JOB1}-P1", "박지훈", [[JOB1, "A"]])
    assert [f["id"] for f in data["facts"]] == [f"{JOB1}-F01", f"{JOB1}-F02"]
    assert data["facts"][0]["source_utterance_ids"] == [f"{JOB1}-0001"]
    assert data["facts"][0]["status"] == "open"
    # The prompt names utterances by short id only.
    assert "[U2] A(고객): 생산관리팀 박지훈 대리입니다." in llm.prompts[0]
    assert JOB1 not in llm.prompts[0]


def test_stored_memory_round_trip_matches_building_all_at_once() -> None:
    llm = ScriptedLLM(REPLIES)
    client = client_with(llm)
    backend = Backend()

    for conv in (FIRST, SECOND):
        body = {"company": "누리정밀", "conversation": conv, "memory": backend.memory()}
        response = client.post("/api/v1/memory/extract", json=body)
        assert response.status_code == 200
        backend.save(conv, response.json()["data"])

    # Only the second conversation has known people to match, so only it gets the matching call.
    [link_prompt] = llm.link_prompts
    assert "- P1 박지훈 대리, champion, 나온 대화: 2026-09-07 (Mon) \"미팅 7일\"" in link_prompt
    assert "  · [requirement] 3개 라인에 적용한다." in link_prompt
    # The second prompt carries the stored memory under short ids.
    assert "- P1 박지훈 대리, champion, 미팅 7일" in llm.prompts[1]
    assert "- F1 [me] me: 자료를 보낸다. (기한 2026-09-10)" in llm.prompts[1]
    assert "- F2 [requirement] P1: 3개 라인에 적용한다." in llm.prompts[1]
    assert JOB1 not in llm.prompts[1]

    whole = MemoryBuilder(ScriptedLLM(REPLIES)).build(
        "누리정밀", [Conversation.model_validate(FIRST), Conversation.model_validate(SECOND)])
    assert backend.people == {p.id: p.model_dump(mode="json") for p in whole.people}
    assert backend.facts == {f.id: f.model_dump(mode="json") for f in whole.facts}
    promise, old, new = (backend.facts[i] for i in (f"{JOB1}-F01", f"{JOB1}-F02", f"{JOB2}-F01"))
    assert (promise["status"], promise["resolved_in"]) == ("done", JOB2)
    assert (old["superseded_by"], new["supersedes"]) == (new["id"], old["id"])


def test_only_new_or_changed_items_are_returned() -> None:
    backend = Backend()
    client = client_with(ScriptedLLM(REPLIES))
    backend.save(FIRST, client.post("/api/v1/memory/extract",
                                    json={"company": "누리정밀", "conversation": FIRST}).json()["data"])
    unrelated = {"id": "old-F01", "conversation_id": JOB1, "type": "decision", "said_by": "me",
                 "content": "그대로인 정보", "source_utterance_ids": [f"{JOB1}-0001"]}
    backend.facts[unrelated["id"]] = unrelated

    body = {"company": "누리정밀", "conversation": SECOND, "memory": backend.memory()}
    data = client.post("/api/v1/memory/extract", json=body).json()["data"]

    # The person was seen again, the promise is done and the requirement was replaced.
    assert [p["id"] for p in data["people"]] == [f"{JOB1}-P1"]
    assert data["people"][0]["appearances"] == [[JOB1, "A"], [JOB2, "A"]]
    assert sorted(f["id"] for f in data["facts"]) == [f"{JOB1}-F01", f"{JOB1}-F02", f"{JOB2}-F01"]


def test_same_conversation_twice_is_rejected() -> None:
    body = {"company": "누리정밀", "conversation": FIRST,
            "memory": {"conversations": [{k: v for k, v in FIRST.items() if k != "utterances"}]}}

    response = client_with(ScriptedLLM(REPLIES)).post("/api/v1/memory/extract", json=body)

    assert response.status_code == 409
    assert response.json()["code"] == "CONVERSATION_ALREADY_EXTRACTED"


@pytest.mark.parametrize("conversation", [
    FIRST | {"utterances": []},
    FIRST | {"utterances": [FIRST["utterances"][0], FIRST["utterances"][0]]},
    FIRST | {"kind": "email"},
    FIRST | {"id": ""},
    FIRST | {"id": "company"},
])
def test_invalid_conversation_is_rejected_without_calling_the_llm(conversation) -> None:
    llm = ScriptedLLM(REPLIES)

    response = client_with(llm).post("/api/v1/memory/extract",
                                     json={"company": "누리정밀", "conversation": conversation})

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_REQUEST"
    assert llm.prompts == []


STORED_PERSON = {"id": "p-1", "name": "박지훈", "title": "대리", "role_in_deal": "champion",
                 "appearances": [["old", "A"]]}
STORED_FACT = {"id": "f-1", "conversation_id": "old", "type": "requirement", "said_by": "p-1",
               "content": "3개 라인", "source_utterance_ids": ["old-0001"]}
STORED_CONVERSATION = {"id": "old", "kind": "call", "occurred_at": "2026-09-01T10:00:00+09:00", "title": "첫 통화"}


@pytest.mark.parametrize("memory", [
    {"people": [STORED_PERSON, STORED_PERSON]},
    {"facts": [STORED_FACT, STORED_FACT]},
    {"conversations": [STORED_CONVERSATION, STORED_CONVERSATION]},
    {"facts": [STORED_FACT | {"type": "rumor"}]},
])
def test_invalid_memory_is_rejected_without_calling_the_llm(memory) -> None:
    llm = ScriptedLLM(REPLIES)

    response = client_with(llm).post("/api/v1/memory/extract",
                                     json={"company": "누리정밀", "conversation": FIRST, "memory": memory})

    assert response.status_code == 422
    assert llm.prompts == []


def test_too_large_memory_is_rejected_without_calling_the_llm(monkeypatch) -> None:
    from app.features.memory import service as memory_service

    monkeypatch.setattr(memory_service, "MAX_PROMPT_MEMORY_LINES", 2)
    llm = ScriptedLLM(REPLIES)
    facts = [STORED_FACT | {"id": f"f-{n}"} for n in range(2)]
    # Replaced facts are not in the prompt, so they do not count.
    replaced = STORED_FACT | {"id": "f-old", "superseded_by": "f-0"}
    small = {"conversations": [STORED_CONVERSATION], "facts": [facts[0], replaced], "people": [STORED_PERSON]}
    large = small | {"facts": facts}

    client = client_with(llm)
    too_large = client.post("/api/v1/memory/extract",
                            json={"company": "누리정밀", "conversation": FIRST, "memory": large})
    assert too_large.status_code == 413
    assert too_large.json()["code"] == "MEMORY_TOO_LARGE"
    assert llm.prompts == []

    fits = client.post("/api/v1/memory/extract", json={"company": "누리정밀", "conversation": FIRST, "memory": small})
    assert fits.status_code == 200


def test_llm_errors_reach_backend_with_their_code() -> None:
    response = client_with(FailingLLM()).post("/api/v1/memory/extract",
                                              json={"company": "누리정밀", "conversation": FIRST})

    assert response.status_code == 502
    assert response.json()["code"] == "LLM_BAD_RESPONSE"


def test_gateway_client_is_created_once_per_process(monkeypatch) -> None:
    created = []

    def fake_create_llm(settings):
        created.append(settings)
        return ScriptedLLM(REPLIES)

    monkeypatch.setattr(memory_api, "create_llm", fake_create_llm)
    client = TestClient(create_app())
    for conv in (FIRST, SECOND):
        assert client.post("/api/v1/memory/extract",
                           json={"company": "누리정밀", "conversation": conv}).status_code == 200

    assert len(created) == 1


def test_missing_llm_key_returns_503(monkeypatch) -> None:
    from app.core.config import Settings
    from app.providers.llm import create_llm

    monkeypatch.setattr(memory_api, "create_llm", lambda _settings: create_llm(Settings(llm_api_key="")))

    response = TestClient(create_app()).post("/api/v1/memory/extract",
                                             json={"company": "누리정밀", "conversation": FIRST})

    assert response.status_code == 503
    assert response.json()["code"] == "LLM_NOT_CONFIGURED"
