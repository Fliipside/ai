# [테스트] 핵심 정보 추출과 관계 기억 테스트
# 가짜 LLM으로 화자 연결, 잘못된 근거 버리기, 약속 완료·요구 변경 반영, 남은 약속 조회, 관계도를 확인합니다.

from datetime import datetime

from app.features.memory.extraction import MemoryBuilder
from app.features.memory.relations import build_graph, open_promises
from app.schemas.memory import CompanyMemory, Conversation, ConversationExtraction


def conversation(cid: str, day: int, lines: list[tuple[str, str, str]]) -> Conversation:
    return Conversation(
        id=cid, kind="meeting", occurred_at=datetime.fromisoformat(f"2026-09-{day:02d}T10:00:00+09:00"),
        title=f"미팅 {cid}",
        utterances=[
            {"id": f"{cid}-U{n}", "start": n, "end": n + 0.5, "speaker": speaker, "role": role, "text": text}
            for n, (speaker, role, text) in enumerate(lines)
        ],
    )


FIRST = conversation("I1", 7, [
    ("SPEAKER_00", "me", "자료는 목요일까지 보내드릴게요."),
    ("SPEAKER_01", "customer", "생산관리팀 박지훈 대리입니다. 3개 라인에 쓰려고요."),
])
SECOND = conversation("I2", 14, [
    ("SPEAKER_00", "me", "보내드린 자료 보셨죠?"),
    ("SPEAKER_02", "customer", "네 박지훈입니다. 라인은 5개로 늘리기로 했어요."),
])

REPLIES = [
    {
        "people": [{"speaker": "SPEAKER_01", "known_person_id": None, "name": "박지훈", "surname": None,
                    "title": "대리", "role_in_deal": "champion"},
                   {"speaker": "SPEAKER_09", "known_person_id": None, "name": "없는 화자", "surname": None,
                    "title": None, "role_in_deal": "unknown"}],
        "facts": [
            {"type": "promise", "speaker": "me", "content": "자료를 보낸다.", "source_utterance_ids": ["I1-U0"],
             "owner": "me", "due_text": "목요일까지", "due_date": "2026-09-10", "supersedes_fact_id": None},
            {"type": "requirement", "speaker": "SPEAKER_01", "content": "3개 라인에 적용한다.",
             "source_utterance_ids": ["I1-U1"], "owner": None, "due_text": None, "due_date": None,
             "supersedes_fact_id": None},
            {"type": "decision", "speaker": "SPEAKER_01", "content": "근거 없는 정보",
             "source_utterance_ids": ["I9-U9"], "owner": None, "due_text": None, "due_date": None,
             "supersedes_fact_id": None},
        ],
        "resolved_promise_ids": [],
    },
    {
        "people": [{"speaker": "SPEAKER_02", "known_person_id": None, "name": None, "surname": "박",
                    "title": "대리", "role_in_deal": "unknown"}],
        "facts": [
            {"type": "requirement", "speaker": "SPEAKER_02", "content": "5개 라인으로 늘린다.",
             "source_utterance_ids": ["I2-U1"], "owner": None, "due_text": None, "due_date": "아무때나",
             "supersedes_fact_id": "I1-F02"},
        ],
        "resolved_promise_ids": ["I1-F01", "I1-F99"],
    },
]


class FakeLLM:
    name = "fake"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, system, prompt, schema):
        self.prompts.append(prompt)
        return schema.model_validate(REPLIES[len(self.prompts) - 1])


def build() -> tuple[CompanyMemory, FakeLLM]:
    llm = FakeLLM()
    # Given out of order on purpose: conversations are processed by date.
    return MemoryBuilder(llm).build("누리정밀", [SECOND, FIRST]), llm


def test_conversations_are_extracted_in_date_order_with_earlier_memory() -> None:
    _, llm = build()

    assert "[I1-U1] SPEAKER_01(고객): 생산관리팀 박지훈 대리입니다." in llm.prompts[0]
    assert "2026-09-07 (Mon) 미팅" in llm.prompts[0]
    # The second prompt carries what the first conversation left behind.
    assert "- P1 박지훈 대리, champion, 미팅 I1" in llm.prompts[1]
    assert "- I1-F01 [me] 자료를 보낸다. (기한 2026-09-10)" in llm.prompts[1]
    assert "- I1-F02 [requirement] 3개 라인에 적용한다." in llm.prompts[1]


def test_surname_and_title_link_to_the_known_person() -> None:
    memory, _ = build()

    [person] = memory.people
    assert (person.name, person.title, person.role_in_deal) == ("박지훈", "대리", "champion")
    assert person.appearances == [("I1", "SPEAKER_01"), ("I2", "SPEAKER_02")]
    assert (person.surname, person.inferred_in) == ("박", [])
    assert {f.said_by for f in memory.facts} == {"me", person.id}


def test_facts_without_real_evidence_are_dropped() -> None:
    memory, _ = build()

    assert [f.id for f in memory.facts] == ["I1-F01", "I1-F02", "I2-F01"]
    assert memory.facts[2].due_at is None


def test_promise_resolution_and_requirement_change() -> None:
    memory, _ = build()
    promise, old, new = memory.facts

    assert (promise.status, promise.resolved_in, promise.due_at) == ("done", "I2", "2026-09-10")
    assert (old.superseded_by, new.supersedes) == ("I2-F01", "I1-F02")


def test_open_promises_respect_owner_and_visibility() -> None:
    memory = MemoryBuilder(FakeLLM()).build("누리정밀", [FIRST])

    assert [f.id for f in open_promises(memory, "me")] == ["I1-F01"]
    assert open_promises(memory, "customer") == []
    assert open_promises(memory, "me", visible_conversations={"I2"}) == []


def test_graph_links_people_conversations_and_facts() -> None:
    memory, _ = build()

    graph = build_graph(memory, as_of="2026-11-01")
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = {(e["from"], e["to"], e["label"]) for e in graph["edges"]}

    assert nodes["P1"]["label"] == "박지훈 대리\n내부 지지자"
    assert nodes["I1-F02"]["group"] == "superseded"
    assert nodes["I1-F01"]["label"] == "약속 · 완료 · ~2026-09-10"
    assert {("P1", "company", "소속"), ("P1", "I2-F01", "말함"), ("I2-F01", "I1-F02", "변경"),
            ("I1-F01", "I2", "완료"), ("P1", "I1", "참석")} <= edges


def test_overdue_promise_is_marked() -> None:
    memory = MemoryBuilder(FakeLLM()).build("누리정밀", [FIRST])

    nodes = {n["id"]: n for n in build_graph(memory, as_of="2026-11-01")["nodes"]}
    assert nodes["I1-F01"]["group"] == "overdue"
    assert ConversationExtraction.model_json_schema()["required"] == ["people", "facts", "resolved_promise_ids"]


def person(speaker: str, **fields) -> dict:
    return {"speaker": speaker, "known_person_id": None, "name": None, "surname": None, "title": None,
            "role_in_deal": "unknown", **fields}


def extract(people_per_conversation: list[list[dict]]) -> CompanyMemory:
    replies = [{"people": people, "facts": [], "resolved_promise_ids": []} for people in people_per_conversation]

    class Scripted:
        name = "scripted"

        def generate(self, system, prompt, schema):
            return schema.model_validate(replies.pop(0))

    return MemoryBuilder(Scripted()).build("누리정밀", [FIRST, SECOND][:len(people_per_conversation)])


def test_context_link_is_marked_as_inferred() -> None:
    memory = extract([
        [person("SPEAKER_01", name="최은영", title="팀장", role_in_deal="influencer")],
        [person("SPEAKER_02", known_person_id="P1", title="팀장")],
    ])

    [p] = memory.people
    assert (p.name, p.inferred_in) == ("최은영", ["I2"])


def test_partial_names_stay_separate_people_and_label_what_is_known() -> None:
    from app.features.memory.extraction import person_label

    memory = extract([
        [person("SPEAKER_01", title="과장")],
        [person("SPEAKER_02", surname="최", title="팀장")],
    ])

    assert [person_label(p) for p in memory.people] == ["과장(이름 미상)", "최 팀장"]


def test_context_link_needs_identifying_details_without_conflict() -> None:
    nothing_known = extract([[person("SPEAKER_01")], [person("SPEAKER_02", known_person_id="P1")]])
    other_title = extract([[person("SPEAKER_01", title="구매 매니저")],
                           [person("SPEAKER_02", known_person_id="P1", title="총지배인")]])

    assert len(nothing_known.people) == 2
    assert [p.title for p in other_title.people] == ["구매 매니저", "총지배인"]


def test_conflicting_full_names_are_not_merged() -> None:
    memory = extract([
        [person("SPEAKER_01", name="박지훈", title="대리")],
        [person("SPEAKER_02", name="박민수", known_person_id="P1", title="대리")],
    ])

    assert [p.name for p in memory.people] == ["박지훈", "박민수"]


def test_given_name_only_is_joined_with_the_surname() -> None:
    memory = extract([
        [person("SPEAKER_01", name="박지훈", title="대리")],
        [person("SPEAKER_02", name="지훈", surname="박", title="대리")],
    ])

    [p] = memory.people
    assert (p.name, p.appearances) == ("박지훈", [("I1", "SPEAKER_01"), ("I2", "SPEAKER_02")])
