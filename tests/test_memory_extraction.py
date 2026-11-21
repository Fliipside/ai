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
            {"type": "promise", "speaker": "me", "content": "자료를 보낸다.", "source_utterance_ids": ["U1"],
             "owner": "me", "due_text": "목요일까지", "due_date": "2026-09-10", "supersedes_fact_id": None},
            {"type": "requirement", "speaker": "SPEAKER_01", "content": "3개 라인에 적용한다.",
             "source_utterance_ids": ["U2"], "owner": None, "due_text": None, "due_date": None,
             "supersedes_fact_id": None},
            {"type": "decision", "speaker": "SPEAKER_01", "content": "근거 없는 정보",
             "source_utterance_ids": ["U9"], "owner": None, "due_text": None, "due_date": None,
             "supersedes_fact_id": None},
        ],
        "resolved_promise_ids": [],
    },
    {
        "people": [{"speaker": "SPEAKER_02", "known_person_id": None, "name": None, "surname": "박",
                    "title": "대리", "role_in_deal": "unknown"}],
        "facts": [
            {"type": "requirement", "speaker": "SPEAKER_02", "content": "5개 라인으로 늘린다.",
             "source_utterance_ids": ["U2"], "owner": None, "due_text": None, "due_date": "아무때나",
             "supersedes_fact_id": "F2"},
        ],
        "resolved_promise_ids": ["F1", "F99"],
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

    assert "[U2] SPEAKER_01(고객): 생산관리팀 박지훈 대리입니다." in llm.prompts[0]
    assert "2026-09-07 (Mon) 미팅" in llm.prompts[0]
    # The second prompt carries what the first conversation left behind.
    assert "- P1 박지훈 대리, champion, 미팅 I1" in llm.prompts[1]
    assert "- F1 [me] me: 자료를 보낸다. (기한 2026-09-10)" in llm.prompts[1]
    # Earlier facts name who said them, so an unnamed speaker can be recognized.
    assert "- F2 [requirement] P1: 3개 라인에 적용한다." in llm.prompts[1]


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

    assert nodes["I1-P1"]["label"] == "박지훈 대리\n내부 지지자"
    assert nodes["I1-F02"]["group"] == "superseded"
    assert nodes["I1-F01"]["label"] == "약속 · 완료 · ~2026-09-10"
    assert {("I1-P1", "company", "소속"), ("I1-P1", "I2-F01", "말함"), ("I2-F01", "I1-F02", "변경"),
            ("I1-F01", "I2", "완료"), ("I1-P1", "I1", "참석")} <= edges


def test_overdue_promise_is_marked() -> None:
    memory = MemoryBuilder(FakeLLM()).build("누리정밀", [FIRST])

    nodes = {n["id"]: n for n in build_graph(memory, as_of="2026-11-01")["nodes"]}
    assert nodes["I1-F01"]["group"] == "overdue"
    assert ConversationExtraction.model_json_schema()["required"] == ["people", "facts", "resolved_promise_ids"]


def person(speaker: str, **fields) -> dict:
    return {"speaker": speaker, "known_person_id": None, "name": None, "surname": None, "title": None,
            "role_in_deal": "unknown", **fields}


# The second conversation without anyone's name in it.
UNNAMED = conversation("I2", 14, [
    ("SPEAKER_00", "me", "보내드린 자료 보셨죠?"),
    ("SPEAKER_02", "customer", "네, 라인은 5개로 늘리기로 했어요."),
])


def extract(people_per_conversation: list[list[dict]], second: Conversation = SECOND) -> CompanyMemory:
    replies = [{"people": people, "facts": [], "resolved_promise_ids": []} for people in people_per_conversation]

    class Scripted:
        name = "scripted"

        def generate(self, system, prompt, schema):
            return schema.model_validate(replies.pop(0))

    return MemoryBuilder(Scripted()).build("누리정밀", [FIRST, second][:len(people_per_conversation)])


def test_context_link_is_marked_as_inferred() -> None:
    memory = extract([
        [person("SPEAKER_01", name="박지훈", title="팀장", role_in_deal="influencer")],
        [person("SPEAKER_02", known_person_id="P1", title="팀장")],
    ])

    [p] = memory.people
    assert (p.name, p.inferred_in) == ("박지훈", ["I2"])


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
    other = conversation("I2", 14, [("SPEAKER_02", "customer", "구매팀 박민수 대리입니다.")])
    memory = extract([
        [person("SPEAKER_01", name="박지훈", title="대리")],
        [person("SPEAKER_02", name="박민수", known_person_id="P1", title="대리")],
    ], second=other)

    assert [p.name for p in memory.people] == ["박지훈", "박민수"]


def test_given_name_only_is_joined_with_the_surname() -> None:
    memory = extract([
        [person("SPEAKER_01", name="박지훈", title="대리")],
        [person("SPEAKER_02", name="지훈", surname="박", title="대리")],
    ])

    [p] = memory.people
    assert (p.name, p.appearances) == ("박지훈", [("I1", "SPEAKER_01"), ("I2", "SPEAKER_02")])


def build_with(replies: list[dict]) -> CompanyMemory:
    class Scripted:
        name = "scripted"

        def generate(self, system, prompt, schema):
            return schema.model_validate(replies.pop(0))

    return MemoryBuilder(Scripted()).build("누리정밀", [FIRST, SECOND])


def said(speaker: str, source: str) -> dict:
    return {"type": "requirement", "speaker": speaker, "content": "3개 라인에 적용한다.", "source_utterance_ids": [source],
            "owner": None, "due_text": None, "due_date": None, "supersedes_fact_id": None}


def test_unnamed_person_is_linked_by_what_they_said_before() -> None:
    memory = build_with([
        {"people": [person("SPEAKER_01")], "facts": [said("SPEAKER_01", "U2")], "resolved_promise_ids": []},
        {"people": [person("SPEAKER_02", known_person_id="P1")], "facts": [],
         "resolved_promise_ids": []},
    ])

    [p] = memory.people
    assert (p.appearances, p.inferred_in) == ([("I1", "SPEAKER_01"), ("I2", "SPEAKER_02")], ["I2"])


def test_duty_description_does_not_rule_out_a_title() -> None:
    memory = build_with([
        {"people": [person("SPEAKER_01", title="도입 실무 담당")], "facts": [], "resolved_promise_ids": []},
        {"people": [person("SPEAKER_02", surname="정", title="팀장", known_person_id="P1")], "facts": [], "resolved_promise_ids": []},
    ])

    [p] = memory.people
    assert (p.surname, p.title, len(p.appearances)) == ("정", "팀장", 2)


def test_name_not_said_in_the_conversation_links_as_inferred() -> None:
    # Models copy a known name onto a speaker they only guess is that person.
    memory = extract([
        [person("SPEAKER_01", name="박지훈", title="대리")],
        [person("SPEAKER_02", name="박지훈", title="대리")],
    ], second=UNNAMED)

    [p] = memory.people
    assert (len(p.appearances), p.inferred_in) == (2, ["I2"])


def test_name_said_in_the_conversation_links_for_sure() -> None:
    memory = extract([
        [person("SPEAKER_01", name="박지훈", title="대리")],
        [person("SPEAKER_02", name="박지훈", title="대리")],
    ])

    [p] = memory.people
    assert (len(p.appearances), p.inferred_in) == (2, [])


def test_speaker_label_copied_with_its_role_still_counts() -> None:
    # The prompt shows "SPEAKER_01(고객)"; models sometimes copy the role along with the label.
    memory = build_with([
        {"people": [person("SPEAKER_01(고객)", name="박지훈")],
         "facts": [said("SPEAKER_01(고객)", "U2"), said("me(나)", "U1")], "resolved_promise_ids": []},
        {"people": [], "facts": [], "resolved_promise_ids": []},
    ])

    [p] = memory.people
    assert p.appearances == [("I1", "SPEAKER_01")]
    assert [f.said_by for f in memory.facts] == [p.id, "me"]


def test_surname_not_said_with_a_title_links_as_inferred() -> None:
    memory = extract([
        [person("SPEAKER_01", name="박지훈", title="대리")],
        [person("SPEAKER_02", surname="박", title="대리")],
    ], second=UNNAMED)

    [p] = memory.people
    assert p.inferred_in == ["I2"]


def test_known_person_id_picks_between_people_with_the_same_surname_and_title() -> None:
    first = conversation("I1", 7, [
        ("SPEAKER_00", "me", "두 분 다 와 주셨네요."),
        ("SPEAKER_01", "customer", "생산관리팀 박지훈 대리입니다."),
        ("SPEAKER_03", "customer", "구매팀 박민수 대리입니다."),
    ])
    later = conversation("I2", 14, [
        ("SPEAKER_00", "me", "박 대리님, 견적 보셨어요?"),
        ("SPEAKER_02", "customer", "네, 구매팀에서 검토했습니다."),
    ])
    replies = [
        {"people": [person("SPEAKER_01", name="박지훈", title="대리"), person("SPEAKER_03", name="박민수", title="대리")],
         "facts": [], "resolved_promise_ids": []},
        {"people": [person("SPEAKER_02", surname="박", title="대리", known_person_id="P2")],
         "facts": [], "resolved_promise_ids": []},
    ]

    class Scripted:
        name = "scripted"

        def generate(self, system, prompt, schema):
            return schema.model_validate(replies.pop(0))

    memory = MemoryBuilder(Scripted()).build("누리정밀", [first, later])

    assert {p.name: [c for c, _ in p.appearances] for p in memory.people} == {
        "박지훈": ["I1"], "박민수": ["I1", "I2"]}
    # "박 대리" is said, so the link is not just a guess.
    assert memory.people[1].inferred_in == []


# Two people met in the first meeting; in the second nobody is named.
MET_TWO = conversation("I1", 7, [
    ("SPEAKER_00", "me", "두 분 다 와 주셨네요."),
    ("SPEAKER_01", "customer", "생산관리팀 박지훈 대리입니다. 3개 라인에 쓰려고요."),
    ("SPEAKER_02", "customer", "구매팀 김서연 과장입니다. 견적은 제가 봅니다."),
])
NOBODY_NAMED = conversation("I2", 14, [
    ("SPEAKER_00", "me", "견적 보셨어요?"),
    ("SPEAKER_01", "customer", "네, 구매 쪽에서 검토했어요."),
    ("SPEAKER_02", "customer", "라인은 3개에서 5개로 늘리기로 했어요."),
])


def matched(links: list[dict] | Exception) -> CompanyMemory:
    """The second meeting where the extraction pairs speakers by order (wrongly), then the matching call."""
    from app.schemas.memory import PersonLinks

    replies = [
        {"people": [person("SPEAKER_01", name="박지훈", title="대리"), person("SPEAKER_02", name="김서연", title="과장")],
         "facts": [], "resolved_promise_ids": []},
        {"people": [person("SPEAKER_01", name="박지훈", known_person_id="P1"),
                    person("SPEAKER_02", name="김서연", known_person_id="P2")],
         "facts": [], "resolved_promise_ids": []},
    ]

    class Scripted:
        name = "scripted"

        def generate(self, system, prompt, schema):
            if schema is PersonLinks:
                if isinstance(links, Exception):
                    raise links
                return PersonLinks(links=links)
            return schema.model_validate(replies.pop(0))

    return MemoryBuilder(Scripted(), relink=True).build("누리정밀", [MET_TWO, NOBODY_NAMED])


def link(speaker: str, known: str | None, evidence: list[str]) -> dict:
    return {"speaker": speaker, "reason": "", "known_person_id": known, "evidence_utterance_ids": evidence}


def spoke_as(memory: CompanyMemory) -> dict[str, list]:
    return {p.name or p.title: [s for c, s in p.appearances if c == "I2"] for p in memory.people}


def test_matching_call_overrides_names_paired_by_speaking_order() -> None:
    memory = matched([link("SPEAKER_01", "P2", ["U2"]), link("SPEAKER_02", "P1", ["U3"])])

    assert spoke_as(memory) == {"박지훈": ["SPEAKER_02"], "김서연": ["SPEAKER_01"]}
    assert all(p.inferred_in == ["I2"] for p in memory.people)


def test_unsure_or_unsupported_match_keeps_the_extraction_guess() -> None:
    unsure = matched([link("SPEAKER_01", None, []), link("SPEAKER_02", None, [])])
    # An id without evidence from this conversation counts as unsure.
    no_evidence = matched([link("SPEAKER_01", "P2", []), link("SPEAKER_02", "P1", ["U99"])])

    for memory in (unsure, no_evidence):
        assert spoke_as(memory) == {"박지훈": ["SPEAKER_01"], "김서연": ["SPEAKER_02"]}


def test_failed_matching_call_keeps_the_extraction() -> None:
    from app.core.errors import AppError

    memory = matched(AppError(status_code=502, code="LLM_UNAVAILABLE", message="실패"))

    assert spoke_as(memory) == {"박지훈": ["SPEAKER_01"], "김서연": ["SPEAKER_02"]}
