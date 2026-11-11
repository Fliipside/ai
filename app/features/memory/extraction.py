# [기능 처리] 대화에서 핵심 정보 추출 + 고객사별 관계 기억 쌓기
# 대화를 날짜순으로 하나씩 LLM에 보내 요구사항·가격 반응·약속·구매 시기·구매 신호·결정을 뽑고,
# 사람(화자) 연결, 바뀐 정보(supersedes), 지켜진 약속(resolved)을 이전 기억에 반영합니다.

from datetime import date

from app.features.memory.chunking import KIND_LABELS, ROLE_LABELS
from app.providers.llm.base import StructuredLLM
from app.schemas.memory import (
    CompanyMemory,
    Conversation,
    ConversationExtraction,
    ExtractedPerson,
    MemoryFact,
    Person,
)

WEEKDAYS = "월화수목금토일"

# Examples are from an invented office-furniture deal, not from the evaluation scenarios.
SYSTEM_PROMPT = """너는 B2B 영업 대화를 기록하는 비서다. 영업 담당자("나")와 고객사 사람들의 대화 1건을 읽고, 나중에 영업 담당자가 다시 찾아볼 핵심 정보를 뽑는다.

정보 종류(type)와 예시:
- requirement: 고객이 겪는 문제, 필요한 기능·조건, 도입 범위·수량, 제약
  예) "지금은 의자 수리 요청을 종이로 받아 누락이 잦다", "30개 층 중 본사 5개 층부터 적용한다"
- price_reaction: 가격·견적·할인·비용에 대한 고객의 반응이나 요구
  예) "1인당 월 2만 원은 비싸다며 10% 할인을 원했다"
- promise: 이 대화 뒤에 누군가 해야 하는 구체적인 후속 행동. owner는 할 사람(me = 영업 담당자, customer = 고객)
  해당: 자료·견적·문서·샘플·데이터를 보내기, 확인하고 답하기(다음 만남 때 알려 주기 포함), 검토 후 다시 연락하기, 기능 수정·반영하기, 일정을 확정해 알려 주기, 날짜가 정해진 실행(시범 운영 시작, 납품·입고, 설치), 다음 자리에 특정 사람을 참석시키기
  고객 쪽 약속도 빠뜨리지 않는다: "보내 드릴게요", "다시 연락드릴게요", "공유드릴게요", "들어오시라고 할게요"
  해당 아님: 다음 미팅·방문 날짜를 잡은 것 자체, 제품이 원래 제공하는 서비스나 계약 조건 설명(→ requirement 또는 decision), 고객이 자기 내부에서만 할 일, 이 대화 안에서 바로 끝난 일
  한 발화가 약속이면서 다른 종류이기도 하면(예: "5천 개로 확정, 11월 18일까지 입고") 둘 다 쓴다.
- purchase_timing: 결정·예산 편성·결재·계약·도입·설치 시점
  예) "내년 예산은 12월 이사회에서 확정된다"
- buying_signal: 구매 의지를 보여 주는 말이나 행동, 제품·자료에 대한 좋은 평가, 내부 보고·확산 의사
  예) "시범 사용 후 직원 만족도가 높아 다른 지점에도 쓰고 싶다고 했다"
- decision: 결정 과정 — 누가 결정하는지, 판단 기준, 경쟁사 비교, 결재·계약 경로, 이미 결정된 사항
  예) "최종 결정은 총무이사가 하며 유지보수 비용을 가장 중요하게 본다", "경쟁사 A와 비교 중이다"
  requirement와 헷갈리면: 고객이 "필요하다/원한다"고 한 것은 requirement, "누가/무엇을 기준으로/어떻게 정했다·정한다"는 decision이다.

규칙:
- 대화에 실제로 나온 내용만 쓴다. 추측하지 않는다.
- content는 한국어 한 문장으로, 숫자·이름·제품명을 그대로 살려 원문 없이도 이해되게 쓴다. 화자 라벨(SPEAKER_01, A 등)이나 "(나)"는 쓰지 않는다. 고객은 이름·직함으로 부르고, 영업 담당자가 할 일은 주어 없이 쓴다(예: "견적서를 금요일까지 보낸다").
- source_utterance_ids에는 근거가 되는 발화 id를 대화에 적힌 그대로 넣는다.
- speaker는 그 정보를 말한 화자 라벨이다("me" 또는 고객 화자 라벨). 약속은 약속한 사람의 라벨이다.
- 같은 사실을 여러 번 쓰지 않는다. 같은 화자의 같은 주제는 하나로 묶는다.
- promise가 아니면 owner, due_text, due_date는 null이다. promise의 due_text는 말한 그대로("이번 주 목요일까지"), due_date는 대화 날짜와 요일을 기준으로 계산한 YYYY-MM-DD이며 기한이 없거나 계산할 수 없으면 null이다.

이전 기억과 잇기:
- "아직 지켜지지 않은 약속"을 하나씩 확인한다. 이번 대화에서 그 일이 끝났다는 말이 있거나(받았다, 봤다, 보내 주신 ~, 확인해 보니 ~), 이번 대화 자체가 그 일의 이행이면(예: 데모하기로 한 약속 → 이번 대화가 그 데모) resolved_promise_ids에 그 id를 넣는다.
- 아직 안 끝난 이전 약속이 이번 대화에서 다시 나오거나 기한·내용이 구체화되면, 새 promise를 쓰고 supersedes_fact_id에 이전 약속 id를 넣는다. 그대로 반복만 됐으면 새로 쓰지 않는다.
- "이전에 기록된 정보"와 같은 주제의 값·범위·조건이 이번 대화에서 바뀌었으면(예: 3개 층 → 5개 층, 구매 → 리스) 새 정보의 supersedes_fact_id에 이전 정보 id를 넣는다. 종류는 달라도 된다. 단순히 덧붙이거나 다시 확인한 것이면 null이다.

사람(people):
- 이번 대화의 고객 쪽 화자마다 하나씩 쓴다.
- 아는 만큼만 쓴다. 이름 전체(성 포함)가 나오면 name, 성만 나오면 surname(예: "박 대리님" → surname "박", title "대리"), 직함만 알면 title만 쓴다.
- 이름·직함은 그 화자 본인을 가리키는 단서만 쓴다: 자기소개("구매팀 한소라입니다"), 다른 사람이 그 화자를 부르는 말("상무님, …" 다음에 그 화자가 답함), 대화 제목("현장소장 동석", "상무 보고 미팅").
- 화자가 말 속에서 다른 사람을 언급한 이름·직함은 그 화자의 것이 아니다. 예: SPEAKER_01이 "파일럿은 박 대리가 챙길게요"라고 하면 SPEAKER_01은 박 대리가 아니다. "신 매니저님이 고르시고 발주는 제가"라고 말한 화자는 신 매니저가 아니다.
- 직함이 안 나오면 맡은 일이나 소속을 title에 쓴다(예: "현장소장", "대표", "구매 담당", "IT 담당"). 무엇도 알 수 없을 때만 null이다.
- "지금까지 알려진 고객 쪽 사람" 중 같은 사람이라는 근거가 있을 때만 known_person_id에 그 id를 넣는다. 근거: 같은 이름·성, 이전 대화에서 이번 자리에 오기로 한 사람, 이전에 그 사람이 맡은 일(약속·요청)을 이어서 말함. 같은 회사에는 같은 팀·비슷한 일을 하는 사람이 여럿이므로, 소속이나 맡은 일이 비슷하다는 것만으로는 연결하지 않는다. 근거가 없으면 null이다. 그럴듯하다는 이유만으로 이름을 지어내지 않는다.
- 역할: champion(내부에서 도입을 밀어 주는 실무자), influencer(의견을 내는 관계자), decision_maker(최종 결정권자), procurement(구매·계약 담당), unknown"""


def format_date(value: date) -> str:
    return f"{value.isoformat()} ({WEEKDAYS[value.weekday()]})"


def speaker_label(role: str, speaker: str) -> str:
    return "me" if role == "me" else speaker


def person_label(person: Person) -> str:
    if person.name:
        return f"{person.name} {person.title or ''}".strip()
    if person.surname and person.title:
        return f"{person.surname} {person.title}"
    return f"{person.title or '고객'}(이름 미상)"


def render_prompt(company: str, conversation: Conversation, memory: CompanyMemory) -> str:
    when = format_date(conversation.occurred_at.date())
    lines = [
        f"고객사: {company}",
        f"대화: {when} {KIND_LABELS.get(conversation.kind, conversation.kind)} \"{conversation.title}\"",
        "",
        "지금까지 알려진 고객 쪽 사람 (id 이름·직함, 역할, 나온 대화):",
    ]
    titles = {c.id: c.title for c in memory.conversations}
    lines += [
        f"- {p.id} {person_label(p)}, {p.role_in_deal}, "
        + ", ".join(titles[c] for c in dict.fromkeys(c for c, _ in p.appearances))
        for p in memory.people
    ] or ["- 없음"]

    open_promises = [f for f in memory.facts if f.type == "promise" and f.status == "open" and not f.superseded_by]
    lines += ["", "아직 지켜지지 않은 약속:"]
    lines += [
        f"- {f.id} [{f.owner}] {f.content} (기한 {f.due_at or f.due_text or '없음'})" for f in open_promises
    ] or ["- 없음"]

    current = [f for f in memory.facts if f.type != "promise" and f.superseded_by is None]
    lines += ["", "이전에 기록된 정보:"]
    lines += [f"- {f.id} [{f.type}] {f.content}" for f in current] or ["- 없음"]

    lines += ["", "대화 (발화 id | 화자 라벨(역할): 내용):"]
    lines += [
        f"[{u.id}] {speaker_label(u.role, u.speaker)}({ROLE_LABELS[u.role]}): {u.text}"
        for u in conversation.utterances
    ]
    return "\n".join(lines)


def valid_date(value: str | None) -> str | None:
    try:
        return date.fromisoformat(value).isoformat() if value else None
    except ValueError:
        return None


def clean(value: str | None) -> str | None:
    return (value or "").strip() or None


class MemoryBuilder:
    """Builds one company's memory by extracting its conversations in date order."""

    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def build(self, company: str, conversations: list[Conversation]) -> CompanyMemory:
        memory = CompanyMemory(company=company, conversations=[], people=[], facts=[])
        for conversation in sorted(conversations, key=lambda c: c.occurred_at):
            extraction = self.llm.generate(
                SYSTEM_PROMPT, render_prompt(company, conversation, memory), ConversationExtraction
            )
            apply_extraction(memory, conversation, extraction)
        return memory


def full_name(extracted: ExtractedPerson) -> str | None:
    """The name with the surname in front; models sometimes give only the given name ("지훈" + "박")."""
    name, surname = clean(extracted.name), clean(extracted.surname)
    if name and surname and len(name) <= 2 and not name.startswith(surname):
        return surname + name
    return name


def find_person(memory: CompanyMemory, extracted: ExtractedPerson, taken: set[str]) -> tuple[Person | None, bool]:
    """The known person this speaker is, and whether that was only inferred from context."""
    people = {p.id: p for p in memory.people if p.id not in taken}
    name, title = full_name(extracted), clean(extracted.title)
    surname = clean(extracted.surname) or (name[:1] if name else None)
    if name:
        for p in people.values():
            if p.name == name:
                return p, False
    if surname:
        for p in people.values():
            if (
                (p.surname == surname or (p.name or "").startswith(surname))
                and not (name and p.name and p.name != name)
                and (not title or not p.title or title in p.title or p.title in title)
            ):
                return p, False
    # A context-only link needs something to link on, and nothing that contradicts it;
    # merging two real people is worse than leaving a speaker unnamed.
    known = people.get(extracted.known_person_id or "")
    if (
        known is not None
        and (known.name or known.surname or known.title)
        and not (name and known.name and known.name != name)
        and not (surname and known.surname and known.surname != surname)
        and not (title and known.title and title not in known.title and known.title not in title)
    ):
        return known, True
    return None, False


def apply_extraction(memory: CompanyMemory, conversation: Conversation, extraction: ConversationExtraction) -> None:
    """Add one conversation's extraction, dropping anything that does not point at real data."""
    memory.conversations.append(conversation)
    utterance_ids = {u.id for u in conversation.utterances}
    speakers = {speaker_label(u.role, u.speaker) for u in conversation.utterances}

    person_of_speaker: dict[str, str] = {}
    for extracted in extraction.people:
        if extracted.speaker not in speakers or extracted.speaker == "me" or extracted.speaker in person_of_speaker:
            continue
        person, inferred = find_person(memory, extracted, set(person_of_speaker.values()))
        if person is None:
            person = Person(id=f"P{len(memory.people) + 1}", name=None, title=None,
                            role_in_deal=extracted.role_in_deal, appearances=[])
            memory.people.append(person)
        person.name = person.name or full_name(extracted)
        person.surname = person.surname or clean(extracted.surname) or (person.name or "")[:1] or None
        person.title = clean(extracted.title) or person.title
        if extracted.role_in_deal != "unknown":
            person.role_in_deal = extracted.role_in_deal
        person.appearances.append((conversation.id, extracted.speaker))
        if inferred:
            person.inferred_in.append(conversation.id)
        person_of_speaker[extracted.speaker] = person.id

    facts = {f.id: f for f in memory.facts}
    for n, extracted in enumerate(extraction.facts, start=1):
        sources = [i for i in extracted.source_utterance_ids if i in utterance_ids]
        if not sources:
            continue
        said_by = "me" if extracted.speaker == "me" else person_of_speaker.get(extracted.speaker)
        fact = MemoryFact(
            id=f"{conversation.id}-F{n:02d}",
            conversation_id=conversation.id,
            type=extracted.type,
            said_by=said_by,
            content=extracted.content,
            source_utterance_ids=sources,
        )
        if extracted.type == "promise":
            fact.owner = extracted.owner or ("me" if extracted.speaker == "me" else "customer")
            fact.due_text = extracted.due_text
            fact.due_at = valid_date(extracted.due_date)
            fact.status = "open"
        previous = facts.get(extracted.supersedes_fact_id or "")
        if previous is not None and previous.superseded_by is None:
            fact.supersedes = previous.id
            previous.superseded_by = fact.id
        memory.facts.append(fact)

    for fact_id in extraction.resolved_promise_ids:
        promise = facts.get(fact_id)
        if promise is not None and promise.type == "promise" and promise.status == "open":
            promise.status = "done"
            promise.resolved_in = conversation.id
