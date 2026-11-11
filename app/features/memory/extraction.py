# [기능 처리] 대화에서 핵심 정보 추출 + 고객사별 관계 기억 쌓기
# 대화를 날짜순으로 하나씩 LLM에 보내 요구사항·가격 반응·약속·구매 시기·구매 신호·결정을 뽑고,
# 사람(화자) 연결, 바뀐 정보(supersedes), 지켜진 약속(resolved)을 이전 기억에 반영합니다.

from datetime import date
from pathlib import Path

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

PROMPTS = Path(__file__).parent / "prompts"

# General rules only; no names or sentences from the evaluation scenarios.
SYSTEM_PROMPT = (PROMPTS / "extraction_system.txt").read_text(encoding="utf-8").strip()


def format_date(value: date) -> str:
    return f"{value.isoformat()} ({value.strftime('%a')})"


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
