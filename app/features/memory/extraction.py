# [기능 처리] 대화에서 핵심 정보 추출 + 고객사별 관계 기억 쌓기
# 대화를 날짜순으로 하나씩 LLM에 보내 요구사항·가격 반응·약속·구매 시기·구매 신호·결정을 뽑고,
# 사람(화자) 연결, 바뀐 정보(supersedes), 지켜진 약속(resolved)을 이전 기억에 반영합니다.

import logging
import re
from datetime import date
from pathlib import Path

from app.core.errors import AppError
from app.features.memory.chunking import KIND_LABELS, ROLE_LABELS
from app.providers.llm.base import StructuredLLM
from app.schemas.memory import (
    CompanyMemory,
    Conversation,
    ConversationExtraction,
    ExtractedPerson,
    MemoryFact,
    Person,
    PersonLinks,
)

logger = logging.getLogger(__name__)

PROMPTS = Path(__file__).parent / "prompts"

# General rules only; no names or sentences from the evaluation scenarios.
SYSTEM_PROMPT = (PROMPTS / "extraction_system.txt").read_text(encoding="utf-8").strip()
LINKING_PROMPT = (PROMPTS / "linking_system.txt").read_text(encoding="utf-8").strip()


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


class PromptIds:
    """Short ids shown to the LLM (P1, F1, U1) in place of real ids.

    Real ids can be long (e.g. "<analysis job uuid>-0001"); copying them costs tokens, and a
    single wrong character makes apply_extraction drop the fact.
    """

    def __init__(self, memory: CompanyMemory, conversation: Conversation) -> None:
        self.short: dict[str, str] = {}
        for prefix, ids in (
            ("P", [p.id for p in memory.people]),
            ("F", [f.id for f in memory.facts]),
            ("U", [u.id for u in conversation.utterances]),
        ):
            self.short.update({real: f"{prefix}{n}" for n, real in enumerate(ids, start=1)})
        self.real = {short: real for real, short in self.short.items()}

    def restore(self, value: str | None) -> str | None:
        # Unknown values pass through; apply_extraction drops ids that point at nothing.
        return self.real.get(value, value) if value else value

    def restore_extraction(self, extraction: ConversationExtraction) -> ConversationExtraction:
        return extraction.model_copy(update={
            "people": [
                p.model_copy(update={"known_person_id": self.restore(p.known_person_id)})
                for p in extraction.people
            ],
            "facts": [
                f.model_copy(update={
                    "source_utterance_ids": [self.restore(i) for i in f.source_utterance_ids],
                    "supersedes_fact_id": self.restore(f.supersedes_fact_id),
                })
                for f in extraction.facts
            ],
            "resolved_promise_ids": [self.restore(i) for i in extraction.resolved_promise_ids],
        })


def prompt_memory(memory: CompanyMemory) -> tuple[list[MemoryFact], list[MemoryFact]]:
    """What the prompt shows from earlier facts: open promises and current (not replaced) other facts."""
    open_promises = [f for f in memory.facts if f.type == "promise" and f.status == "open" and not f.superseded_by]
    current = [f for f in memory.facts if f.type != "promise" and f.superseded_by is None]
    return open_promises, current


def render_prompt(company: str, conversation: Conversation, memory: CompanyMemory, ids: PromptIds) -> str:
    when = format_date(conversation.occurred_at.date())
    lines = [
        f"고객사: {company}",
        f"대화: {when} {KIND_LABELS.get(conversation.kind, conversation.kind)} \"{conversation.title}\"",
        "",
        "지금까지 알려진 고객 쪽 사람 (id 이름·직함, 역할, 나온 대화):",
    ]
    titles = {c.id: c.title for c in memory.conversations}
    lines += [
        f"- {ids.short[p.id]} {person_label(p)}, {p.role_in_deal}, "
        + ", ".join(titles.get(c, c) for c in dict.fromkeys(c for c, _ in p.appearances))
        for p in memory.people
    ] or ["- 없음"]

    # Who said each earlier fact, so a speaker who goes on with it can be recognized without a name.
    def said(f: MemoryFact) -> str:
        return "me" if f.said_by == "me" else ids.short.get(f.said_by or "", "고객")

    open_promises, current = prompt_memory(memory)
    lines += ["", "아직 지켜지지 않은 약속 (id [할 사람] 말한 사람: 내용):"]
    lines += [
        f"- {ids.short[f.id]} [{f.owner}] {said(f)}: {f.content} (기한 {f.due_at or f.due_text or '없음'})"
        for f in open_promises
    ] or ["- 없음"]

    lines += ["", "이전에 기록된 정보 (id [종류] 말한 사람: 내용):"]
    lines += [f"- {ids.short[f.id]} [{f.type}] {said(f)}: {f.content}" for f in current] or ["- 없음"]

    lines += ["", "대화 (발화 id | 화자 라벨(역할): 내용):"]
    lines += [
        f"[{ids.short[u.id]}] {speaker_label(u.role, u.speaker)}({ROLE_LABELS[u.role]}): {u.text}"
        for u in conversation.utterances
    ]
    return "\n".join(lines)


# What the matching prompt shows of each known person; their latest facts, so the prompt stays
# bounded for long-running customers (no evaluation scenario has more than 16).
LINK_FACTS_PER_PERSON = 20


def render_link_prompt(company: str, conversation: Conversation, memory: CompanyMemory, ids: PromptIds) -> str:
    """Known people with what each said before, then this conversation, for matching speakers only."""
    when = format_date(conversation.occurred_at.date())
    titles = {c.id: f"{format_date(c.occurred_at.date())} \"{c.title}\"" for c in memory.conversations}
    lines = [f"고객사: {company}", "", "이전 대화에서 만난 고객 쪽 사람:"]
    for p in memory.people:
        seen = ", ".join(titles.get(c, c) for c in dict.fromkeys(c for c, _ in p.appearances))
        lines.append(f"- {ids.short[p.id]} {person_label(p)}, {p.role_in_deal}, 나온 대화: {seen}")
        for f in [f for f in memory.facts if f.said_by == p.id][-LINK_FACTS_PER_PERSON:]:
            status = f" ({'완료' if f.status == 'done' else '남음'})" if f.type == "promise" else ""
            lines.append(f"  · [{f.type}] {f.content}{status}")
    lines += ["", f"이번 대화: {when} {KIND_LABELS.get(conversation.kind, conversation.kind)} \"{conversation.title}\"",
              "(발화 id | 화자 라벨(역할): 내용)"]
    lines += [
        f"[{ids.short[u.id]}] {speaker_label(u.role, u.speaker)}({ROLE_LABELS[u.role]}): {u.text}"
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

    def __init__(self, llm: StructuredLLM, relink: bool = False) -> None:
        self.llm = llm
        # A second call that only matches speakers to known people (see relink()).
        self.relink = relink

    def build(self, company: str, conversations: list[Conversation]) -> CompanyMemory:
        memory = CompanyMemory(company=company, conversations=[], people=[], facts=[])
        for conversation in sorted(conversations, key=lambda c: c.occurred_at):
            self.add(memory, conversation)
        return memory

    def add(self, memory: CompanyMemory, conversation: Conversation) -> None:
        """Extract one conversation with the memory so far and apply it to that memory."""
        ids = PromptIds(memory, conversation)
        extraction = self.llm.generate(
            SYSTEM_PROMPT, render_prompt(memory.company, conversation, memory, ids), ConversationExtraction
        )
        extraction = ids.restore_extraction(extraction)
        if self.relink and memory.people:
            extraction = self.linked(memory, conversation, extraction, ids)
        apply_extraction(memory, conversation, extraction)

    def linked(self, memory: CompanyMemory, conversation: Conversation, extraction: ConversationExtraction,
               ids: PromptIds) -> ConversationExtraction:
        """The extraction with known_person_id decided by a call that does nothing but match people.

        The extraction call also pairs speakers with known people, but alongside everything else it
        often skips them or pairs them by speaking order. A name said in this conversation still wins.
        """
        try:
            links = self.llm.generate(
                LINKING_PROMPT, render_link_prompt(memory.company, conversation, memory, ids), PersonLinks
            )
        except AppError as exc:
            # The extraction already has its own matching; losing this check is better than losing it all.
            logger.warning("Person matching failed, keeping the extraction's own: %s", exc.message)
            return extraction
        speakers = {speaker_label(u.role, u.speaker) for u in conversation.utterances}
        utterance_ids = {u.id for u in conversation.utterances}
        known = {p.id for p in memory.people}
        decided: dict[str, str | None] = {}
        for link in links.links:
            label = link.speaker.strip()
            label = label if label in speakers else ROLE_SUFFIX.sub("", label)
            person_id = ids.restore(link.known_person_id)
            evidence = [e for e in map(ids.restore, link.evidence_utterance_ids) if e in utterance_ids]
            decided[label] = person_id if person_id in known and evidence else None

        by_id = {p.id: p for p in memory.people}
        people, kept = [], []
        for p in with_own_labels(extraction, speakers).people:
            chosen = by_id.get(decided.get(p.speaker) or "")
            # Not sure here means no better answer: the extraction's own guess stays (it is right
            # more often than not). A name said in this conversation also stays, unless only the
            # speaker says it and it is another known person's ("김서연 과장님께 전달했습니다").
            name = full_name(p)
            if chosen is None or name and said_of(conversation, p.speaker, name, clean(p.name),
                                                  fits=not any(k.name == name and k is not chosen
                                                               for k in memory.people)):
                kept.append(p)
                continue
            # A name or surname not said for this speaker was copied from a guess; drop it so the
            # speaker is matched only by the decision above. The title stays even when not said: it
            # is often guessed right from what the speaker does, and one that disagrees with the
            # person picked keeps a wrong pick from merging two people (S086 in the eval).
            surname, title = clean(p.surname), clean(p.title)
            said_surname = surname and title and said_of(
                conversation, p.speaker, surname + title.split()[-1],
                fits=surname == chosen.surname or (chosen.name or "").startswith(surname)
                or not (chosen.surname or chosen.name))
            people.append(p.model_copy(update={
                "known_person_id": chosen.id, "name": None, "surname": surname if said_surname else None,
            }))
        # Matched speakers first, so a kept guess cannot take a person a matched speaker is.
        return extraction.model_copy(update={"people": people + kept})


def full_name(extracted: ExtractedPerson) -> str | None:
    """The name with the surname in front; models sometimes give only the given name ("지훈" + "박")."""
    name, surname = clean(extracted.name), clean(extracted.surname)
    if name and surname and len(name) <= 2 and not name.startswith(surname):
        return surname + name
    return name


# The prompt asks for what someone does ("구매 담당") when no title is said, so such a duty
# can sit next to any real title and never rules a person out. Two real titles must agree.
# "책임자" and not "책임": 책임 and 책임연구원 are real titles.
DUTY_WORDS = ("담당", "실무", "책임자")


def is_duty(title: str) -> bool:
    return any(w in title for w in DUTY_WORDS)


def titles_agree(a: str | None, b: str | None) -> bool:
    if not a or not b or a in b or b in a:
        return True
    return is_duty(a) or is_duty(b)


def named_in(conversation: Conversation, *names: str | None, not_by: str | None = None) -> bool:
    """Whether any of these names is said in the conversation (spaces ignored), leaving out what
    the speaker `not_by` says."""
    said = (u.text for u in conversation.utterances if speaker_label(u.role, u.speaker) != not_by)
    text = "".join([conversation.title, *said]).replace(" ", "")
    return any(n and n.replace(" ", "") in text for n in names)


def said_of(conversation: Conversation, speaker: str, *names: str | None, fits: bool) -> bool:
    """Whether a name is said for this speaker: by someone else (calling them by it), or by the
    speaker themselves when it fits the person matched (introducing themselves, not someone else)."""
    return named_in(conversation, *names, not_by=speaker) or fits and named_in(conversation, *names)


def find_person(memory: CompanyMemory, conversation: Conversation, extracted: ExtractedPerson,
                taken: set[str]) -> tuple[Person | None, bool]:
    """The known person this speaker is, and whether that was only inferred from context."""
    people = {p.id: p for p in memory.people if p.id not in taken}
    name, title = full_name(extracted), clean(extracted.title)
    surname = clean(extracted.surname) or (name[:1] if name else None)
    if name:
        for p in people.values():
            if p.name == name:
                # Models also copy a known person's name onto a speaker they only guess is that
                # person; a name this conversation never says is such a guess, so mark it inferred.
                return p, not named_in(conversation, name, clean(extracted.name))
    known = people.get(extracted.known_person_id or "")
    if surname:
        candidates = [
            p for p in people.values()
            if (p.surname == surname or (p.name or "").startswith(surname))
            and not (name and p.name and p.name != name)
            and titles_agree(title, p.title)
        ]
        if candidates:
            # Two people can share a surname and title (two 박 대리); the model's pick decides, and
            # without one the speaker stays apart: merging two people is worse than not linking.
            if known in candidates:
                p = known
            elif len(candidates) == 1:
                [p] = candidates
            else:
                return None, False
            # Same for a surname: it counts when said with a title ("박 대리") or in the full name.
            titles = [t.split()[-1] for t in (title, p.title) if t]
            return p, not named_in(conversation, name, p.name, *(surname + t for t in titles))
    # A context-only link needs something to link on (a name, a title or what the person said
    # before, which the prompt shows) and nothing that contradicts it; merging two real people
    # is worse than leaving a speaker unnamed.
    if (
        known is not None
        and (known.name or known.surname or known.title or any(f.said_by == known.id for f in memory.facts))
        and not (name and known.name and known.name != name)
        and not (surname and known.surname and known.surname != surname)
        and titles_agree(title, known.title)
    ):
        return known, True
    return None, False


# The prompt shows speakers as "A(고객)"; with short labels the model sometimes copies the role too.
ROLE_SUFFIX = re.compile(r"\s*\([^()]*\)$")


def with_own_labels(extraction: ConversationExtraction, speakers: set[str]) -> ConversationExtraction:
    def own(label: str) -> str:
        label = label.strip()
        return label if label in speakers else ROLE_SUFFIX.sub("", label)

    return extraction.model_copy(update={
        "people": [p.model_copy(update={"speaker": own(p.speaker)}) for p in extraction.people],
        "facts": [f.model_copy(update={"speaker": own(f.speaker)}) for f in extraction.facts],
    })


def unused_id(memory: CompanyMemory, prefix: str, n: int, width: int = 1) -> str:
    """`{prefix}{n}`, or the next number free of every stored id (backend may send any ids)."""
    used = {i.id for items in (memory.conversations, memory.people, memory.facts) for i in items}
    while f"{prefix}{n:0{width}d}" in used:
        n += 1
    return f"{prefix}{n:0{width}d}"


def apply_extraction(memory: CompanyMemory, conversation: Conversation, extraction: ConversationExtraction) -> None:
    """Add one conversation's extraction, dropping anything that does not point at real data."""
    # A conversation may come after later ones (backend retrying a failed one). What it says can
    # then change or settle only what was said before it; a newer fact stays the current one.
    occurred = {c.id: c.occurred_at for c in memory.conversations}
    not_newer = {f.id for f in memory.facts if occurred.get(f.conversation_id, conversation.occurred_at)
                 <= conversation.occurred_at}
    memory.conversations.append(conversation)
    utterance_ids = {u.id for u in conversation.utterances}
    speakers = {speaker_label(u.role, u.speaker) for u in conversation.utterances}
    extraction = with_own_labels(extraction, speakers)

    person_of_speaker: dict[str, str] = {}
    for extracted in extraction.people:
        if extracted.speaker not in speakers or extracted.speaker == "me" or extracted.speaker in person_of_speaker:
            continue
        person, inferred = find_person(memory, conversation, extracted, set(person_of_speaker.values()))
        if person is None:
            # Scoped to the conversation so it never collides with ids backend already stores.
            person = Person(id=unused_id(memory, f"{conversation.id}-P", len(person_of_speaker) + 1),
                            name=None, title=None,
                            role_in_deal=extracted.role_in_deal, appearances=[])
            memory.people.append(person)
        person.name = person.name or full_name(extracted)
        person.surname = person.surname or clean(extracted.surname) or (person.name or "")[:1] or None
        title = clean(extracted.title)
        # A duty guessed from what was said never replaces a title someone actually used.
        if title and not (person.title and is_duty(title) and not is_duty(person.title)):
            person.title = title
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
            id=unused_id(memory, f"{conversation.id}-F", n, width=2),
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
            if previous.id in not_newer:
                fact.supersedes = previous.id
                previous.superseded_by = fact.id
            else:
                # The prompt shows only current facts, so a late conversation can only be matched
                # with the newer fact on the same point; that one replaced this, not the reverse.
                fact.superseded_by = previous.id
                previous.supersedes = previous.supersedes or fact.id
        memory.facts.append(fact)

    for fact_id in extraction.resolved_promise_ids:
        promise = facts.get(fact_id)
        if promise is not None and promise.id in not_newer and promise.type == "promise" and promise.status == "open":
            promise.status = "done"
            promise.resolved_in = conversation.id
