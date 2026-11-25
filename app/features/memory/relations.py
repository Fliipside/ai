# [기능 처리] 관계 기억 조회, 관계도, 관계 타임라인
# 고객사 기억에서 남은 약속을 찾고, 고객사·사람·대화·핵심 정보를 잇는 관계도(점과 선)와
# 대화별로 무슨 일이 있었는지 날짜순으로 보여 주는 타임라인을 만듭니다.

from collections import defaultdict

from app.features.memory.chunking import KIND_LABELS
from app.features.memory.extraction import person_label, valid_date
from app.schemas.memory import (
    CompanyMemory,
    MemoryFact,
    PromiseOwner,
    Timeline,
    TimelineEntry,
    TimelineEvent,
    TimelinePerson,
    UpcomingPromise,
)

FACT_LABELS = {
    "requirement": "요구사항",
    "price_reaction": "가격 반응",
    "promise": "약속",
    "purchase_timing": "구매 시기",
    "buying_signal": "구매 신호",
    "decision": "결정",
}
ROLE_IN_DEAL_LABELS = {
    "champion": "내부 지지자",
    "influencer": "의견 관계자",
    "decision_maker": "결정권자",
    "procurement": "구매·계약",
    "unknown": "역할 미상",
}


def open_promises(
    memory: CompanyMemory,
    owner: PromiseOwner,
    visible_conversations: set[str] | None = None,
) -> list[MemoryFact]:
    """Promises not yet fulfilled; owner "me" = our to-dos, "customer" = what we are waiting for.

    Used by the evaluation script. Whether a promise is done or replaced is read from the whole
    memory; visible_conversations only hides promises made in conversations the viewer cannot
    see. So a promise fulfilled or changed in a conversation the viewer cannot see is left out
    here. The graph and timeline APIs differ: visible_part shows such a promise as still open.
    """
    return [
        f for f in memory.facts
        if f.type == "promise" and f.status == "open" and f.owner == owner and f.superseded_by is None
        and (visible_conversations is None or f.conversation_id in visible_conversations)
    ]


def visible_part(memory: CompanyMemory) -> CompanyMemory:
    """Only what belongs to the conversations given; backend leaves out ones the viewer may not see."""
    shown = {c.id for c in memory.conversations}
    people = []
    for p in memory.people:
        appearances = [(c, s) for c, s in p.appearances if c in shown]
        if appearances:
            people.append(p.model_copy(update={
                "appearances": appearances, "inferred_in": [c for c in p.inferred_in if c in shown],
            }))
    known = {p.id for p in people}
    # A promise kept in a conversation left out still looks open; that it was kept comes from that conversation.
    facts = [
        f.model_copy(update={"said_by": f.said_by if f.said_by in known or f.said_by == "me" else None,
                             "resolved_in": f.resolved_in if f.resolved_in in shown else None,
                             "status": "open" if f.status == "done" and f.resolved_in not in shown else f.status})
        for f in memory.facts if f.conversation_id in shown
    ]
    ids = {f.id for f in facts}
    for f in facts:
        f.supersedes = f.supersedes if f.supersedes in ids else None
        f.superseded_by = f.superseded_by if f.superseded_by in ids else None
    return memory.model_copy(update={"people": people, "facts": facts})


def is_overdue(fact: MemoryFact, as_of: str | None) -> bool:
    # A due date that is not a real date ("아무때나") counts as no due date.
    due = valid_date(fact.due_at)
    return bool(as_of and fact.type == "promise" and fact.status == "open" and not fact.superseded_by
                and due and due < as_of)


def attendees(memory: CompanyMemory) -> dict[str, list]:
    """Conversation id -> the people who spoke in it, in memory order."""
    by_conversation = defaultdict(list)
    for p in memory.people:
        for conversation_id in dict.fromkeys(c for c, _ in p.appearances):
            by_conversation[conversation_id].append(p)
    return by_conversation


def build_graph(memory: CompanyMemory, as_of: str | None = None) -> dict:
    """Nodes and edges for the relationship map (vis-network shape: from/to, group, label).

    as_of (YYYY-MM-DD) marks overdue promises. Each node also has `type` and plain fields
    (role_in_deal, fact_type, status, due_at) for screens that draw it themselves.
    """
    memory = visible_part(memory)
    nodes: list[dict] = [{"id": "company", "type": "company", "group": "company", "label": memory.company}]
    edges: list[dict] = []
    nodes.append({"id": "me", "type": "me", "group": "me", "label": "나 (영업 담당)"})
    edges.append({"from": "me", "to": "company", "type": "sells_to", "label": "영업"})

    for p in memory.people:
        guessed = " (추정 연결)" if p.inferred_in else ""
        nodes.append({"id": p.id, "type": "person", "group": p.role_in_deal, "role_in_deal": p.role_in_deal,
                      "inferred_in": p.inferred_in,
                      "label": f"{person_label(p)}\n{ROLE_IN_DEAL_LABELS[p.role_in_deal]}{guessed}"})
        edges.append({"from": p.id, "to": "company", "type": "member_of", "label": "소속"})

    spoke_in = attendees(memory)
    for c in sorted(memory.conversations, key=lambda c: c.occurred_at):
        kind = KIND_LABELS.get(c.kind, c.kind)
        nodes.append({"id": c.id, "type": "conversation", "group": "conversation",
                      "label": f"{c.occurred_at.date().isoformat()} {kind}\n{c.title}"})
        edges.append({"from": "me", "to": c.id, "type": "attended", "label": "참석", "dashes": True})
        for p in spoke_in[c.id]:
            edges.append({"from": p.id, "to": c.id, "type": "attended", "label": "참석", "dashes": True})

    for f in memory.facts:
        group = f.type
        if f.superseded_by:
            group = "superseded"
        elif f.type == "promise" and f.status == "open":
            group = "overdue" if is_overdue(f, as_of) else "open_promise"
        label = FACT_LABELS[f.type]
        if f.type == "promise":
            due = f"~{f.due_at}" if f.due_at else (f.due_text or "기한 없음")
            label += f" · {'완료' if f.status == 'done' else '남음'} · {due}"
        nodes.append({"id": f.id, "type": "fact", "group": group, "label": label, "title": f.content,
                      "fact_type": f.type, "status": f.status, "due_at": f.due_at})
        edges.append({"from": f.id, "to": f.conversation_id, "type": "mentioned_in", "label": "언급",
                      "dashes": True})
        if f.said_by:
            edges.append({"from": f.said_by, "to": f.id, "type": "said", "label": "말함"})
        if f.type == "promise" and f.owner == "me" and f.said_by != "me":
            edges.append({"from": "me", "to": f.id, "type": "todo", "label": "할 일"})
        if f.supersedes:
            edges.append({"from": f.id, "to": f.supersedes, "type": "changed", "label": "변경"})
        if f.resolved_in:
            edges.append({"from": f.id, "to": f.resolved_in, "type": "resolved_in", "label": "완료"})
    return {"nodes": nodes, "edges": edges}


def build_timeline(memory: CompanyMemory, as_of: str | None = None) -> Timeline:
    """What happened in each conversation, oldest first, then the promises still open."""
    memory = visible_part(memory)
    people = {p.id: p for p in memory.people}
    facts = {f.id: f for f in memory.facts}
    seen: set[str] = set()

    def who(f: MemoryFact) -> str | None:
        return "나" if f.said_by == "me" else person_label(people[f.said_by]) if f.said_by else None

    spoke_in = attendees(memory)
    made_in, done_in = defaultdict(list), defaultdict(list)
    for f in memory.facts:
        made_in[f.conversation_id].append(f)
        if f.resolved_in:
            done_in[f.resolved_in].append(f)

    entries = []
    for c in sorted(memory.conversations, key=lambda c: c.occurred_at):
        participants = []
        for p in spoke_in[c.id]:
            participants.append(TimelinePerson(id=p.id, label=person_label(p), role_in_deal=p.role_in_deal,
                                               first_seen=p.id not in seen, inferred=c.id in p.inferred_in))
            seen.add(p.id)
        events = []
        for f in made_in[c.id]:
            previous = facts.get(f.supersedes or "")
            events.append(TimelineEvent(
                type="changed" if previous else "promise" if f.type == "promise" else "fact",
                fact_id=f.id, fact_type=f.type, content=f.content, said_by=who(f), owner=f.owner,
                due_at=f.due_at, previous_content=previous.content if previous else None,
            ))
        events += [
            TimelineEvent(type="promise_done", fact_id=f.id, fact_type=f.type, content=f.content, said_by=who(f),
                          owner=f.owner, due_at=f.due_at)
            for f in done_in[c.id]
        ]
        entries.append(TimelineEntry(conversation_id=c.id, occurred_at=c.occurred_at, kind=c.kind, title=c.title,
                                     participants=participants, events=events))

    still_open = [f for f in memory.facts if f.type == "promise" and f.status == "open" and not f.superseded_by]
    # Dated promises first, earliest due first; undated ones keep the order they were made in.
    still_open.sort(key=lambda f: (valid_date(f.due_at) is None, valid_date(f.due_at) or ""))
    upcoming = [
        UpcomingPromise(fact_id=f.id, conversation_id=f.conversation_id, owner=f.owner, content=f.content,
                        due_at=f.due_at, due_text=f.due_text, overdue=is_overdue(f, as_of))
        for f in still_open
    ]
    return Timeline(company=memory.company, as_of=as_of, entries=entries, upcoming=upcoming)
