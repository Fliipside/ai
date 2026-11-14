# [기능 처리] 관계 기억 조회와 관계도 데이터
# 고객사 기억에서 남은 약속을 찾고, 고객사·사람·대화·핵심 정보를 잇는 관계도(점과 선)를 만듭니다.

from app.features.memory.chunking import KIND_LABELS
from app.features.memory.extraction import person_label
from app.schemas.memory import CompanyMemory, MemoryFact, PromiseOwner

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
    """Promises not yet fulfilled; owner "me" = our to-dos, "customer" = what we are waiting for."""
    return [
        f for f in memory.facts
        if f.type == "promise" and f.status == "open" and f.owner == owner and f.superseded_by is None
        and (visible_conversations is None or f.conversation_id in visible_conversations)
    ]


def build_graph(memory: CompanyMemory, as_of: str | None = None) -> dict:
    """Nodes and edges for the relationship map. as_of (YYYY-MM-DD) marks overdue promises."""
    nodes: list[dict] = [{"id": "company", "group": "company", "label": memory.company}]
    edges: list[dict] = []
    nodes.append({"id": "me", "group": "me", "label": "나 (영업 담당)"})
    edges.append({"from": "me", "to": "company", "label": "영업"})

    for p in memory.people:
        guessed = " (추정 연결)" if p.inferred_in else ""
        nodes.append({"id": p.id, "group": p.role_in_deal,
                      "label": f"{person_label(p)}\n{ROLE_IN_DEAL_LABELS[p.role_in_deal]}{guessed}"})
        edges.append({"from": p.id, "to": "company", "label": "소속"})

    for c in sorted(memory.conversations, key=lambda c: c.occurred_at):
        kind = KIND_LABELS.get(c.kind, c.kind)
        nodes.append({"id": c.id, "group": "conversation",
                      "label": f"{c.occurred_at.date().isoformat()} {kind}\n{c.title}"})
        edges.append({"from": "me", "to": c.id, "label": "참석", "dashes": True})
        for p in memory.people:
            if any(conversation_id == c.id for conversation_id, _ in p.appearances):
                edges.append({"from": p.id, "to": c.id, "label": "참석", "dashes": True})

    for f in memory.facts:
        group = f.type
        if f.superseded_by:
            group = "superseded"
        elif f.type == "promise" and f.status == "open":
            group = "overdue" if as_of and f.due_at and f.due_at < as_of else "open_promise"
        label = FACT_LABELS[f.type]
        if f.type == "promise":
            due = f"~{f.due_at}" if f.due_at else (f.due_text or "기한 없음")
            label += f" · {'완료' if f.status == 'done' else '남음'} · {due}"
        nodes.append({"id": f.id, "group": group, "label": label, "title": f.content})
        edges.append({"from": f.id, "to": f.conversation_id, "label": "언급", "dashes": True})
        if f.said_by:
            edges.append({"from": f.said_by, "to": f.id, "label": "말함"})
        if f.type == "promise" and f.owner == "me" and f.said_by != "me":
            edges.append({"from": "me", "to": f.id, "label": "할 일"})
        if f.supersedes:
            edges.append({"from": f.id, "to": f.supersedes, "label": "변경"})
        if f.resolved_in:
            edges.append({"from": f.id, "to": f.resolved_in, "label": "완료"})
    return {"nodes": nodes, "edges": edges}
