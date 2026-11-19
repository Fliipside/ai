"""Build relationship memory from fixtures/sales and score it against the gold facts.

For each customer, conversations are extracted in date order (LLM via the gateway),
then compared with gold_facts and the structured gold queries (open promises).
LLM replies are cached under data/memory/extractions/, so reruns cost nothing.
Writes one memory JSON and one relationship map (HTML) per scenario.

    fact recall     gold facts matched by a fact of the same type sharing evidence utterances
    precision       extracted facts that matched a gold fact
    speaker         matched facts attributed to the right person ("me" or the gold contact)
    promise fields  owner / due date / open-done status of matched promises
    supersedes      gold changes (later fact replaces an earlier one) linked to the earlier fact
    people          each customer speaker per conversation: right by full name or surname+title,
                    title only, unknown, or wrong (another person); merged = one person covering
                    several real contacts
    open promises   structured queries answered with exactly the gold promise set

Scenarios S001-S007 are used to tune the prompt; S008-S010 are held out to check it.

Usage:
    python scripts/memory_eval.py --gold                 # no LLM: gold facts -> memory, map
    python scripts/memory_eval.py                        # LLM extraction (needs LLM_API_KEY)
    python scripts/memory_eval.py fixtures/sales/s001_nuri_precision.json --model claude-sonnet-5-5
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.features.memory.extraction import MemoryBuilder, person_label, speaker_label  # noqa: E402
from app.features.memory.relations import (  # noqa: E402
    FACT_LABELS,
    ROLE_IN_DEAL_LABELS,
    build_graph,
    open_promises,
)
from app.providers.llm.gateway import GatewayLLM  # noqa: E402
from app.schemas.memory import CompanyMemory, Conversation, MemoryFact, Person  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "sales"


def conversations(scenario: dict) -> list[Conversation]:
    return [Conversation.model_validate(i) for i in scenario["interactions"]]


def speaker_contacts(scenario: dict) -> dict[tuple[str, str], str]:
    """(conversation id, speaker label) -> gold contact id ("me" for the seller)."""
    votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for i in scenario["interactions"]:
        for u in i["utterances"]:
            votes[(i["id"], speaker_label(u["role"], u["speaker"]))][u["contact_id"] or u["role"]] += 1
    return {key: c.most_common(1)[0][0] for key, c in votes.items()}


def gold_memory(scenario: dict) -> CompanyMemory:
    """The memory a perfect extractor would build, from gold_facts and contacts."""
    appearances = defaultdict(list)
    for (conversation_id, speaker), contact in speaker_contacts(scenario).items():
        appearances[contact].append((conversation_id, speaker))
    people = [
        Person(id=c["id"], name=c["name"], title=c.get("title"), role_in_deal=c.get("role_in_deal", "unknown"),
               appearances=sorted(appearances[c["id"]]))
        for c in scenario["contacts"]
    ]
    superseded_by = {f["supersedes"]: f["id"] for f in scenario["gold_facts"] if f.get("supersedes")}
    facts = [
        MemoryFact(
            id=f["id"], conversation_id=f["interaction_id"], type=f["type"], said_by=f["said_by"],
            content=f["content"], source_utterance_ids=f["source_utterance_ids"], owner=f.get("owner"),
            due_text=f.get("due_text"), due_at=f.get("due_at"), status=f.get("status"),
            resolved_in=f.get("resolved_in"), supersedes=f.get("supersedes"), superseded_by=superseded_by.get(f["id"]),
        )
        for f in scenario["gold_facts"]
    ]
    return CompanyMemory(company=scenario["company"]["name"], conversations=conversations(scenario),
                         people=people, facts=facts)


class CachedLLM:
    """Reuses a saved reply when the same model gets exactly the same request."""

    def __init__(self, llm, cache_dir: Path) -> None:
        self.llm = llm
        self.name = llm.name
        self.cache_dir = cache_dir
        self.calls = 0

    def generate(self, system, prompt, schema):
        key = hashlib.sha256(f"{self.name}\n{system}\n{prompt}\n{schema.__name__}".encode()).hexdigest()[:16]
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return schema.model_validate_json(path.read_text(encoding="utf-8"))
        result = self.llm.generate(system, prompt, schema)
        self.calls += 1
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result


def match_facts(predicted: list[MemoryFact], gold: list[dict]) -> dict[str, str]:
    """Pair predicted and gold facts of one conversation: same type first, then evidence overlap."""
    pairs = []
    for p in predicted:
        for g in gold:
            shared = set(p.source_utterance_ids) & set(g["source_utterance_ids"])
            if shared:
                union = set(p.source_utterance_ids) | set(g["source_utterance_ids"])
                pairs.append(((p.type == g["type"], len(shared) / len(union)), p.id, g["id"]))
    matched, used = {}, set()
    for _, p, g in sorted(pairs, reverse=True):
        if p not in matched and g not in used:
            matched[p] = g
            used.add(g)
    return matched


def score_extraction(scenario: dict, memory: CompanyMemory) -> dict:
    gold = {f["id"]: f for f in scenario["gold_facts"]}
    by_conversation = defaultdict(list)
    for f in scenario["gold_facts"]:
        by_conversation[f["interaction_id"]].append(f)
    contacts = speaker_contacts(scenario)
    speaker_of = {(p.id, c): s for p in memory.people for c, s in p.appearances}

    matched = {}
    for c in memory.conversations:
        predicted = [f for f in memory.facts if f.conversation_id == c.id]
        matched.update(match_facts(predicted, by_conversation[c.id]))
    facts = {f.id: f for f in memory.facts}
    same_type = {p: g for p, g in matched.items() if facts[p].type == gold[g]["type"]}

    def who(f: MemoryFact) -> str | None:
        if f.said_by in (None, "me"):
            return f.said_by
        return contacts.get((f.conversation_id, speaker_of.get((f.said_by, f.conversation_id), "")))

    promises = [(facts[p], gold[g]) for p, g in same_type.items() if gold[g]["type"] == "promise"]
    gold_changes = [g for g in gold.values() if g.get("supersedes")]
    gold_to_pred = {g: p for p, g in matched.items()}
    linked = sum(
        1 for g in gold_changes
        if g["id"] in gold_to_pred
        and facts[gold_to_pred[g["id"]]].supersedes is not None
        and matched.get(facts[gold_to_pred[g["id"]]].supersedes) == g["supersedes"]
    )
    people = score_people(scenario, memory, contacts)
    return {
        "gold": len(gold), "predicted": len(memory.facts), "matched": len(same_type),
        "matched_any_type": len(matched),
        "by_type": Counter(gold[g]["type"] for g in same_type.values()),
        "gold_by_type": Counter(g["type"] for g in gold.values()),
        "speaker_ok": sum(who(facts[p]) == gold[g]["said_by"] for p, g in same_type.items()),
        "promises": len(promises),
        "owner_ok": sum(p.owner == g["owner"] for p, g in promises),
        "due_total": sum(1 for _, g in promises if g.get("due_at")),
        "due_ok": sum(p.due_at == g["due_at"] for p, g in promises if g.get("due_at")),
        "status_ok": sum(p.status == g["status"] for p, g in promises),
        "changes": len(gold_changes), "changes_linked": linked,
        **people,
        "matched_pairs": matched,
    }


def title_matches(predicted: str | None, gold: str | None) -> bool:
    return bool(predicted and gold) and any(word in gold for word in predicted.split())


def score_people(scenario: dict, memory: CompanyMemory, contacts: dict) -> Counter:
    real = {c["id"]: c for c in scenario["contacts"]}
    result = Counter()
    for p in memory.people:
        covered = set()
        for conversation_id, speaker in p.appearances:
            contact = real.get(contacts.get((conversation_id, speaker), ""))
            if contact is None:  # e.g. a worker who is not a listed contact
                continue
            covered.add(contact["id"])
            result["people_seen"] += 1
            surname = p.surname or (p.name or "")[:1]
            if p.name:
                result["people_full" if p.name == contact["name"] else "people_wrong"] += 1
            elif surname:
                ok = contact["name"].startswith(surname) and (not p.title or title_matches(p.title, contact.get("title")))
                result["people_surname" if ok else "people_wrong"] += 1
            elif p.title:
                result["people_title" if title_matches(p.title, contact.get("title")) else "people_wrong"] += 1
            else:
                result["people_unknown"] += 1
        result["people_merged"] += len(covered) > 1
    return result


def visible_conversations(scenario: dict, viewer: int) -> set[str]:
    teams = {u["user_id"]: u["team_id"] for u in scenario["users"]}
    return {
        i["id"] for i in scenario["interactions"]
        if i["owner_user_id"] == viewer
        or (i["visibility"] == "team" and teams[i["owner_user_id"]] == teams[viewer])
    }


def open_promise_query(scenario: dict, query: dict) -> set[str] | None:
    """Whose open promises a structured query asks for, or None if it asks something else."""
    facts = {f["id"]: f for f in scenario["gold_facts"]}
    superseded = {f["supersedes"] for f in scenario["gold_facts"] if f.get("supersedes")}
    expected = [facts[i] for i in query["relevant_fact_ids"]]
    if all(f["type"] == "promise" and f["status"] == "open" and f["id"] not in superseded for f in expected):
        return {f["owner"] for f in expected}
    return None  # e.g. "which promises were kept?"


def score_queries(scenario: dict, memory: CompanyMemory, matched: dict[str, str] | None) -> list[dict]:
    results = []
    for q in scenario["gold_queries"]:
        owners = open_promise_query(scenario, q) if q["kind"] == "structured" else None
        if not owners:
            continue
        visible = visible_conversations(scenario, q["viewer_user_id"])
        found = [f for owner in sorted(owners) for f in open_promises(memory, owner, visible)]
        ids = {matched.get(f.id, f.id) if matched is not None else f.id for f in found}
        expected = set(q["relevant_fact_ids"])
        results.append({"id": q["id"], "question": q["question"], "exact": ids == expected,
                        "found": len(ids & expected), "expected": len(expected), "extra": len(ids - expected),
                        "answer": [f.content for f in found]})
    return results


GROUP_STYLE = {
    "company": ("#1f3a5f", "#ffffff", "box"), "me": ("#2e7d32", "#ffffff", "box"),
    "champion": ("#4a90d9", "#ffffff", "ellipse"), "influencer": ("#7aa7d9", "#ffffff", "ellipse"),
    "decision_maker": ("#c0392b", "#ffffff", "ellipse"), "procurement": ("#8e6bbf", "#ffffff", "ellipse"),
    "unknown": ("#95a5a6", "#ffffff", "ellipse"), "conversation": ("#f4f1ea", "#333333", "box"),
    "requirement": ("#fff4cc", "#333333", "box"), "price_reaction": ("#ffe0cc", "#333333", "box"),
    "promise": ("#e3f2e1", "#333333", "box"), "open_promise": ("#ffd54f", "#333333", "box"),
    "overdue": ("#ef5350", "#ffffff", "box"), "purchase_timing": ("#e1ecf7", "#333333", "box"),
    "buying_signal": ("#d7f0f5", "#333333", "box"), "decision": ("#ece1f5", "#333333", "box"),
    "superseded": ("#eeeeee", "#999999", "box"),
}
LEGEND = [
    ("decision_maker", "결정권자"), ("champion", "내부 지지자"), ("procurement", "구매·계약"),
    ("open_promise", "남은 약속"), ("overdue", "기한 지난 약속"), ("promise", "완료된 약속"),
    ("requirement", "요구사항"), ("price_reaction", "가격 반응"), ("decision", "결정"),
    ("purchase_timing", "구매 시기"), ("buying_signal", "구매 신호"), ("superseded", "바뀐 이전 정보"),
]


def render_html(memory: CompanyMemory, graph: dict, as_of: str, source: str) -> str:
    groups = {
        g: {"color": {"background": bg, "border": "#00000033"}, "font": {"color": fg}, "shape": shape}
        for g, (bg, fg, shape) in GROUP_STYLE.items()
    }
    legend = "".join(
        f'<span class="chip" style="background:{GROUP_STYLE[g][0]};color:{GROUP_STYLE[g][1]}">{label}</span>'
        for g, label in LEGEND
    )

    def items(owner: str) -> str:
        rows = [f"<li>{f.content} <em>({f.due_at or f.due_text or '기한 없음'})</em></li>"
                for f in open_promises(memory, owner)]
        return "".join(rows) or "<li>없음</li>"

    people = "".join(
        f"<li><b>{person_label(p)}</b> · {ROLE_IN_DEAL_LABELS[p.role_in_deal]}"
        f"{' · 일부 대화는 추정 연결' if p.inferred_in else ''}</li>"
        for p in memory.people
    )
    counts = Counter(f.type for f in memory.facts)
    summary = " · ".join(f"{FACT_LABELS[t]} {counts[t]}" for t in FACT_LABELS if counts[t])
    data = json.dumps(graph, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{memory.company} 관계도</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/standalone/umd/vis-network.min.js"></script>
<style>
body {{ margin:0; font-family:-apple-system, 'Apple SD Gothic Neo', sans-serif; background:#fafafa; color:#222; display:flex; height:100vh; }}
#side {{ width:320px; padding:16px; overflow:auto; border-right:1px solid #ddd; background:#fff; box-sizing:border-box; }}
#map {{ flex:1; }}
h1 {{ font-size:20px; margin:0 0 4px; }} h2 {{ font-size:14px; margin:18px 0 6px; }}
.meta {{ color:#777; font-size:12px; }} ul {{ padding-left:18px; margin:0; font-size:13px; line-height:1.5; }}
.chip {{ display:inline-block; font-size:11px; padding:2px 6px; border-radius:4px; margin:2px; }}
#detail {{ font-size:13px; background:#f4f6f8; padding:8px; border-radius:6px; min-height:40px; }}
</style></head><body>
<div id="side">
<h1>{memory.company}</h1>
<div class="meta">기준일 {as_of} · {source} · 대화 {len(memory.conversations)}건 · {summary}</div>
<h2>우리 쪽 남은 할 일</h2><ul>{items("me")}</ul>
<h2>고객에게 받을 것</h2><ul>{items("customer")}</ul>
<h2>사람</h2><ul>{people}</ul>
<h2>선택한 항목</h2><div id="detail">점을 누르면 내용이 보입니다.</div>
<h2>색 구분</h2><div>{legend}</div>
</div>
<div id="map"></div>
<script>
const graph = {data};
const nodes = new vis.DataSet(graph.nodes);
const network = new vis.Network(document.getElementById('map'), {{nodes, edges: new vis.DataSet(graph.edges)}}, {{
  groups: {json.dumps(groups)},
  nodes: {{ font: {{ size: 13, face: 'Apple SD Gothic Neo' }}, margin: 8, widthConstraint: {{ maximum: 200 }} }},
  edges: {{ arrows: 'to', font: {{ size: 10, color: '#888', strokeWidth: 0 }}, color: '#bbbbbb', smooth: true }},
  physics: {{ solver: 'forceAtlas2Based', forceAtlas2Based: {{ gravitationalConstant: -60, springLength: 120 }}, stabilization: {{ iterations: 300 }} }},
}});
network.on('click', e => {{
  const n = e.nodes.length ? nodes.get(e.nodes[0]) : null;
  document.getElementById('detail').textContent = n ? (n.title || n.label) : '점을 누르면 내용이 보입니다.';
}});
</script></body></html>
"""


# Scenarios whose number ends in 8, 9 or 0 are never looked at while tuning the prompt.
TRAIN, HOLDOUT = "개선용 (끝자리 1~7)", "확인용 (끝자리 8·9·0)"


def is_holdout(scenario_id: str) -> bool:
    return scenario_id[-1] in "890"

# Anthropic list prices, USD per 1M tokens (input, output); the gateway may bill differently.
PRICES = {"claude-opus-5-5": (4, 20), "claude-sonnet-5-5": (2, 10), "claude-sonnet": (3, 15),
          "claude-haiku": (1, 5), "claude-fable-5-1": (10, 50)}
CACHED_INPUT_RATE = 0.1
KRW_PER_USD = 1400


def usage_report(model: str, usage: Counter) -> str:
    if not usage["calls"]:
        return ""
    text = (f"토큰: 입력 {usage['prompt']:,} (캐시 {usage['cached']:,}), 출력 {usage['completion']:,} "
            f"(그중 사고 {usage['reasoning']:,}), 재요청 {usage['retries']}건")
    if model in PRICES:
        rate_in, rate_out = PRICES[model]
        fresh = usage["prompt"] - usage["cached"]
        usd = (fresh + usage["cached"] * CACHED_INPUT_RATE) * rate_in / 1e6 + usage["completion"] * rate_out / 1e6
        text += f", 정가 기준 약 {usd * KRW_PER_USD:,.0f}원 (${usd:.2f})"
    return text


def print_report(name: str, group: dict, with_extraction: bool) -> None:
    print(f"\n[{name}]")
    if with_extraction:
        t, by_type, gold_by_type = group["totals"], group["by_type"], group["gold_by_type"]
        print(f"  핵심 정보 재현율 {pct(t['matched'], t['gold'])}  정밀도 {pct(t['matched'], t['predicted'])}  "
              f"(종류 무관 근거 일치 {pct(t['matched_any_type'], t['gold'])})")
        print("  종류별 재현율: " + ", ".join(
            f"{FACT_LABELS[k]} {pct(by_type[k], gold_by_type[k])}" for k in FACT_LABELS))
        print(f"  말한 사람 {pct(t['speaker_ok'], t['matched'])}")
        right = t["people_full"] + t["people_surname"]
        print(f"  고객 알아보기 {pct(right, t['people_seen'])} 맞음 (이름 {t['people_full']}, 성+직함 {t['people_surname']}), "
              f"직함만 {t['people_title']}, 모름 {t['people_unknown']}, 틀림 {t['people_wrong']}, "
              f"잘못 합친 사람 {t['people_merged']}")
        print(f"  약속: 할 사람 {pct(t['owner_ok'], t['promises'])}  기한 날짜 {pct(t['due_ok'], t['due_total'])}  "
              f"완료 여부 {pct(t['status_ok'], t['promises'])}")
        print(f"  바뀐 정보 연결 {pct(t['changes_linked'], t['changes'])}")
    queries = group["queries"]
    exact = sum(q["exact"] for q in queries)
    print(f"  남은 약속 질문 {pct(exact, len(queries))} 정확히 일치  "
          f"(찾음 {sum(q['found'] for q in queries)}/{sum(q['expected'] for q in queries)}, "
          f"불필요 {sum(q['extra'] for q in queries)}개)")
    for q in queries:
        if not q["exact"]:
            print(f"    ✗ {q['id']} {q['question']} → {q['answer']}")


def pct(a: int, b: int) -> str:
    return f"{a / b:.1%} ({a}/{b})" if b else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixtures", nargs="*", type=Path, help="scenario JSON files (default: all)")
    parser.add_argument("--gold", action="store_true", help="build memory from gold facts instead of the LLM")
    parser.add_argument("--model", help="gateway model id (default: LLM_MODEL)")
    parser.add_argument("--effort", choices=["low", "medium", "high"],
                        help="reasoning effort sent to the gateway (default: LLM_REASONING_EFFORT)")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "memory")
    args = parser.parse_args()

    paths = args.fixtures or sorted(FIXTURES.glob("*.json"))
    scenarios = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    llm = None
    if not args.gold:
        settings = get_settings()
        model = args.model or settings.llm_model
        effort = args.effort or settings.llm_reasoning_effort
        gateway = GatewayLLM(settings.llm_api_key.get_secret_value(), model, settings.llm_base_url, effort)
        llm = CachedLLM(gateway, args.out / "extractions" / gateway.name.split(":", 1)[1].replace(":", "-"))
    label = "gold" if args.gold else llm.name.split(":", 1)[1].replace(":", "-")
    out_dir = args.out / label
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = {name: {"totals": Counter(), "by_type": Counter(), "gold_by_type": Counter(), "queries": []}
              for name in (TRAIN, HOLDOUT)}
    for scenario in scenarios:
        sid = scenario["scenario_id"]
        group = groups[HOLDOUT if is_holdout(sid) else TRAIN]
        if args.gold:
            memory, matched = gold_memory(scenario), None
        else:
            memory = MemoryBuilder(llm).build(scenario["company"]["name"], conversations(scenario))
            score = score_extraction(scenario, memory)
            matched = score.pop("matched_pairs")
            group["by_type"] += score.pop("by_type")
            group["gold_by_type"] += score.pop("gold_by_type")
            group["totals"] += Counter(score)
            print(f"{sid} {memory.company}: 추출 {score['predicted']}개, 정답 {score['gold']}개 중 "
                  f"{score['matched']}개 일치")
        group["queries"] += score_queries(scenario, memory, matched)
        (out_dir / f"{sid}.json").write_text(memory.model_dump_json(indent=2), encoding="utf-8")
        graph = build_graph(memory, scenario["as_of"])
        (out_dir / f"{sid}.html").write_text(
            render_html(memory, graph, scenario["as_of"], "정답 데이터" if args.gold else label), encoding="utf-8")

    if llm is not None:
        print(f"\n{label}: 새로 호출 {llm.calls}건  {usage_report(model, llm.llm.usage)}")
    for name, group in groups.items():
        if group["queries"] or group["totals"]:
            print_report(name, group, with_extraction=not args.gold)
    print(f"\n결과: {out_dir}/ (S00x.json 기억, S00x.html 관계도)")


if __name__ == "__main__":
    main()
