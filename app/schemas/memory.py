# [데이터 형식] 대화 기억 API의 요청·응답 모양
# 필드 이름, 필수 여부, 길이 제한을 정합니다. backend와 맞춰야 하는 계약입니다.

from datetime import date, datetime
from zoneinfo import ZoneInfo
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.schemas.voice import Utterance

InteractionKind = Literal["meeting", "call"]
QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
KST = ZoneInfo("Asia/Seoul")


def korean_time_if_naive(value: datetime) -> datetime:
    # Conversations are sorted by time; one without an offset cannot be compared with one that has
    # it, so a missing offset is read as Korean time, where the recordings are made.
    return value.replace(tzinfo=KST) if value.tzinfo is None else value


ConversationTime = Annotated[datetime, AfterValidator(korean_time_if_naive)]


class StoredUtterance(Utterance):
    # Stable id so chunks can point back to the original speech and time.
    id: str


class MemoryChunk(BaseModel):
    text: str
    start: float
    end: float
    utterance_ids: list[str]


class InteractionContext(BaseModel):
    # Becomes the "회사 | 날짜 미팅/통화 | 제목" header embedded with every chunk.
    # occurred_at keeps the offset it was sent with; the header uses that local date.
    company: str = Field(min_length=1)
    kind: InteractionKind
    occurred_at: datetime
    title: str


class ChunkRequest(BaseModel):
    utterances: list[StoredUtterance] = Field(max_length=5000)
    context: InteractionContext | None = None


class EmbeddedChunk(MemoryChunk):
    # Text that was embedded: `text` with the context header, if any.
    embed_text: str
    vector: list[float]


class ChunkResult(BaseModel):
    # Store with the vectors; only vectors from the same model are comparable.
    model: str
    chunks: list[EmbeddedChunk]


class QueryRequest(BaseModel):
    text: QueryText


class QueryVector(BaseModel):
    model: str
    vector: list[float]


class RerankCandidate(BaseModel):
    id: str
    # Send the chunk's embed_text so the reranker also sees the context header.
    text: str


class RerankRequest(BaseModel):
    query: QueryText
    candidates: list[RerankCandidate] = Field(max_length=50)
    top_k: int | None = Field(default=None, ge=1)


class RankedCandidate(BaseModel):
    id: str
    score: float


class RerankResult(BaseModel):
    model: str
    results: list[RankedCandidate]


# Key information extraction (one conversation -> facts) and relationship memory.

FactType = Literal["requirement", "price_reaction", "promise", "purchase_timing", "buying_signal", "decision"]
DealRole = Literal["champion", "influencer", "decision_maker", "procurement", "unknown"]
PromiseOwner = Literal["me", "customer"]


class ConversationRef(BaseModel):
    # A conversation without its utterances: enough to name it in the prompt and the map.
    id: str = Field(min_length=1)
    kind: InteractionKind
    occurred_at: ConversationTime
    title: str


class Conversation(ConversationRef):
    utterances: list[StoredUtterance] = Field(min_length=1, max_length=5000)

    @model_validator(mode="after")
    def utterance_ids_are_unique(self) -> Self:
        # Facts point back to utterances by id, so an id must name one utterance.
        if len({u.id for u in self.utterances}) != len(self.utterances):
            raise ValueError("utterance ids must be unique within a conversation")
        return self


# What the LLM returns for one conversation. Every field is required (null when unknown)
# so the structured output schema stays strict.
class ExtractedPerson(BaseModel):
    speaker: str  # label in this conversation, e.g. "A" or "SPEAKER_01"
    known_person_id: str | None  # an already known person this speaker is, judged from context
    name: str | None  # full name, only when said
    surname: str | None  # family name alone, e.g. "박" from "박 대리님"
    title: str | None
    role_in_deal: DealRole


class ExtractedFact(BaseModel):
    type: FactType
    speaker: str  # "me" or a customer speaker label
    content: str
    source_utterance_ids: list[str]
    owner: PromiseOwner | None  # promise only: who must act
    due_text: str | None  # promise only: the deadline as said
    due_date: str | None  # promise only: YYYY-MM-DD resolved from the conversation date
    supersedes_fact_id: str | None  # an earlier fact this one changes


class ConversationExtraction(BaseModel):
    people: list[ExtractedPerson]
    facts: list[ExtractedFact]
    resolved_promise_ids: list[str]  # earlier open promises fulfilled in this conversation


class Person(BaseModel):
    id: str
    name: str | None
    surname: str | None = None
    title: str | None
    role_in_deal: DealRole
    # (conversation id, speaker label) pairs this person spoke as
    appearances: list[tuple[str, str]]
    # conversations where this person was recognized from context rather than by name
    inferred_in: list[str] = []


class MemoryFact(BaseModel):
    id: str
    conversation_id: str
    type: FactType
    said_by: str | None  # "me", a person id, or None when the speaker is not identified
    content: str
    source_utterance_ids: list[str]
    owner: PromiseOwner | None = None
    due_text: str | None = None
    due_at: str | None = None
    status: Literal["open", "done"] | None = None  # promise only
    resolved_in: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None


class CompanyMemory(BaseModel):
    company: str
    # Conversations already applied; earlier ones may come without utterances (from MemoryState).
    conversations: list[ConversationRef]
    people: list[Person]
    facts: list[MemoryFact]


# POST /memory/extract: backend keeps the memory and sends it with each new conversation.

class MemoryState(BaseModel):
    # Everything stored so far for this customer company; empty for its first conversation.
    conversations: list[ConversationRef] = Field(default=[], max_length=1000)
    people: list[Person] = Field(default=[], max_length=500)
    facts: list[MemoryFact] = Field(default=[], max_length=5000)

    @model_validator(mode="after")
    def ids_are_unique(self) -> Self:
        # Each id gets one short alias in the prompt; a repeated id would make it ambiguous.
        for name, items in (("conversations", self.conversations), ("people", self.people), ("facts", self.facts)):
            if len({i.id for i in items}) != len(items):
                raise ValueError(f"{name} ids must be unique")
        return self


class ExtractRequest(BaseModel):
    company: str = Field(min_length=1)
    conversation: Conversation
    memory: MemoryState = MemoryState()


class ExtractResult(BaseModel):
    model: str
    # Upsert both by id: new ones plus earlier ones this conversation changed
    # (a person seen again, a promise now done, a fact replaced by a newer one).
    people: list[Person]
    facts: list[MemoryFact]


# POST /memory/graph and /memory/timeline: drawn from the stored memory, no LLM call.

class RelationRequest(BaseModel):
    company: str = Field(min_length=1)
    # Send only what the viewer may see; facts of conversations left out are dropped.
    memory: MemoryState
    # Promises due before this day are marked overdue; today in Korea when left out.
    as_of: date | None = None


GraphNodeType = Literal["company", "me", "person", "conversation", "fact"]
GraphEdgeType = Literal["sells_to", "member_of", "attended", "mentioned_in", "said", "todo", "changed", "resolved_in"]


class GraphNode(BaseModel):
    id: str
    type: GraphNodeType
    # Style group: a role for people, a fact type or superseded / open_promise / overdue for facts.
    group: str
    label: str
    title: str | None = None  # full text of a fact
    role_in_deal: DealRole | None = None
    # person only: conversations where this person was linked by a guess, not by a name said there
    inferred_in: list[str] | None = None
    fact_type: FactType | None = None
    status: Literal["open", "done"] | None = None
    due_at: str | None = None


class GraphEdge(BaseModel):
    # vis-network names the ends from/to; `from` is a Python keyword, hence the alias.
    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(alias="from")
    to: str
    type: GraphEdgeType
    label: str
    dashes: bool = False


class RelationGraph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class TimelinePerson(BaseModel):
    id: str
    label: str
    role_in_deal: DealRole
    first_seen: bool  # this is the first conversation this person appears in
    inferred: bool = False  # linked to the earlier person by a guess here; worth a check by the user


class TimelineEvent(BaseModel):
    # fact: new information, promise: a new promise, changed: replaces an earlier fact
    # (previous_content), promise_done: an earlier promise fulfilled in this conversation.
    type: Literal["fact", "promise", "changed", "promise_done"]
    fact_id: str
    fact_type: FactType
    content: str
    said_by: str | None  # "나", a person's label, or None when not identified
    owner: PromiseOwner | None = None
    due_at: str | None = None
    previous_content: str | None = None


class TimelineEntry(BaseModel):
    conversation_id: str
    occurred_at: datetime
    kind: InteractionKind
    title: str
    participants: list[TimelinePerson]
    events: list[TimelineEvent]


class UpcomingPromise(BaseModel):
    fact_id: str
    conversation_id: str
    owner: PromiseOwner | None
    content: str
    due_at: str | None
    due_text: str | None
    overdue: bool


class Timeline(BaseModel):
    company: str
    as_of: str | None
    entries: list[TimelineEntry]  # oldest conversation first
    upcoming: list[UpcomingPromise]  # open promises, earliest due first, undated last
