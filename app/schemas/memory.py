# [데이터 형식] 대화 기억 API의 요청·응답 모양
# 필드 이름, 필수 여부, 길이 제한을 정합니다. backend와 맞춰야 하는 계약입니다.

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from app.schemas.voice import Utterance

InteractionKind = Literal["meeting", "call"]
QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


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


class Conversation(BaseModel):
    id: str
    kind: InteractionKind
    occurred_at: datetime
    title: str
    utterances: list[StoredUtterance]


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
    conversations: list[Conversation]
    people: list[Person]
    facts: list[MemoryFact]
