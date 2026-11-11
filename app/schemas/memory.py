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

