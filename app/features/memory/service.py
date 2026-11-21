# [기능 처리] 대화 기억 API의 실제 작업
# 청크 나누기 → 머리말 붙이기 → 임베딩, 질문 임베딩, 후보 재정렬, 대화 1건의 핵심 정보 추출,
# 저장된 기억으로 관계도·타임라인 만들기를 합니다.
# 저장·검색(pgvector)은 하지 않고 backend가 맡습니다.

from datetime import date, datetime
from zoneinfo import ZoneInfo

from app.core.errors import AppError
from app.features.memory.chunking import chunk_utterances, with_context
from app.features.memory.extraction import MemoryBuilder, prompt_memory
from app.features.memory.relations import build_graph, build_timeline
from app.providers.llm.base import StructuredLLM
from app.providers.rerank.base import Reranker
from app.providers.text_embedding.base import TextEmbedder
from app.schemas.memory import (
    ChunkRequest,
    ChunkResult,
    CompanyMemory,
    EmbeddedChunk,
    ExtractRequest,
    ExtractResult,
    QueryVector,
    RankedCandidate,
    RelationGraph,
    RelationRequest,
    RerankRequest,
    RerankResult,
    Timeline,
)


class MemoryService:
    """Chunks and vectors for conversation memory search; backend stores and queries pgvector."""

    def __init__(self, embedder: TextEmbedder, reranker: Reranker) -> None:
        self.embedder = embedder
        self.reranker = reranker

    def chunk(self, request: ChunkRequest) -> ChunkResult:
        chunks = chunk_utterances(request.utterances)
        context = request.context
        embed_texts = [
            with_context(
                c.text,
                context.company,
                context.kind,
                context.occurred_at.isoformat(),
                context.title,
            ) if context else c.text
            for c in chunks
        ]
        vectors = self.embedder.embed_documents(embed_texts)
        return ChunkResult(
            model=self.embedder.name,
            chunks=[
                EmbeddedChunk(**c.model_dump(), embed_text=t, vector=v)
                for c, t, v in zip(chunks, embed_texts, vectors)
            ],
        )

    def embed_query(self, text: str) -> QueryVector:
        return QueryVector(model=self.embedder.name, vector=self.embedder.embed_query(text))

    def rerank(self, request: RerankRequest) -> RerankResult:
        candidates = request.candidates
        scores = self.reranker.score(request.query, [c.text for c in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return RerankResult(
            model=self.reranker.name,
            results=[RankedCandidate(id=c.id, score=s) for c, s in ranked[:request.top_k]],
        )


# People, open promises and current facts are all written into every prompt. Twenty
# conversations with one customer come to about 200 lines; past this the call gets slow and costly.
MAX_PROMPT_MEMORY_LINES = 600


class ExtractionService:
    """One conversation + the stored memory -> what to upsert; backend stores the memory."""

    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def extract(self, request: ExtractRequest) -> ExtractResult:
        state = request.memory
        if request.conversation.id in {c.id for c in state.conversations}:
            # Running it again would add every fact twice.
            raise AppError(status_code=409, code="CONVERSATION_ALREADY_EXTRACTED",
                           message="이미 정보를 추출한 대화입니다.")
        memory = CompanyMemory(
            company=request.company,
            conversations=list(state.conversations),
            people=[p.model_copy(deep=True) for p in state.people],
            facts=[f.model_copy(deep=True) for f in state.facts],
        )
        open_promises, current = prompt_memory(memory)
        if len(memory.people) + len(open_promises) + len(current) > MAX_PROMPT_MEMORY_LINES:
            raise AppError(status_code=413, code="MEMORY_TOO_LARGE",
                           message=f"이 고객사의 기억이 너무 커서 한 번에 보낼 수 없습니다 "
                                   f"(사람·남은 약속·현재 정보 합계 {MAX_PROMPT_MEMORY_LINES}개까지).")
        people_before = {p.id: p.model_dump() for p in memory.people}
        facts_before = {f.id: f.model_dump() for f in memory.facts}
        # relink: a second call that only matches speakers to known people (fewer wrong merges).
        MemoryBuilder(self.llm, relink=True).add(memory, request.conversation)
        return ExtractResult(
            model=self.llm.name,
            people=[p for p in memory.people if people_before.get(p.id) != p.model_dump()],
            facts=[f for f in memory.facts if facts_before.get(f.id) != f.model_dump()],
        )


def stored_memory(request: RelationRequest) -> CompanyMemory:
    state = request.memory
    return CompanyMemory(company=request.company, conversations=state.conversations, people=state.people,
                         facts=state.facts)


def as_of_day(request: RelationRequest) -> str:
    # Due dates are Korean calendar days.
    day: date = request.as_of or datetime.now(ZoneInfo("Asia/Seoul")).date()
    return day.isoformat()


def relation_graph(request: RelationRequest) -> RelationGraph:
    return RelationGraph.model_validate(build_graph(stored_memory(request), as_of_day(request)))


def relation_timeline(request: RelationRequest) -> Timeline:
    return build_timeline(stored_memory(request), as_of_day(request))
