# [기능 처리] 대화 기억 API의 실제 작업
# 청크 나누기 → 머리말 붙이기 → 임베딩, 질문 임베딩, 후보 재정렬을 합니다.
# 저장·검색(pgvector)은 하지 않고 backend가 맡습니다.

from app.features.memory.chunking import chunk_utterances, with_context
from app.providers.rerank.base import Reranker
from app.providers.text_embedding.base import TextEmbedder
from app.schemas.memory import (
    ChunkRequest,
    ChunkResult,
    EmbeddedChunk,
    QueryVector,
    RankedCandidate,
    RerankRequest,
    RerankResult,
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
