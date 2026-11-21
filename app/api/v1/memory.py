# [API 입구] 대화 기억 API (/api/v1/memory/...)
# backend 요청을 받아 service.py로 넘기고 결과를 돌려줍니다.
# /chunks: 대화 → 청크+벡터, /embed-query: 질문 → 벡터, /rerank: 검색 후보 재정렬
# /extract: 대화 1건 + 지금까지의 기억 → 새로 생기거나 바뀐 사람·핵심 정보
# /graph, /timeline: 저장된 기억 → 관계도(점과 선), 대화별 타임라인 (LLM 없음)

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.features.memory.service import ExtractionService, MemoryService, relation_graph, relation_timeline
from app.providers.llm import create_llm
from app.providers.rerank import create_reranker
from app.providers.text_embedding import create_text_embedder
from app.schemas.memory import (
    ChunkRequest,
    ChunkResult,
    ExtractRequest,
    ExtractResult,
    QueryRequest,
    QueryVector,
    RelationGraph,
    RelationRequest,
    RerankRequest,
    RerankResult,
    Timeline,
)
from app.schemas.response import ApiResponse

router = APIRouter()


def get_memory_service(request: Request) -> MemoryService:
    # Models load once per process on first use (cached in the providers).
    settings = request.app.state.settings
    return MemoryService(create_text_embedder(settings), create_reranker(settings))


def get_extraction_service(request: Request) -> ExtractionService:
    # One gateway client per process; it is created on the first extraction.
    state = request.app.state
    if getattr(state, "llm", None) is None:
        state.llm = create_llm(state.settings)
    return ExtractionService(state.llm)


Service = Annotated[MemoryService, Depends(get_memory_service)]
Extraction = Annotated[ExtractionService, Depends(get_extraction_service)]


@router.post("/chunks", response_model=ApiResponse[ChunkResult])
async def chunks(body: ChunkRequest, service: Service) -> ApiResponse[ChunkResult]:
    # One interaction's utterances -> chunks with vectors for backend to store in pgvector.
    return ApiResponse[ChunkResult].ok(await asyncio.to_thread(service.chunk, body))


@router.post("/embed-query", response_model=ApiResponse[QueryVector])
async def embed_query(body: QueryRequest, service: Service) -> ApiResponse[QueryVector]:
    # Question -> vector for backend's pgvector search.
    return ApiResponse[QueryVector].ok(await asyncio.to_thread(service.embed_query, body.text))


@router.post("/rerank", response_model=ApiResponse[RerankResult])
async def rerank(body: RerankRequest, service: Service) -> ApiResponse[RerankResult]:
    # Reorder backend's top pgvector matches, best first.
    return ApiResponse[RerankResult].ok(await asyncio.to_thread(service.rerank, body))


@router.post("/extract", response_model=ApiResponse[ExtractResult])
async def extract(body: ExtractRequest, service: Extraction) -> ApiResponse[ExtractResult]:
    # One LLM call, usually 10-60 seconds; backend should run it after saving the analysis.
    return ApiResponse[ExtractResult].ok(await asyncio.to_thread(service.extract, body))


@router.post("/graph", response_model=ApiResponse[RelationGraph])
async def graph(body: RelationRequest) -> ApiResponse[RelationGraph]:
    # Stored memory -> nodes and edges for the relationship map. Up to about a second for the
    # largest memory, so it runs off the event loop like the other endpoints.
    return ApiResponse[RelationGraph].ok(await asyncio.to_thread(relation_graph, body))


@router.post("/timeline", response_model=ApiResponse[Timeline])
async def timeline(body: RelationRequest) -> ApiResponse[Timeline]:
    # Stored memory -> what happened in each conversation, then the promises still open.
    return ApiResponse[Timeline].ok(await asyncio.to_thread(relation_timeline, body))
