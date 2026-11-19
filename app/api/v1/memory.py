# [API 입구] 대화 기억 API (/api/v1/memory/...)
# backend 요청을 받아 service.py로 넘기고 결과를 돌려줍니다.
# /chunks: 대화 → 청크+벡터, /embed-query: 질문 → 벡터, /rerank: 검색 후보 재정렬

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.features.memory.service import MemoryService
from app.providers.rerank import create_reranker
from app.providers.text_embedding import create_text_embedder
from app.schemas.memory import (
    ChunkRequest,
    ChunkResult,
    QueryRequest,
    QueryVector,
    RerankRequest,
    RerankResult,
)
from app.schemas.response import ApiResponse

router = APIRouter()


def get_memory_service(request: Request) -> MemoryService:
    # Models load once per process on first use (cached in the providers).
    settings = request.app.state.settings
    return MemoryService(create_text_embedder(settings), create_reranker(settings))


Service = Annotated[MemoryService, Depends(get_memory_service)]


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
