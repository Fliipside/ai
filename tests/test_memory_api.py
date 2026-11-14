# [테스트] 대화 기억 API 테스트
# 가짜 모델로 정상 응답, 잘못된 요청(422), 모델 없음(503)을 빠르게 확인합니다.

import pytest
from fastapi.testclient import TestClient

from app.api.v1.memory import get_memory_service
from app.core.errors import AppError
from app.features.memory.service import MemoryService
from app.main import create_app


class FakeEmbedder:
    name = "fake:embedder"

    def __init__(self) -> None:
        self.documents: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents += texts
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.0]


class FakeReranker:
    name = "fake:reranker"

    def score(self, query: str, texts: list[str]) -> list[float]:
        # Count of query characters in the text.
        return [float(sum(ch in t for ch in query)) for t in texts]


class BrokenEmbedder(FakeEmbedder):
    def embed_query(self, text: str) -> list[float]:
        raise AppError(status_code=503, code="EMBEDDING_NOT_CONFIGURED", message="없음")


def utterance(i: int, text: str, role: str = "customer") -> dict:
    return {"id": f"I1-U{i}", "start": i * 10.0, "end": i * 10.0 + 5, "speaker": "S", "role": role,
            "text": text}


CONTEXT = {"company": "누리정밀", "kind": "meeting", "occurred_at": "2026-09-08T10:30:00+09:00",
           "title": "가격 협의"}


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def client(embedder) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_memory_service] = lambda: MemoryService(embedder, FakeReranker())
    return TestClient(app)


def test_chunks_embed_text_with_context_header(client, embedder) -> None:
    body = {"utterances": [utterance(0, "안녕하세요", "me"), utterance(1, "견적이 부담돼요")],
            "context": CONTEXT}

    response = client.post("/api/v1/memory/chunks", json=body)

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["model"] == "fake:embedder"
    [chunk] = data["chunks"]
    assert chunk["text"] == "[나] 안녕하세요\n[고객] 견적이 부담돼요"
    assert chunk["embed_text"] == "누리정밀 | 2026-09-08 미팅 | 가격 협의\n" + chunk["text"]
    assert chunk["utterance_ids"] == ["I1-U0", "I1-U1"]
    assert (chunk["start"], chunk["end"]) == (0.0, 15.0)
    assert chunk["vector"] == [float(len(chunk["embed_text"])), 1.0]
    assert embedder.documents == [chunk["embed_text"]]


def test_chunks_split_long_conversation_with_overlap(client) -> None:
    body = {"utterances": [utterance(i, "가" * 200) for i in range(5)]}

    chunks = client.post("/api/v1/memory/chunks", json=body).json()["data"]["chunks"]

    assert [c["utterance_ids"] for c in chunks] == [
        ["I1-U0", "I1-U1"], ["I1-U1", "I1-U2"], ["I1-U2", "I1-U3"], ["I1-U3", "I1-U4"],
    ]
    assert all(c["embed_text"] == c["text"] for c in chunks)


def test_chunks_empty_conversation(client, embedder) -> None:
    response = client.post("/api/v1/memory/chunks", json={"utterances": []})

    assert response.json()["data"] == {"model": "fake:embedder", "chunks": []}
    assert embedder.documents == []


@pytest.mark.parametrize("body", [
    {"utterances": [utterance(0, "네") | {"role": "boss"}]},
    {"utterances": [utterance(0, "네")], "context": CONTEXT | {"kind": "email"}},
    {"utterances": [{"start": 0, "end": 1, "speaker": "S", "role": "me", "text": "id 없음"}]},
])
def test_chunks_rejects_invalid_request(client, body) -> None:
    response = client.post("/api/v1/memory/chunks", json=body)

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_REQUEST"


def test_embed_query(client) -> None:
    response = client.post("/api/v1/memory/embed-query", json={"text": "  가격 반응은?  "})

    assert response.json()["data"] == {"model": "fake:embedder", "vector": [7.0, 0.0]}


@pytest.mark.parametrize("text", ["", "   ", "가" * 1001])
def test_embed_query_rejects_blank_or_long_text(client, text) -> None:
    assert client.post("/api/v1/memory/embed-query", json={"text": text}).status_code == 422


def test_rerank_orders_best_first_and_keeps_top_k(client) -> None:
    body = {
        "query": "가격",
        "candidates": [{"id": "a", "text": "인사"}, {"id": "b", "text": "가격 협의"},
                       {"id": "c", "text": "가"}],
        "top_k": 2,
    }

    data = client.post("/api/v1/memory/rerank", json=body).json()["data"]

    assert data["model"] == "fake:reranker"
    assert data["results"] == [{"id": "b", "score": 2.0}, {"id": "c", "score": 1.0}]


def test_rerank_without_top_k_returns_all(client) -> None:
    body = {"query": "가", "candidates": [{"id": "a", "text": "나"}, {"id": "b", "text": "가"}]}

    results = client.post("/api/v1/memory/rerank", json=body).json()["data"]["results"]

    assert [r["id"] for r in results] == ["b", "a"]


@pytest.mark.parametrize("body", [
    {"query": "가", "candidates": [], "top_k": 0},
    {"query": "가", "candidates": [{"id": str(i), "text": "가"} for i in range(51)]},
])
def test_rerank_rejects_invalid_request(client, body) -> None:
    assert client.post("/api/v1/memory/rerank", json=body).status_code == 422


def test_model_not_available_returns_503() -> None:
    app = create_app()
    app.dependency_overrides[get_memory_service] = lambda: MemoryService(BrokenEmbedder(), FakeReranker())

    response = TestClient(app).post("/api/v1/memory/embed-query", json={"text": "가격"})

    assert response.status_code == 503
    assert response.json()["code"] == "EMBEDDING_NOT_CONFIGURED"
