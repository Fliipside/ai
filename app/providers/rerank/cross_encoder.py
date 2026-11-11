# [AI 모델] 재정렬 실제 구현 (기본 bge-reranker-v2-m3)
# 질문과 후보 글을 함께 읽고 관련도 점수를 매깁니다. 모델은 처음 한 번만 불러옵니다.

import logging
import threading
from functools import lru_cache

from app.core.errors import AppError

logger = logging.getLogger(__name__)

BGE_RERANKER = "BAAI/bge-reranker-v2-m3"

score_lock = threading.Lock()


class CrossEncoderReranker:
    """Reads the query and each text together, so it catches matches worded differently."""

    def __init__(self, model: str = BGE_RERANKER, batch_size: int = 16) -> None:
        self.model = model
        self.batch_size = batch_size
        self.name = f"cross_encoder:{model}"

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        model = load_model(self.model)
        with score_lock:
            scores = model.predict([(query, t) for t in texts], batch_size=self.batch_size)
        return [float(s) for s in scores]


@lru_cache
def load_model(model: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise AppError(
            status_code=503,
            code="RERANK_NOT_CONFIGURED",
            message="sentence-transformers가 설치되지 않았습니다.",
        ) from exc
    try:
        return CrossEncoder(model)
    except Exception as exc:
        logger.error("Failed to load rerank model %s: %s", model, exc)
        raise AppError(
            status_code=503,
            code="RERANK_NOT_CONFIGURED",
            message="재정렬 모델을 불러올 수 없습니다.",
        ) from exc
