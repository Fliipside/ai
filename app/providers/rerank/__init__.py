# [AI 모델] 재정렬 모델 고르기
# 설정(RERANK_MODEL)을 보고 사용할 재정렬 모델을 만들어 줍니다.

from app.core.config import Settings
from app.providers.rerank.base import Reranker


def create_reranker(settings: Settings) -> Reranker:
    from app.providers.rerank.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker(settings.rerank_model)
