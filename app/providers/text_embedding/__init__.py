# [AI 모델] 텍스트 임베딩 모델 고르기
# 설정(TEXT_EMBEDDING_MODEL)을 보고 사용할 임베딩 모델을 만들어 줍니다.

from app.core.config import Settings
from app.providers.text_embedding.base import TextEmbedder


def create_text_embedder(settings: Settings) -> TextEmbedder:
    from app.providers.text_embedding.sentence_transformers import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(settings.text_embedding_model)
