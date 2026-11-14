# [AI 모델] 텍스트 임베딩 모델의 공통 형태
# 다른 임베딩 모델로 바꿀 때 이 형태(문서 임베딩, 질문 임베딩)만 맞추면 됩니다.

from typing import Protocol


class TextEmbedder(Protocol):
    # Vectors are only comparable when produced by the same name.
    name: str

    # Unit-length vectors, so cosine similarity is a dot product.
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
