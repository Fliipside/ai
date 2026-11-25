# [AI 모델] 재정렬 모델의 공통 형태
# 다른 재정렬 모델로 바꿀 때 이 형태(질문과 글 목록 → 점수 목록)만 맞추면 됩니다.

from typing import Protocol


class Reranker(Protocol):
    name: str

    # One relevance score per text for the query; higher is more relevant.
    def score(self, query: str, texts: list[str]) -> list[float]: ...
