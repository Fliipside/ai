# [AI 모델] 구조화 출력 LLM의 공통 형태
# 다른 LLM으로 바꿀 때 이 형태(지시문 + 입력 → 정해진 모양의 결과)만 맞추면 됩니다.

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    name: str

    # Returns an instance of `schema`, validated.
    def generate(self, system: str, prompt: str, schema: type[T]) -> T: ...
