# [AI 모델] 대회 지원 LLM 게이트웨이 (OpenAI 호환, 기본 claude-sonnet-5-5 + 사고 low)
# 지시문과 대화를 보내고, 정해진 모양(JSON)대로 검증된 결과를 받습니다. 토큰 사용량도 셉니다.

import json
import logging
import re
from collections import Counter
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.core.errors import AppError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

GATEWAY_URL = "https://gateway.aitestbed.kr/v1"
CLAUDE_SONNET = "claude-sonnet-5-5"
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def compact_schema(schema: type[BaseModel]) -> str:
    """A short JSON-like outline of the schema; much smaller than the full JSON Schema."""
    full = schema.model_json_schema()
    defs = full.get("$defs", {})

    def outline(node: dict) -> str:
        if "$ref" in node:
            return outline(defs[node["$ref"].split("/")[-1]])
        if "anyOf" in node:
            return "|".join(outline(n) for n in node["anyOf"])
        if "enum" in node:
            return "|".join(json.dumps(v, ensure_ascii=False) for v in node["enum"])
        kind = node.get("type")
        if kind == "object":
            return "{" + ", ".join(f'"{k}": {outline(v)}' for k, v in node.get("properties", {}).items()) + "}"
        if kind == "array":
            return "[" + outline(node.get("items", {})) + ", ...]"
        return {"string": "문자열", "integer": "정수", "number": "숫자", "boolean": "true|false"}.get(kind, kind or "값")

    return outline(full)


class GatewayLLM:
    def __init__(self, api_key: str, model: str = CLAUDE_SONNET, base_url: str = GATEWAY_URL,
                 reasoning_effort: str | None = "low", attempts: int = 2, max_tokens: int = 4096) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.attempts = attempts
        # The JSON reply is about 1.5k tokens. Some replies kept generating after the JSON until
        # the limit, and every one of those tokens is billed, so keep the limit close to need.
        self.max_tokens = max_tokens
        self.name = f"gateway:{model}" + (f":{reasoning_effort}" if reasoning_effort else "")
        # prompt / completion / reasoning / cached tokens, calls, retries
        self.usage: Counter = Counter()
        self._client = None

    def generate(self, system: str, prompt: str, schema: type[T]) -> T:
        client = self._get_client()
        # The shape is written into the instructions and validated here instead of using
        # response_format: JSON mode made models behind the gateway run to the output limit.
        instructions = (
            f"{system}\n\n결과는 아래 모양의 JSON 객체 하나로만, 공백·줄바꿈 없이 한 줄로 답한다. "
            f"모든 키를 쓰고 모르는 값은 null이다.\n{compact_schema(schema)}"
        )
        messages = [{"role": "system", "content": instructions}, {"role": "user", "content": prompt}]
        error = ""
        for attempt in range(self.attempts):
            text = self._complete(client, messages)
            try:
                return schema.model_validate_json(FENCE.sub("", text.strip()))
            except ValidationError as exc:
                error = str(exc)[:1000]
                logger.warning("LLM reply did not match %s: %s", schema.__name__, error)
                self.usage["retries"] += 1
                messages += [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"모양과 맞지 않습니다. 오류를 고쳐 JSON만 다시 답하세요.\n{error}"},
                ]
        raise AppError(status_code=502, code="LLM_BAD_RESPONSE",
                       message=f"LLM 응답이 형식에 맞지 않습니다: {error[:200]}")

    def _get_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AppError(status_code=503, code="LLM_NOT_CONFIGURED",
                           message="openai 패키지가 설치되지 않았습니다.") from exc
        if not self.api_key:
            raise AppError(status_code=503, code="LLM_NOT_CONFIGURED",
                           message="LLM_API_KEY가 설정되지 않았습니다.")
        if self._client is None:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=300)
        return self._client

    def _complete(self, client, messages: list[dict]) -> str:
        from openai import APIError

        extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                **extra,
            )
        except APIError as exc:
            logger.error("LLM gateway error: %s", exc)
            raise AppError(status_code=502, code="LLM_UNAVAILABLE",
                           message="LLM 게이트웨이 호출에 실패했습니다.") from exc
        self._count(response.usage)
        choice = response.choices[0]
        if not choice.message.content:
            raise AppError(status_code=502, code="LLM_BAD_RESPONSE",
                           message=f"LLM 응답이 비었거나 잘렸습니다 ({choice.finish_reason}).")
        if choice.finish_reason == "length":
            # Output limit hit; the JSON may still be complete, so let validation decide.
            logger.warning("LLM reply hit max_tokens (%s tokens)", response.usage and response.usage.completion_tokens)
        return choice.message.content

    def _count(self, usage) -> None:
        self.usage["calls"] += 1
        if usage is None:
            return
        self.usage["prompt"] += usage.prompt_tokens or 0
        self.usage["completion"] += usage.completion_tokens or 0
        details = getattr(usage, "completion_tokens_details", None)
        self.usage["reasoning"] += getattr(details, "reasoning_tokens", 0) or 0
        cached = getattr(usage, "prompt_tokens_details", None)
        self.usage["cached"] += getattr(cached, "cached_tokens", 0) or 0
