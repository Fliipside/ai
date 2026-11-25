# [테스트] LLM 게이트웨이 테스트
# 실제 호출 없이 정해진 답을 돌려줘서 빈 응답·형식 오류·일시 오류 재요청, 전체 시간 제한, 코드 블록 제거, 최종 실패 코드를 확인합니다.

import asyncio
import time

import pytest
from pydantic import BaseModel

from app.core.errors import AppError
from app.providers.llm import gateway
from app.providers.llm.gateway import GatewayLLM, TransientLLMError, compact_schema


class Reply(BaseModel):
    answer: str
    score: int | None


TRANSIENT = TransientLLMError("APITimeoutError")


class ScriptedGateway(GatewayLLM):
    """Returns the given replies in order instead of calling the gateway.

    None = empty reply, TRANSIENT = timeout or other error worth retrying.
    """

    def __init__(self, replies: list, **options) -> None:
        super().__init__(api_key="test", **options)
        self.replies = list(replies)
        self.sent: list[list[dict]] = []
        self.timeouts: list[float] = []

    def _get_client(self):
        return object()

    def _complete(self, messages, timeout):
        self.sent.append(list(messages))
        self.timeouts.append(timeout)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_valid_reply_needs_one_call() -> None:
    llm = ScriptedGateway(['{"answer": "네", "score": 3}'])

    assert llm.generate("지시", "입력", Reply) == Reply(answer="네", score=3)
    assert (len(llm.sent), llm.usage["retries"]) == (1, 0)


def test_code_fence_around_json_is_accepted() -> None:
    llm = ScriptedGateway(['```json\n{"answer": "네", "score": null}\n```'])

    assert llm.generate("지시", "입력", Reply).answer == "네"


def test_empty_reply_is_asked_again_with_the_same_messages() -> None:
    llm = ScriptedGateway([None, '{"answer": "네", "score": 1}'])

    assert llm.generate("지시", "입력", Reply).score == 1
    assert llm.sent[0] == llm.sent[1]
    assert llm.usage["retries"] == 1


def test_wrong_shape_is_asked_again_with_the_error() -> None:
    llm = ScriptedGateway(['{"answer": "네"}', '{"answer": "네", "score": 2}'])

    assert llm.generate("지시", "입력", Reply).score == 2
    assert llm.sent[1][-2] == {"role": "assistant", "content": '{"answer": "네"}'}
    assert "score" in llm.sent[1][-1]["content"]


def test_transient_error_is_tried_again() -> None:
    llm = ScriptedGateway([TRANSIENT, '{"answer": "네", "score": 1}'])

    assert llm.generate("지시", "입력", Reply).score == 1
    assert llm.usage["retries"] == 1


@pytest.mark.parametrize(("replies", "code"), [
    ([None, None], "LLM_BAD_RESPONSE"),
    (["아님", "아님"], "LLM_BAD_RESPONSE"),
    ([None, "아님"], "LLM_BAD_RESPONSE"),
    ([TRANSIENT, TRANSIENT], "LLM_UNAVAILABLE"),
    (["아님", TRANSIENT], "LLM_UNAVAILABLE"),
    ([TRANSIENT, "아님"], "LLM_BAD_RESPONSE"),
])
def test_gives_up_after_two_attempts(replies, code) -> None:
    llm = ScriptedGateway(replies)

    with pytest.raises(AppError) as error:
        llm.generate("지시", "입력", Reply)

    assert (error.value.status_code, error.value.code) == (502, code)
    assert len(llm.sent) == 2


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_each_call_waits_at_most_what_is_left_of_the_total(monkeypatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(gateway.time, "monotonic", clock)

    class Slow(ScriptedGateway):
        def _complete(self, messages, timeout):
            clock.now += 100  # the first call used 100 of the 150 seconds
            return super()._complete(messages, timeout)

    llm = Slow([None, '{"answer": "네", "score": 1}'], call_timeout=120, total_timeout=150)

    assert llm.generate("지시", "입력", Reply).score == 1
    assert llm.timeouts == [120, 50]


def test_no_new_call_when_too_little_time_is_left(monkeypatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(gateway.time, "monotonic", clock)

    class Slow(ScriptedGateway):
        def _complete(self, messages, timeout):
            clock.now += 115
            return super()._complete(messages, timeout)

    llm = Slow([TRANSIENT, '{"answer": "네", "score": 1}'], call_timeout=120, total_timeout=120)

    with pytest.raises(AppError) as error:
        llm.generate("지시", "입력", Reply)

    assert error.value.code == "LLM_UNAVAILABLE"
    assert len(llm.sent) == 1


def test_schema_outline_is_short_and_lists_every_key() -> None:
    assert compact_schema(Reply) == '{"answer": 문자열, "score": 정수|null}'


def openai_error(name: str):
    openai = pytest.importorskip("openai")
    import httpx

    request = httpx.Request("POST", "https://gateway.test/v1/chat/completions")
    if name == "timeout":
        return openai.APITimeoutError(request=request)
    status = {"rate_limit": 429, "server": 500, "bad_request": 400}[name]
    cls = {429: openai.RateLimitError, 500: openai.InternalServerError, 400: openai.BadRequestError}[status]
    return cls("error", response=httpx.Response(status, request=request), body=None)


class RaisingClient:
    """Stands in for the async OpenAI client and raises the given error from chat.completions.create."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        pass

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        raise self.error


class HangingClient(RaisingClient):
    """A gateway that never answers; records whether the request was cancelled."""

    def __init__(self) -> None:
        super().__init__(error=None)
        self.cancelled = 0

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


@pytest.mark.parametrize("name", ["timeout", "rate_limit", "server"])
def test_openai_transient_errors_are_retried(name) -> None:
    client = RaisingClient(openai_error(name))
    llm = GatewayLLM(api_key="test")
    llm._get_client = lambda: client

    with pytest.raises(AppError) as error:
        llm.generate("지시", "입력", Reply)

    assert error.value.code == "LLM_UNAVAILABLE"
    assert len(client.calls) == 2
    assert client.calls[0]["timeout"] == 120


def test_openai_request_error_is_not_retried() -> None:
    client = RaisingClient(openai_error("bad_request"))
    llm = GatewayLLM(api_key="test")
    llm._get_client = lambda: client

    with pytest.raises(AppError) as error:
        llm.generate("지시", "입력", Reply)

    assert error.value.code == "LLM_UNAVAILABLE"
    assert len(client.calls) == 1


def test_client_leaves_retries_to_generate() -> None:
    pytest.importorskip("openai")

    assert GatewayLLM(api_key="test")._get_client().max_retries == 0


def test_request_still_running_at_the_limit_is_cancelled() -> None:
    # httpx limits each read, not the whole request, so each call's limit (at most what is
    # left of the total) is enforced around it.
    client = HangingClient()
    llm = GatewayLLM(api_key="test", call_timeout=0.1, total_timeout=60)
    llm._get_client = lambda: client

    started = time.monotonic()
    with pytest.raises(AppError) as error:
        llm.generate("지시", "입력", Reply)

    assert error.value.code == "LLM_UNAVAILABLE"
    assert time.monotonic() - started < 1
    assert client.cancelled == len(client.calls) == 2
