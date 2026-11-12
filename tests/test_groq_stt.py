import asyncio
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest

from app.core.errors import AppError
from app.providers.stt.groq import GroqSpeechToText


def make_stt(handler) -> GroqSpeechToText:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GroqSpeechToText(api_key="test-key", model="whisper-large-v3", client=client)


def test_transcribe_parses_segments(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={
            "text": "안녕하세요 네",
            "segments": [
                {"start": 0.0, "end": 1.2, "text": " 안녕하세요"},
                {"start": 1.2, "end": 1.5, "text": "  "},
                {"start": 1.5, "end": 2.0, "text": " 네"},
            ],
        })

    segments = asyncio.run(make_stt(handler).transcribe(audio, prompt="플립사이드"))

    assert [(s.start, s.end, s.text) for s in segments] == [
        (0.0, 1.2, "안녕하세요"),
        (1.5, 2.0, "네"),
    ]
    request = requests[0]
    body = request.read().decode(errors="ignore")
    assert request.headers["Authorization"] == "Bearer test-key"
    for field in ("whisper-large-v3", "verbose_json", "ko", "플립사이드"):
        assert field in body


@pytest.mark.parametrize(("status", "code"), [(429, "STT_RATE_LIMITED"), (500, "STT_FAILED")])
def test_transcribe_errors(tmp_path: Path, status: int, code: str) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    stt = make_stt(lambda _request: httpx.Response(status, json={}))

    with pytest.raises(AppError) as exc:
        asyncio.run(stt.transcribe(audio))

    assert exc.value.code == code


def test_transcribe_requires_api_key(tmp_path: Path) -> None:
    stt = GroqSpeechToText(api_key="", model="whisper-large-v3")

    with pytest.raises(AppError) as exc:
        asyncio.run(stt.transcribe(tmp_path / "a.wav"))

    assert exc.value.code == "STT_NOT_CONFIGURED"


def test_rate_limit_passes_retry_after(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "7.2"}, json={})

    with pytest.raises(AppError) as exc:
        asyncio.run(make_stt(handler).transcribe(audio))

    assert exc.value.headers == {"Retry-After": "8"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("7.2", "8"),
        ("-3", "0"),
        ("1e400", None),
        ("nan", None),
        ("not-a-date", None),
    ],
)
def test_rate_limit_retry_after_values(tmp_path: Path, value: str, expected: str | None) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": value}, json={})

    with pytest.raises(AppError) as exc:
        asyncio.run(make_stt(handler).transcribe(audio))

    assert exc.value.code == "STT_RATE_LIMITED"
    assert exc.value.headers == (None if expected is None else {"Retry-After": expected})


def test_rate_limit_retry_after_http_date(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=90), usegmt=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": when}, json={})

    with pytest.raises(AppError) as exc:
        asyncio.run(make_stt(handler).transcribe(audio))

    assert 85 <= int(exc.value.headers["Retry-After"]) <= 90
