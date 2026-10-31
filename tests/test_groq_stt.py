import asyncio
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
