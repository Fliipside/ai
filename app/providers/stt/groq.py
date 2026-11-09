import math
from pathlib import Path

import httpx

from app.core.errors import AppError
from app.schemas.voice import TranscriptSegment

GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


class GroqSpeechToText:
    """Groq Whisper API (OpenAI-compatible /audio/transcriptions).

    Free plan: 25MB per file, 7,200 audio seconds/hour, 28,800/day.
    `prompt` is limited to 224 tokens.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.client = client
        self.timeout = timeout

    async def transcribe(
        self,
        audio: Path,
        *,
        language: str = "ko",
        prompt: str | None = None,
    ) -> list[TranscriptSegment]:
        if not self.api_key:
            raise AppError(
                status_code=503,
                code="STT_NOT_CONFIGURED",
                message="GROQ_API_KEY가 설정되지 않았습니다.",
            )

        data = {
            "model": self.model,
            "language": language,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
            "temperature": "0",
        }
        if prompt:
            data["prompt"] = prompt

        with audio.open("rb") as file:
            response = await self._post(
                data=data,
                files={"file": (audio.name, file)},
            )

        if response.status_code == 429:
            raise AppError(
                status_code=429,
                code="STT_RATE_LIMITED",
                message="음성 인식 사용량 한도를 초과했습니다. 잠시 후 다시 시도해 주세요.",
                headers=_retry_after(response),
            )
        if response.status_code != 200:
            raise AppError(
                status_code=502,
                code="STT_FAILED",
                message=f"음성 인식 요청이 실패했습니다. (status {response.status_code})",
            )

        return [
            TranscriptSegment(
                start=segment["start"],
                end=segment["end"],
                text=segment["text"].strip(),
            )
            for segment in response.json().get("segments", [])
            if segment["text"].strip()
        ]

    async def _post(self, **kwargs) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            if self.client is not None:
                return await self.client.post(
                    GROQ_TRANSCRIPTIONS_URL, headers=headers, timeout=self.timeout, **kwargs
                )
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                return await client.post(GROQ_TRANSCRIPTIONS_URL, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise AppError(
                status_code=502,
                code="STT_FAILED",
                message="음성 인식 서버에 연결할 수 없습니다.",
            ) from exc


def _retry_after(response: httpx.Response) -> dict[str, str] | None:
    # Pass Groq's wait time on so backend retries when the quota frees up.
    try:
        seconds = math.ceil(float(response.headers["retry-after"]))
    except (KeyError, ValueError):
        return None
    return {"Retry-After": str(max(seconds, 0))}
