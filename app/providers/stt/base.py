from pathlib import Path
from typing import Protocol

from app.schemas.voice import TranscriptSegment


class SpeechToText(Protocol):

    async def transcribe(
        self,
        audio: Path,
        *,
        language: str = "ko",
        prompt: str | None = None,
    ) -> list[TranscriptSegment]: ...
