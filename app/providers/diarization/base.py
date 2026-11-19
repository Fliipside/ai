from pathlib import Path
from typing import Protocol

from app.schemas.voice import SpeakerTurn


class SpeakerDiarizer(Protocol):
    # Runs locally on CPU; call from a worker thread.
    # `num_speakers` fixes the speaker count; None lets the engine estimate it.
    def diarize(self, audio: Path, num_speakers: int | None = None) -> list[SpeakerTurn]: ...
