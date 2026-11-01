from pathlib import Path
from typing import Protocol

from app.schemas.voice import SpeakerTurn


class SpeakerDiarizer(Protocol):
    # Runs locally on CPU; call from a worker thread.
    def diarize(self, audio: Path) -> list[SpeakerTurn]: ...
