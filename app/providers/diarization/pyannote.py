from pathlib import Path

from app.schemas.voice import SpeakerTurn


class PyannoteDiarizer:
    """pyannote/speaker-diarization-community-1 (requires `voice-pyannote` extra and HF token)."""

    def __init__(self, hf_token: str) -> None:
        self.hf_token = hf_token

    def diarize(self, audio: Path) -> list[SpeakerTurn]:
        raise NotImplementedError
