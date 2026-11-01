from pathlib import Path

from app.schemas.voice import SpeakerTurn


class SherpaOnnxDiarizer:
    """sherpa-onnx offline diarization (requires `voice-sherpa` extra and ONNX model files)."""

    def __init__(self, segmentation_model: str, embedding_model: str) -> None:
        self.segmentation_model = segmentation_model
        self.embedding_model = embedding_model

    def diarize(self, audio: Path) -> list[SpeakerTurn]:
        raise NotImplementedError
