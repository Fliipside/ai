from pathlib import Path


class SherpaOnnxEmbedder:
    name = "sherpa_onnx"

    def __init__(self, model: str) -> None:
        self.model = model

    def embed(
        self,
        audio: Path,
        spans: list[tuple[float, float]] | None = None,
    ) -> list[float]:
        raise NotImplementedError
