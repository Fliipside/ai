from pathlib import Path


class PyannoteEmbedder:
    name = "pyannote"

    def __init__(self, hf_token: str) -> None:
        self.hf_token = hf_token

    def embed(
        self,
        audio: Path,
        spans: list[tuple[float, float]] | None = None,
    ) -> list[float]:
        raise NotImplementedError
