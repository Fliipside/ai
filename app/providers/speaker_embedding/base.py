from pathlib import Path
from typing import Protocol


class SpeakerEmbedder(Protocol):
    name: str

    # Embed the whole file, or only the given (start, end) spans in seconds.
    def embed(
        self,
        audio: Path,
        spans: list[tuple[float, float]] | None = None,
    ) -> list[float]: ...
