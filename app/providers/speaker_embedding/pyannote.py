from functools import lru_cache
from pathlib import Path

from app.core.errors import InvalidAudioError
from app.providers.diarization.pyannote import (
    COMMUNITY_1,
    import_pyannote,
    inference_lock,
    load_waveform,
    model_load_error,
)
from app.utils.audio import SAMPLE_RATE


class PyannoteEmbedder:
    """Embedding model bundled with community-1 (WeSpeaker ResNet34, 256 dims)."""

    name = "pyannote"

    def __init__(self, hf_token: str, model: str = COMMUNITY_1) -> None:
        self.hf_token = hf_token
        self.model = model

    def embed(
        self,
        audio: Path,
        spans: list[tuple[float, float]] | None = None,
    ) -> list[float]:
        import torch

        inference = load_inference(self.model, self.hf_token)
        waveform = load_waveform(audio)
        if spans is not None:
            waveform = torch.cat(
                [
                    waveform[:, int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
                    for start, end in spans
                ],
                dim=1,
            )
        if waveform.shape[1] == 0:
            raise InvalidAudioError("목소리 구간이 없습니다.")

        with inference_lock:
            vector = inference({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        return [float(x) for x in vector.reshape(-1)]


@lru_cache
def load_inference(model: str, hf_token: str):
    audio = import_pyannote()
    try:
        embedding = audio.Model.from_pretrained(model, subfolder="embedding", token=hf_token or None)
    except Exception as exc:
        raise model_load_error(exc) from exc
    return audio.Inference(embedding, window="whole")
