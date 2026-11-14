import threading
from functools import lru_cache
from pathlib import Path

from app.core.errors import InvalidAudioError
from app.providers.diarization.sherpa_onnx import (
    import_sherpa_onnx,
    not_configured,
    require_model_files,
)
from app.utils.audio import SAMPLE_RATE, crop_samples, read_wav_samples

# Separate from the diarization lock (its own extractor), so enrolling a voice
# does not wait for a long diarization to finish.
embedding_lock = threading.Lock()


class SherpaOnnxEmbedder:
    def __init__(self, model: str, *, num_threads: int = 4) -> None:
        self.model = model
        self.num_threads = num_threads
        # Different embedding models are not comparable, so the model is part of the name.
        self.name = f"sherpa_onnx:{Path(model).stem}"

    def embed(
        self,
        audio: Path,
        spans: list[tuple[float, float]] | None = None,
    ) -> list[float]:
        extractor = load_extractor(self.model, self.num_threads)
        samples = read_wav_samples(audio)
        if spans is not None:
            samples = crop_samples(samples, spans)

        with embedding_lock:
            stream = extractor.create_stream()
            stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=samples)
            stream.input_finished()
            if not extractor.is_ready(stream):
                raise InvalidAudioError("목소리 구간이 너무 짧습니다.")
            vector = extractor.compute(stream)
        return [float(x) for x in vector]


@lru_cache
def load_extractor(model: str, num_threads: int):
    sherpa_onnx = import_sherpa_onnx()
    require_model_files(SHERPA_EMBEDDING_MODEL=model)
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model, num_threads=num_threads)
    if not config.validate():
        raise not_configured("sherpa-onnx 임베딩 설정이 올바르지 않습니다.")
    return sherpa_onnx.SpeakerEmbeddingExtractor(config)
