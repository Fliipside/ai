import logging
import os
import threading
from functools import lru_cache
from pathlib import Path

from app.core.errors import AppError
from app.schemas.voice import SpeakerTurn
from app.utils.audio import SAMPLE_RATE, read_wav_samples

logger = logging.getLogger(__name__)

COMMUNITY_1 = "pyannote/speaker-diarization-community-1"

# Models run on CPU and are shared across requests; run one inference at a time.
inference_lock = threading.Lock()


class PyannoteDiarizer:
    """pyannote/speaker-diarization-community-1 (requires `voice-pyannote` extra and HF token)."""

    def __init__(
        self,
        hf_token: str,
        model: str = COMMUNITY_1,
        segmentation_step: float = 0.5,
    ) -> None:
        self.hf_token = hf_token
        self.model = model
        self.segmentation_step = segmentation_step

    def diarize(self, audio: Path, num_speakers: int | None = None) -> list[SpeakerTurn]:
        pipeline = load_pipeline(self.model, self.hf_token, self.segmentation_step)
        waveform = load_waveform(audio)
        with inference_lock:
            output = pipeline(
                {"waveform": waveform, "sample_rate": SAMPLE_RATE},
                num_speakers=num_speakers,
            )

        # Exclusive diarization has one speaker at a time, which suits matching STT segments.
        return [
            SpeakerTurn(start=segment.start, end=segment.end, speaker=label)
            for segment, _, label in output.exclusive_speaker_diarization.itertracks(
                yield_label=True
            )
        ]


@lru_cache
def load_pipeline(model: str, hf_token: str, segmentation_step: float):
    Pipeline = import_pyannote().Pipeline
    try:
        pipeline = Pipeline.from_pretrained(model, token=hf_token or None)
    except Exception as exc:
        raise model_load_error(exc) from exc
    # The sliding window is built at load time, so set its step on the inference directly.
    # Most of the time goes into one embedding per window: 0.5 (5s of a 10s window) is about
    # 5x faster than the default 0.1 and matched it on our recordings.
    pipeline._segmentation.step = segmentation_step * pipeline._segmentation.duration
    return pipeline


def import_pyannote():
    # pyannote sends usage metrics by default; keep them off unless explicitly enabled.
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
    try:
        from pyannote import audio
    except ImportError as exc:
        raise AppError(
            status_code=503,
            code="DIARIZATION_NOT_CONFIGURED",
            message="pyannote.audio가 설치되지 않았습니다.",
        ) from exc
    return audio


def model_load_error(exc: Exception) -> AppError:
    logger.error("Failed to load pyannote model: %s", exc)
    return AppError(
        status_code=503,
        code="DIARIZATION_NOT_CONFIGURED",
        message="pyannote 모델을 불러올 수 없습니다. HF_TOKEN과 모델 사용 동의를 확인해 주세요.",
    )


def load_waveform(audio: Path):
    """Read a 16kHz mono WAV as a (1, samples) tensor."""
    import torch

    return torch.from_numpy(read_wav_samples(audio)).unsqueeze(0)
