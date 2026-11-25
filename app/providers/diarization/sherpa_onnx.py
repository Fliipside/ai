import threading
from functools import lru_cache
from pathlib import Path

from app.core.errors import AppError
from app.schemas.voice import SpeakerTurn
from app.utils.audio import SAMPLE_RATE, read_wav_samples

# Models run on CPU and are shared across requests; run one inference at a time.
inference_lock = threading.Lock()


class SherpaOnnxDiarizer:
    """sherpa-onnx offline diarization (requires `voice-sherpa` extra and ONNX model files)."""

    def __init__(
        self,
        segmentation_model: str,
        embedding_model: str,
        *,
        cluster_threshold: float = 0.5,
        num_threads: int = 4,
    ) -> None:
        self.segmentation_model = segmentation_model
        self.embedding_model = embedding_model
        self.cluster_threshold = cluster_threshold
        self.num_threads = num_threads

    def diarize(self, audio: Path, num_speakers: int | None = None) -> list[SpeakerTurn]:
        diarizer = load_diarizer(
            self.segmentation_model,
            self.embedding_model,
            num_speakers or -1,
            self.cluster_threshold,
            self.num_threads,
        )
        samples = read_wav_samples(audio)
        with inference_lock:
            result = diarizer.process(samples).sort_by_start_time()
        return [
            SpeakerTurn(start=r.start, end=r.end, speaker=f"SPEAKER_{r.speaker:02d}")
            for r in result
        ]


@lru_cache
def load_diarizer(
    segmentation_model: str,
    embedding_model: str,
    num_clusters: int,
    threshold: float,
    num_threads: int,
):
    sherpa_onnx = import_sherpa_onnx()
    require_model_files(SHERPA_SEGMENTATION_MODEL=segmentation_model, SHERPA_EMBEDDING_MODEL=embedding_model)
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation_model
            ),
            num_threads=num_threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=embedding_model,
            num_threads=num_threads,
        ),
        # num_clusters=-1 lets the threshold decide the speaker count.
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_clusters,
            threshold=threshold,
        ),
    )
    if not config.validate():
        raise not_configured("sherpa-onnx 화자 분리 설정이 올바르지 않습니다.")
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    if diarizer.sample_rate != SAMPLE_RATE:
        raise not_configured(f"sherpa-onnx 모델의 샘플레이트({diarizer.sample_rate})가 16kHz가 아닙니다.")
    return diarizer


def import_sherpa_onnx():
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise not_configured("sherpa-onnx가 설치되지 않았습니다.") from exc
    return sherpa_onnx


def require_model_files(**paths: str) -> None:
    for setting, path in paths.items():
        if not path or not Path(path).is_file():
            raise not_configured(f"{setting} 모델 파일을 찾을 수 없습니다.")


def not_configured(message: str) -> AppError:
    return AppError(status_code=503, code="DIARIZATION_NOT_CONFIGURED", message=message)
