from app.core.config import Settings
from app.providers.diarization.base import SpeakerDiarizer


def create_diarizer(settings: Settings) -> SpeakerDiarizer:
    # Import lazily: each engine is an optional dependency.
    if settings.diarization_engine == "pyannote":
        from app.providers.diarization.pyannote import PyannoteDiarizer

        return PyannoteDiarizer(
            hf_token=settings.hf_token.get_secret_value(),
            model=settings.pyannote_model,
            segmentation_step=settings.pyannote_segmentation_step,
        )

    from app.providers.diarization.sherpa_onnx import SherpaOnnxDiarizer

    return SherpaOnnxDiarizer(
        segmentation_model=settings.sherpa_segmentation_model,
        embedding_model=settings.sherpa_embedding_model,
        cluster_threshold=settings.sherpa_cluster_threshold,
        num_threads=settings.sherpa_num_threads,
    )
