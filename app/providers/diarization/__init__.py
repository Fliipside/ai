from app.core.config import Settings
from app.providers.diarization.base import SpeakerDiarizer


def create_diarizer(settings: Settings) -> SpeakerDiarizer:
    # Import lazily: each engine is an optional dependency.
    if settings.diarization_engine == "pyannote":
        from app.providers.diarization.pyannote import PyannoteDiarizer

        return PyannoteDiarizer(
            hf_token=settings.hf_token.get_secret_value(),
            model=settings.pyannote_model,
        )

    from app.providers.diarization.sherpa_onnx import SherpaOnnxDiarizer

    return SherpaOnnxDiarizer(
        segmentation_model=settings.sherpa_segmentation_model,
        embedding_model=settings.sherpa_embedding_model,
    )
