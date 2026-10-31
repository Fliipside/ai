from app.core.config import Settings
from app.providers.speaker_embedding.base import SpeakerEmbedder


def create_embedder(settings: Settings) -> SpeakerEmbedder:
    # Follow the diarization engine so voiceprints stay comparable.
    if settings.diarization_engine == "pyannote":
        from app.providers.speaker_embedding.pyannote import PyannoteEmbedder

        return PyannoteEmbedder(hf_token=settings.hf_token.get_secret_value())

    from app.providers.speaker_embedding.sherpa_onnx import SherpaOnnxEmbedder

    return SherpaOnnxEmbedder(model=settings.sherpa_embedding_model)
