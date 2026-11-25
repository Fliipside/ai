from app.core.config import Settings
from app.providers.stt.base import SpeechToText


def create_stt(settings: Settings) -> SpeechToText:
    from app.providers.stt.groq import GroqSpeechToText

    return GroqSpeechToText(
        api_key=settings.groq_api_key.get_secret_value(),
        model=settings.groq_stt_model,
    )
