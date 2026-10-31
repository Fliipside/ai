from functools import lru_cache
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def load_environment() -> None:
    env_file = find_dotenv(usecwd=True)
    if env_file:
        load_dotenv(env_file, override=False)


load_environment()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    app_name: str = "Flipside AI"
    api_v1_prefix: str = "/api/v1"

    groq_api_key: SecretStr = SecretStr("")
    groq_stt_model: str = "whisper-large-v3"

    diarization_engine: Literal["pyannote", "sherpa_onnx"] = "pyannote"
    hf_token: SecretStr = SecretStr("")
    # Hugging Face repo id or a local directory with the same layout.
    pyannote_model: str = "pyannote/speaker-diarization-community-1"
    sherpa_segmentation_model: str = ""
    sherpa_embedding_model: str = ""

    # Cosine similarity needed to label a speaker as "me"; set after engine comparison.
    voice_match_threshold: float | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
