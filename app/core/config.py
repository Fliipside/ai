# [공통 설정] 환경 변수(.env)로 바꿀 수 있는 설정값
# API 키, 화자 분리 엔진, 임베딩·재정렬 모델 이름 등을 읽어 옵니다.

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
    # Window step as a ratio of the 10s window (pyannote default 0.1); larger is faster.
    pyannote_segmentation_step: float = 0.5
    sherpa_segmentation_model: str = ""
    sherpa_embedding_model: str = ""
    # Cosine distance for merging speakers when the speaker count is not given.
    sherpa_cluster_threshold: float = 0.5
    sherpa_num_threads: int = 4

    # Cosine similarity needed to label a speaker as "me". On our recordings (pyannote)
    # the user scored about 0.84 and other people about 0.2.
    voice_match_threshold: float | None = 0.5
    # Longer recordings are rejected before any model runs (Groq allows 7,200 audio s/hour).
    voice_max_audio_seconds: float = 3600
    voice_enroll_min_seconds: float = 5
    # Retry-After sent while another analysis is running.
    voice_busy_retry_seconds: int = 60
    # Load models at startup so the first request does not wait for them.
    voice_preload_models: bool = False

    # Hugging Face sentence-transformers model for conversation memory search.
    text_embedding_model: str = "BAAI/bge-m3"
    # Cross-encoder that reorders the top pgvector matches.
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # LLM that extracts key information from conversations (OpenAI-compatible gateway).
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://gateway.aitestbed.kr/v1"
    llm_model: str = "claude-sonnet-5-5"
    # Passed to the gateway as reasoning_effort (low | medium | high); empty = model default.
    # low keeps hidden reasoning tokens, which were ~90% of the bill on Opus, near zero.
    llm_reasoning_effort: str | None = "low"


@lru_cache
def get_settings() -> Settings:
    return Settings()
