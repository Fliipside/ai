import logging
import threading
from functools import lru_cache

from app.core.errors import AppError

logger = logging.getLogger(__name__)

BGE_M3 = "BAAI/bge-m3"

# Some models expect a role prefix on each input (e.g. e5: "query: " / "passage: ").
PREFIXES = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}

# Others ship a named instruction prompt for queries instead (sentence-transformers prompt_name).
QUERY_PROMPTS = {
    "Qwen/Qwen3-Embedding-0.6B": "query",
}

encode_lock = threading.Lock()


class SentenceTransformerEmbedder:
    def __init__(self, model: str = BGE_M3, batch_size: int = 32) -> None:
        self.model = model
        self.batch_size = batch_size
        self.name = f"sentence_transformers:{model}"
        self.query_prefix, self.document_prefix = PREFIXES.get(model, ("", ""))
        self.query_prompt = QUERY_PROMPTS.get(model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text], prompt_name=self.query_prompt)[0]

    def _encode(self, texts: list[str], prompt_name: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        model = load_model(self.model)
        with encode_lock:
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                prompt_name=prompt_name,
            )
        return vectors.tolist()


@lru_cache
def load_model(model: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise AppError(
            status_code=503,
            code="EMBEDDING_NOT_CONFIGURED",
            message="sentence-transformers가 설치되지 않았습니다.",
        ) from exc
    try:
        return SentenceTransformer(model)
    except Exception as exc:
        logger.error("Failed to load embedding model %s: %s", model, exc)
        raise AppError(
            status_code=503,
            code="EMBEDDING_NOT_CONFIGURED",
            message="임베딩 모델을 불러올 수 없습니다.",
        ) from exc
