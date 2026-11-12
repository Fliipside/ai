from app.core.config import Settings
from app.providers.text_embedding.base import TextEmbedder


def create_text_embedder(settings: Settings) -> TextEmbedder:
    from app.providers.text_embedding.sentence_transformers import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(settings.text_embedding_model)
