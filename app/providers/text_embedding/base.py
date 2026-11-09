from typing import Protocol


class TextEmbedder(Protocol):
    # Vectors are only comparable when produced by the same name.
    name: str

    # Unit-length vectors, so cosine similarity is a dot product.
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
