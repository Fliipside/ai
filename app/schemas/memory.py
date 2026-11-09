from pydantic import BaseModel

from app.schemas.voice import Utterance


class StoredUtterance(Utterance):
    # Stable id so chunks can point back to the original speech and time.
    id: str


class MemoryChunk(BaseModel):
    text: str
    start: float
    end: float
    utterance_ids: list[str]
