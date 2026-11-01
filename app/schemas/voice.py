from typing import Literal

from pydantic import BaseModel

SpeakerRole = Literal["me", "customer", "unknown"]


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class SpeakerTurn(BaseModel):
    start: float
    end: float
    speaker: str  # engine label, e.g. "SPEAKER_00"


class Utterance(BaseModel):
    start: float
    end: float
    speaker: str
    role: SpeakerRole
    text: str


class VoiceAnalysis(BaseModel):
    utterances: list[Utterance]


class Voiceprint(BaseModel):
    # Embeddings are only comparable when produced by the same engine.
    engine: str
    vector: list[float]
