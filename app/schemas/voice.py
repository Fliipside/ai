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
    speaker: str  # "me", or "speaker_1", "speaker_2"... per person
    role: SpeakerRole
    text: str


class Speaker(BaseModel):
    speaker: str
    role: SpeakerRole
    speech_seconds: float


class VoiceAnalysis(BaseModel):
    speakers: list[Speaker]
    utterances: list[Utterance]


class Voiceprint(BaseModel):
    # Embeddings are only comparable when produced by the same engine.
    engine: str
    vector: list[float]
