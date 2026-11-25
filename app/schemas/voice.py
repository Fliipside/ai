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
    speaker: str  # "me", or "A", "B", "C"... per person
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
    # Embeddings are only comparable when produced by the same engine and model, so this
    # also names the model when it is not the engine's default (e.g. "sherpa_onnx:<file>:<hash>").
    engine: str
    vector: list[float]
