import json

from fastapi.testclient import TestClient

from app.core.errors import VoiceBusyError
from app.main import create_app
from app.schemas.voice import Speaker, Utterance, VoiceAnalysis, Voiceprint


class FakePipeline:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = []

    async def analyze(self, audio, voiceprint=None):
        self.calls.append((audio.read_bytes(), audio.suffix, voiceprint))
        if self.error:
            raise self.error
        return VoiceAnalysis(
            speakers=[Speaker(speaker="me", role="me", speech_seconds=1.0)],
            utterances=[Utterance(start=0.0, end=1.0, speaker="me", role="me", text="안녕")],
        )

    async def enroll(self, audio):
        self.calls.append((audio.read_bytes(), audio.suffix, None))
        return Voiceprint(engine="pyannote", vector=[0.1, 0.2])


def make_client(pipeline: FakePipeline) -> TestClient:
    app = create_app()
    app.state.voice_pipeline = pipeline
    return TestClient(app)


def test_health() -> None:
    response = make_client(FakePipeline()).get("/health")

    assert response.status_code == 200
    assert response.json()["data"] == {"status": "ok"}


def test_analyze_with_voiceprint() -> None:
    pipeline = FakePipeline()
    voiceprint = {"engine": "pyannote", "vector": [0.1, 0.2]}

    response = make_client(pipeline).post(
        "/api/v1/voice/analyze",
        files={"file": ("meeting.m4a", b"audio-bytes", "audio/mp4")},
        data={"voiceprint": json.dumps(voiceprint)},
    )

    assert response.status_code == 200
    assert response.json()["data"]["speakers"] == [
        {"speaker": "me", "role": "me", "speech_seconds": 1.0}
    ]
    assert pipeline.calls == [(b"audio-bytes", ".m4a", Voiceprint(**voiceprint))]


def test_analyze_without_voiceprint_or_content_type() -> None:
    pipeline = FakePipeline()

    response = make_client(pipeline).post(
        "/api/v1/voice/analyze",
        files={"file": ("recording", b"audio-bytes")},
    )

    assert response.status_code == 200
    assert pipeline.calls == [(b"audio-bytes", "", None)]


def test_analyze_rejects_malformed_voiceprint() -> None:
    response = make_client(FakePipeline()).post(
        "/api/v1/voice/analyze",
        files={"file": ("a.wav", b"x")},
        data={"voiceprint": "{not json"},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST"


def test_busy_response_has_retry_after() -> None:
    response = make_client(FakePipeline(VoiceBusyError(60))).post(
        "/api/v1/voice/analyze",
        files={"file": ("a.wav", b"x")},
    )

    assert response.status_code == 503
    assert response.json()["code"] == "VOICE_BUSY"
    assert response.headers["retry-after"] == "60"


def test_enroll() -> None:
    response = make_client(FakePipeline()).post(
        "/api/v1/voice/enroll",
        files={"file": ("me.wav", b"voice")},
    )

    assert response.status_code == 200
    assert response.json()["data"] == {"engine": "pyannote", "vector": [0.1, 0.2]}
