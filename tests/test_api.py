from fastapi.testclient import TestClient

from app.main import create_app


client = TestClient(create_app())


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["data"] == {"status": "ok"}


def test_voice_analyze_not_ready() -> None:
    response = client.post(
        "/api/v1/voice/analyze",
        files={"file": ("sample.wav", b"", "audio/wav")},
    )


def test_voice_analyze_returns_not_ready_without_parsing_large_upload() -> None:
    response = client.post(
        "/api/v1/voice/analyze",
        files={"file": ("large.wav", b"x" * (10 * 1024 * 1024), "audio/wav")},
    )

    assert response.status_code == 501
    assert response.json()["code"] == "FEATURE_NOT_READY"
