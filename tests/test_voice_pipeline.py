import asyncio
from pathlib import Path

import pytest

from app.core.errors import AppError
from app.features.voice import pipeline as pipeline_module
from app.features.voice.pipeline import VoicePipeline
from app.schemas.voice import SpeakerTurn, TranscriptSegment, Voiceprint
from app.utils.audio import split_wav
from tests.helpers import requires_ffmpeg, write_tone


class FakeStt:
    def __init__(self, calls=None) -> None:
        self.calls = calls if calls is not None else []

    async def transcribe(self, audio, *, language="ko", prompt=None):
        self.calls.append("stt")
        return [TranscriptSegment(start=0.5, end=1.5, text=audio.stem)]


class FakeDiarizer:
    def __init__(self, calls=None) -> None:
        self.calls = calls if calls is not None else []

    def diarize(self, audio, num_speakers=None):
        self.calls.append("diarize")
        return [
            SpeakerTurn(start=0.0, end=2.0, speaker="S0"),
            SpeakerTurn(start=2.0, end=4.0, speaker="S1"),
        ]


class FakeEmbedder:
    name = "fake"

    def __init__(self) -> None:
        self.calls = []

    def embed(self, audio, spans=None):
        self.calls.append(spans)
        if spans is None:
            return [1.0, 0.0]
        return [1.0, 0.0] if spans[0][0] == 0.0 else [0.0, 1.0]


def make_pipeline(threshold: float | None = 0.5, **kwargs) -> VoicePipeline:
    return VoicePipeline(
        FakeStt(), FakeDiarizer(), FakeEmbedder(), match_threshold=threshold, **kwargs
    )


@requires_ffmpeg
def test_analyze_shifts_chunks_and_labels_roles(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline_module, "split_wav", lambda wav, out: split_wav(wav, out, max_seconds=2)
    )
    audio = write_tone(tmp_path / "call.wav", seconds=4)
    voiceprint = Voiceprint(engine="fake", vector=[1.0, 0.0])

    result = asyncio.run(make_pipeline().analyze(audio, voiceprint))

    assert [(u.start, u.end, u.speaker, u.role, u.text) for u in result.utterances] == [
        (0.5, 1.5, "me", "me", "audio_000"),
        (2.5, 3.5, "A", "customer", "audio_001"),
    ]
    assert [(s.speaker, s.role, s.speech_seconds) for s in result.speakers] == [
        ("me", "me", 2.0),
        ("A", "customer", 2.0),
    ]


@requires_ffmpeg
def test_analyze_without_voiceprint(tmp_path: Path) -> None:
    audio = write_tone(tmp_path / "call.wav", seconds=1)

    result = asyncio.run(make_pipeline().analyze(audio))

    assert [(u.speaker, u.role) for u in result.utterances] == [("A", "unknown")]
    assert [s.speaker for s in result.speakers] == ["A", "B"]


@requires_ffmpeg
def test_enroll(tmp_path: Path) -> None:
    audio = write_tone(tmp_path / "me.wav", seconds=1)

    voiceprint = asyncio.run(make_pipeline().enroll(audio))

    assert voiceprint == Voiceprint(engine="fake", vector=[1.0, 0.0])


@pytest.mark.parametrize(
    ("threshold", "engine", "code"),
    [(None, "fake", "VOICE_NOT_CONFIGURED"), (0.5, "pyannote", "VOICEPRINT_ENGINE_MISMATCH")],
)
def test_analyze_rejects_unusable_voiceprint(
    tmp_path: Path, threshold: float | None, engine: str, code: str
) -> None:
    voiceprint = Voiceprint(engine=engine, vector=[1.0, 0.0])

    with pytest.raises(AppError) as exc:
        asyncio.run(make_pipeline(threshold).analyze(tmp_path / "x.wav", voiceprint))

    assert exc.value.code == code


@requires_ffmpeg
def test_analyze_runs_stt_before_diarization(tmp_path: Path) -> None:
    calls = []
    pipeline = VoicePipeline(FakeStt(calls), FakeDiarizer(calls), FakeEmbedder())

    asyncio.run(pipeline.analyze(write_tone(tmp_path / "call.wav", seconds=1)))

    assert calls == ["stt", "diarize"]


@requires_ffmpeg
def test_analyze_rejects_too_long_audio(tmp_path: Path) -> None:
    calls = []
    pipeline = VoicePipeline(
        FakeStt(calls), FakeDiarizer(calls), FakeEmbedder(), max_audio_seconds=1
    )

    with pytest.raises(AppError) as exc:
        asyncio.run(pipeline.analyze(write_tone(tmp_path / "call.wav", seconds=2)))

    assert (exc.value.status_code, exc.value.code) == (400, "AUDIO_TOO_LONG")
    assert calls == []


def test_analyze_rejects_while_another_runs(tmp_path: Path) -> None:
    pipeline = make_pipeline(busy_retry_seconds=30)

    async def run() -> None:
        async with pipeline._analyze_lock:
            await pipeline.analyze(tmp_path / "x.wav")

    with pytest.raises(AppError) as exc:
        asyncio.run(run())

    assert (exc.value.status_code, exc.value.code) == (503, "VOICE_BUSY")
    assert exc.value.headers == {"Retry-After": "30"}


@requires_ffmpeg
def test_analyze_rejects_voiceprint_of_other_size(tmp_path: Path) -> None:
    voiceprint = Voiceprint(engine="fake", vector=[1.0, 0.0, 0.0])

    with pytest.raises(AppError) as exc:
        asyncio.run(make_pipeline().analyze(write_tone(tmp_path / "call.wav", seconds=4), voiceprint))

    assert exc.value.code == "VOICEPRINT_ENGINE_MISMATCH"


@requires_ffmpeg
def test_enroll_rejects_short_audio(tmp_path: Path) -> None:
    with pytest.raises(AppError) as exc:
        asyncio.run(make_pipeline(enroll_min_seconds=5).enroll(write_tone(tmp_path / "me.wav", seconds=1)))

    assert exc.value.code == "INVALID_AUDIO"
