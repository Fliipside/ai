import wave
from pathlib import Path

import pytest

from app.core.errors import AppError, InvalidAudioError
from app.utils import audio as audio_module
from app.utils.audio import split_wav, to_wav_16k_mono, wav_duration
from tests.helpers import requires_ffmpeg, write_tone


@requires_ffmpeg
def test_to_wav_16k_mono(tmp_path: Path) -> None:
    source = write_tone(tmp_path / "source.wav", seconds=1.5)

    wav = to_wav_16k_mono(source, tmp_path / "out.wav")

    with wave.open(str(wav), "rb") as result:
        assert result.getframerate() == 16_000
        assert result.getnchannels() == 1
        assert result.getsampwidth() == 2
    assert wav_duration(wav) == pytest.approx(1.5, abs=0.01)


@requires_ffmpeg
def test_to_wav_rejects_invalid_audio(tmp_path: Path) -> None:
    source = tmp_path / "broken.m4a"
    source.write_bytes(b"not audio")

    with pytest.raises(InvalidAudioError):
        to_wav_16k_mono(source, tmp_path / "out.wav")


def test_split_wav_keeps_short_file(tmp_path: Path) -> None:
    wav = write_tone(tmp_path / "short.wav", seconds=1, rate=16_000, channels=1)

    chunks = split_wav(wav, tmp_path, max_seconds=2)

    assert [(c.path, c.offset) for c in chunks] == [(wav, 0.0)]


def test_split_wav_offsets(tmp_path: Path) -> None:
    wav = write_tone(tmp_path / "long.wav", seconds=5, rate=16_000, channels=1)

    chunks = split_wav(wav, tmp_path, max_seconds=2)

    assert [c.offset for c in chunks] == [0.0, 2.0, 4.0]
    assert [wav_duration(c.path) for c in chunks] == [2.0, 2.0, 1.0]


def test_missing_ffmpeg_is_not_configured(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(audio_module.shutil, "which", lambda name: None)

    with pytest.raises(AppError) as exc:
        audio_module.to_wav_16k_mono(tmp_path / "a.m4a", tmp_path / "a.wav")

    assert (exc.value.status_code, exc.value.code) == (503, "FFMPEG_NOT_CONFIGURED")


@requires_ffmpeg
def test_to_wav_16k_mono_stops_at_max_seconds(tmp_path: Path) -> None:
    source = write_tone(tmp_path / "source.wav", seconds=3)

    wav = to_wav_16k_mono(source, tmp_path / "out.wav", max_seconds=1)

    assert wav_duration(wav) == pytest.approx(1.0, abs=0.05)
