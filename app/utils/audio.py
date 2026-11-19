import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import AppError, InvalidAudioError

SAMPLE_RATE = 16_000
# 16kHz mono 16-bit WAV is 32KB/s, so 10 minutes (19.2MB) stays under Groq's 25MB limit.
STT_CHUNK_SECONDS = 600


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    offset: float  # seconds from the start of the original audio


def to_wav_16k_mono(source: Path, dest: Path, max_seconds: float | None = None) -> Path:
    """Convert to 16kHz mono 16-bit WAV, stopping after max_seconds if given."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AppError(
            status_code=503,
            code="FFMPEG_NOT_CONFIGURED",
            message="ffmpeg가 설치되지 않았습니다.",
        )

    limit = ["-t", str(max_seconds)] if max_seconds is not None else []
    result = subprocess.run(
        [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16",
            *limit,
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise InvalidAudioError("오디오 파일을 읽을 수 없습니다.")
    return dest


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def split_wav(
    path: Path,
    output_dir: Path,
    max_seconds: float = STT_CHUNK_SECONDS,
) -> list[AudioChunk]:
    with wave.open(str(path), "rb") as src:
        params = src.getparams()
        frames_per_chunk = int(max_seconds * params.framerate)
        if params.nframes <= frames_per_chunk:
            return [AudioChunk(path=path, offset=0.0)]

        chunks = []
        start = 0
        while start < params.nframes:
            chunk_path = output_dir / f"{path.stem}_{len(chunks):03d}.wav"
            with wave.open(str(chunk_path), "wb") as dst:
                dst.setparams(params)
                dst.writeframes(src.readframes(frames_per_chunk))
            chunks.append(AudioChunk(path=chunk_path, offset=start / params.framerate))
            start += frames_per_chunk
        return chunks



def read_wav_samples(path: Path):
    """Read a 16kHz mono 16-bit WAV (see `to_wav_16k_mono`) as float32 samples in [-1, 1].

    Needs numpy, which comes with the diarization extras.
    """
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"expected 16kHz mono 16-bit WAV: {path}")
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def crop_samples(samples, spans: list[tuple[float, float]]):
    """Concatenate the given (start, end) spans in seconds."""
    import numpy as np

    return np.concatenate(
        [samples[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)] for start, end in spans]
        or [samples[:0]]
    )
