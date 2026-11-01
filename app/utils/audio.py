import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import InvalidAudioError

SAMPLE_RATE = 16_000
# 16kHz mono 16-bit WAV is 32KB/s, so 10 minutes (19.2MB) stays under Groq's 25MB limit.
STT_CHUNK_SECONDS = 600


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    offset: float  # seconds from the start of the original audio


def to_wav_16k_mono(source: Path, dest: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed")

    result = subprocess.run(
        [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16",
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

