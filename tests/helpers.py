import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)


def write_tone(path: Path, seconds: float, rate: int = 44_100, channels: int = 2) -> Path:
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        samples = (int(8000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(frames))
        wav.writeframes(b"".join(struct.pack("<h", s) * channels for s in samples))
    return path
