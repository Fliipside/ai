"""Compare diarization engines on sample recordings.

Usage (planned):
    python scripts/voice_benchmark.py data/samples/*.wav --engines pyannote sherpa_onnx
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+")
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=["pyannote", "sherpa_onnx"],
        default=["pyannote", "sherpa_onnx"],
    )
    parser.parse_args()
    raise SystemExit("voice benchmark is not implemented yet")


if __name__ == "__main__":
    main()
