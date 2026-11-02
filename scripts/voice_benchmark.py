"""Compare diarization engines on sample recordings.

Without reference labels, engines are compared by speed, number of speakers,
agreement with each other and how clearly the enrolled voice stands out.

Usage:
    python scripts/voice_benchmark.py data/voice/*.m4a \\
        --engines pyannote sherpa_onnx \\
        --sherpa-embedding models/sherpa/wespeaker_en_voxceleb_resnet34_LM.onnx \\
        --num-speakers auto 2 --enroll data/voice/enroll_me.wav --out result.json
"""

import argparse
import itertools
import json
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.features.voice.pipeline import SPEAKER_EMBED_SECONDS  # noqa: E402
from app.features.voice.speaker_id import cosine_similarity, longest_spans  # noqa: E402
from app.schemas.voice import SpeakerTurn  # noqa: E402
from app.utils.audio import to_wav_16k_mono  # noqa: E402

FRAME = 0.01  # seconds per frame when comparing engines


def build_engines(names: list[str], sherpa_embeddings: list[str]) -> dict:
    settings = get_settings()
    engines = {}
    if "pyannote" in names:
        from app.providers.diarization.pyannote import PyannoteDiarizer
        from app.providers.speaker_embedding.pyannote import PyannoteEmbedder

        token = settings.hf_token.get_secret_value()
        engines["pyannote"] = (
            PyannoteDiarizer(token, settings.pyannote_model, settings.pyannote_segmentation_step),
            PyannoteEmbedder(token, settings.pyannote_model),
        )
    if "sherpa_onnx" in names:
        from app.providers.diarization.sherpa_onnx import SherpaOnnxDiarizer
        from app.providers.speaker_embedding.sherpa_onnx import SherpaOnnxEmbedder

        for model in sherpa_embeddings or [settings.sherpa_embedding_model]:
            embedder = SherpaOnnxEmbedder(model, num_threads=settings.sherpa_num_threads)
            engines[embedder.name] = (
                SherpaOnnxDiarizer(
                    settings.sherpa_segmentation_model,
                    model,
                    cluster_threshold=settings.sherpa_cluster_threshold,
                    num_threads=settings.sherpa_num_threads,
                ),
                embedder,
            )
    return engines


def trim_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "rb") as src:
        params = src.getparams()
        frames = src.readframes(int(seconds * params.framerate))
    with wave.open(str(path), "wb") as dst:
        dst.setparams(params)
        dst.writeframes(frames)


def speaker_spans(turns: list[SpeakerTurn]) -> dict[str, list[tuple[float, float]]]:
    spans: dict[str, list[tuple[float, float]]] = {}
    for turn in turns:
        spans.setdefault(turn.speaker, []).append((turn.start, turn.end))
    return spans


def to_frames(turns: list[SpeakerTurn], length: int) -> list[str | None]:
    frames: list[str | None] = [None] * length
    for turn in turns:
        for i in range(int(turn.start / FRAME), min(int(turn.end / FRAME), length)):
            frames[i] = turn.speaker
    return frames


def agreement(a: list[SpeakerTurn], b: list[SpeakerTurn], duration: float) -> float:
    """Share of frames where either engine hears speech and both agree, after the best label mapping."""
    length = int(duration / FRAME)
    fa, fb = to_frames(a, length), to_frames(b, length)
    counts: dict[tuple[str | None, str | None], int] = {}
    for x, y in zip(fa, fb):
        if x is not None or y is not None:
            counts[(x, y)] = counts.get((x, y), 0) + 1
    total = sum(counts.values())
    if total == 0:
        return 1.0

    la = sorted({x for x, _ in counts if x is not None})
    lb = sorted({y for _, y in counts if y is not None})
    return best_mapping_frames(counts, la, lb) / total


def best_mapping_frames(counts: dict, la: list[str], lb: list[str]) -> int:
    """Frames that agree under the one-to-one label mapping that maximises agreement."""
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        # Greedy fallback: take the largest overlaps first.
        used_a, used_b, matched = set(), set(), 0
        for (x, y), n in sorted(counts.items(), key=lambda item: -item[1]):
            if x in la and y in lb and x not in used_a and y not in used_b:
                used_a.add(x)
                used_b.add(y)
                matched += n
        return matched

    matrix = np.array([[counts.get((x, y), 0) for y in lb] for x in la])
    if matrix.size == 0:
        return 0
    rows, cols = linear_sum_assignment(matrix, maximize=True)
    return int(matrix[rows, cols].sum())


def run(args: argparse.Namespace) -> dict:
    """Diarize every file with every engine; the report is saved after each file."""
    engines = build_engines(args.engines, args.sherpa_embedding)
    modes = [None if n == "auto" else int(n) for n in args.num_speakers]
    report: dict = {"files": {}}

    with tempfile.TemporaryDirectory() as tmp:
        enroll = None
        if args.enroll:
            enroll = to_wav_16k_mono(Path(args.enroll), Path(tmp) / "enroll.wav")

        for index, audio in enumerate(args.audio):
            wav = to_wav_16k_mono(Path(audio), Path(tmp) / f"audio{index}.wav")
            if args.seconds:
                trim_wav(wav, args.seconds)
            with wave.open(str(wav), "rb") as w:
                duration = w.getnframes() / w.getframerate()

            runs = {}
            for name, (diarizer, embedder) in engines.items():
                me = embedder.embed(enroll) if enroll else None
                for mode in modes:
                    key = f"{name} n={mode or 'auto'}"
                    started = time.time()
                    turns = diarizer.diarize(wav, num_speakers=mode)
                    elapsed = time.time() - started

                    speakers = {}
                    for speaker, spans in speaker_spans(turns).items():
                        info = {"speech": round(sum(e - s for s, e in spans), 1), "turns": len(spans)}
                        if me is not None:
                            vector = embedder.embed(wav, longest_spans(spans, SPEAKER_EMBED_SECONDS))
                            info["similarity_to_enroll"] = round(cosine_similarity(vector, me), 3)
                        speakers[speaker] = info
                    runs[key] = {
                        "seconds": round(elapsed, 1),
                        "realtime_factor": round(elapsed / duration, 3),
                        "speakers": speakers,
                        "turns": [t.model_dump() for t in turns],
                    }
                    print(f"[{Path(audio).name}] {key}: {elapsed:.1f}s, {len(speakers)} speakers", flush=True)

            keys = list(runs)
            report["files"][Path(audio).name] = {
                "duration": round(duration, 1),
                "runs": runs,
                "agreement": {
                    f"{a} | {b}": round(agreement(
                        [SpeakerTurn(**t) for t in runs[a]["turns"]],
                        [SpeakerTurn(**t) for t in runs[b]["turns"]],
                        duration,
                    ), 3)
                    for a, b in itertools.combinations(keys, 2)
                },
            }
            if args.out:
                Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def print_summary(report: dict) -> None:
    for name, result in report["files"].items():
        print(f"\n== {name} ({result['duration']}s)")
        for key, run_ in result["runs"].items():
            speakers = ", ".join(
                f"{s} {i['speech']}s"
                + (f" sim={i['similarity_to_enroll']}" if "similarity_to_enroll" in i else "")
                for s, i in sorted(run_["speakers"].items(), key=lambda x: -x[1]["speech"])
            )
            print(f"  {key:60s} {run_['seconds']:7.1f}s (x{run_['realtime_factor']})  {speakers}")
        for pair, value in result["agreement"].items():
            print(f"  agreement {pair}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", nargs="+")
    parser.add_argument("--engines", nargs="+", choices=["pyannote", "sherpa_onnx"], default=["pyannote", "sherpa_onnx"])
    parser.add_argument("--sherpa-embedding", nargs="*", default=[], help="ONNX embedding models to compare")
    parser.add_argument("--num-speakers", nargs="+", default=["auto"], help="'auto' and/or fixed counts")
    parser.add_argument("--enroll", help="audio of the user's voice only")
    parser.add_argument("--seconds", type=float, help="only use the first N seconds")
    parser.add_argument("--out", help="write the full report (with turns) as JSON")
    args = parser.parse_args()

    print_summary(run(args))


if __name__ == "__main__":
    main()
