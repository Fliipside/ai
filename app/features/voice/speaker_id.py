import math
from collections.abc import Iterable

from app.schemas.voice import SpeakerRole, Voiceprint


def identify_roles(
    speaker_embeddings: dict[str, list[float]],
    voiceprint: Voiceprint,
    engine: str,
    threshold: float,
) -> dict[str, SpeakerRole]:
    # Mark the speaker most similar to the user's voiceprint (above threshold) as "me".
    if voiceprint.engine != engine:
        raise ValueError(
            f"voiceprint was made by {voiceprint.engine!r}, not {engine!r}"
        )

    scores = {
        speaker: cosine_similarity(embedding, voiceprint.vector)
        for speaker, embedding in speaker_embeddings.items()
    }
    if not scores:
        return {}

    me = max(scores, key=scores.__getitem__)
    if scores[me] < threshold:
        # The user was not found, so we cannot tell who the customer is either.
        return {speaker: "unknown" for speaker in scores}
    return {speaker: "me" if speaker == me else "customer" for speaker in scores}


def label_speakers(
    speakers: Iterable[str],
    roles: dict[str, SpeakerRole],
) -> dict[str, str]:
    # Engine labels (e.g. "SPEAKER_01") are arbitrary per run; name the user "me" and
    # everyone else "A", "B", "C"... in the order given (first appearance).
    labels: dict[str, str] = {}
    others = 0
    for speaker in speakers:
        if speaker in labels:
            continue
        if roles.get(speaker) == "me":
            labels[speaker] = "me"
        else:
            labels[speaker] = speaker_letters(others)
            others += 1
    return labels


def speaker_letters(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, like spreadsheet columns."""
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def longest_spans(
    spans: list[tuple[float, float]],
    max_seconds: float,
) -> list[tuple[float, float]]:
    # Embedding a whole meeting per speaker is slow and some models reject very long input;
    # the longest turns (least likely to include the other voice) are enough to compare.
    picked = []
    total = 0.0
    for start, end in sorted(spans, key=lambda span: span[1] - span[0], reverse=True):
        if total >= max_seconds:
            break
        end = min(end, start + max_seconds - total)
        picked.append((start, end))
        total += end - start
    return sorted(picked)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"embedding size mismatch: {len(a)} != {len(b)}")
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / norm
