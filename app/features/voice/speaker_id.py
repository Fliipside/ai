import math

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


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"embedding size mismatch: {len(a)} != {len(b)}")
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / norm
