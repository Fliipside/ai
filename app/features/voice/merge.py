from app.schemas.voice import SpeakerTurn, TranscriptSegment

UNKNOWN_SPEAKER = "unknown"


def assign_speakers(
    transcript: list[TranscriptSegment],
    turns: list[SpeakerTurn],
) -> list[tuple[TranscriptSegment, str]]:
    return [(segment, _pick_speaker(segment, turns)) for segment in transcript]


def _pick_speaker(segment: TranscriptSegment, turns: list[SpeakerTurn]) -> str:
    # Pick the speaker whose turns overlap the segment the most.
    overlaps: dict[str, float] = {}
    for turn in turns:
        overlap = min(segment.end, turn.end) - max(segment.start, turn.start)
        if overlap > 0:
            overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + overlap
    if overlaps:
        return max(overlaps, key=overlaps.__getitem__)

    # No overlap (e.g. a short gap between turns): fall back to the nearest turn.
    if not turns:
        return UNKNOWN_SPEAKER
    nearest = min(turns, key=lambda turn: _gap(segment, turn))
    return nearest.speaker


def _gap(segment: TranscriptSegment, turn: SpeakerTurn) -> float:
    return max(turn.start - segment.end, segment.start - turn.end, 0.0)
