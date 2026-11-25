from app.features.voice.merge import UNKNOWN_SPEAKER, assign_speakers
from app.schemas.voice import SpeakerTurn, TranscriptSegment


def seg(start: float, end: float) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text="t")


def turn(start: float, end: float, speaker: str) -> SpeakerTurn:
    return SpeakerTurn(start=start, end=end, speaker=speaker)


def speakers(transcript, turns) -> list[str]:
    return [speaker for _, speaker in assign_speakers(transcript, turns)]


def test_picks_largest_overlap() -> None:
    turns = [turn(0, 2, "A"), turn(2, 5, "B")]

    assert speakers([seg(0, 1.5), seg(1.5, 4)], turns) == ["A", "B"]


def test_sums_overlap_per_speaker() -> None:
    # A: 1 + 1 = 2s, B: 1.5s
    turns = [turn(0, 1, "A"), turn(1, 2.5, "B"), turn(2.5, 3.5, "A")]

    assert speakers([seg(0, 3.5)], turns) == ["A"]


def test_falls_back_to_nearest_turn() -> None:
    turns = [turn(0, 1, "A"), turn(3, 4, "B")]

    assert speakers([seg(1.2, 1.5), seg(2.6, 2.9)], turns) == ["A", "B"]


def test_unknown_without_turns() -> None:
    assert speakers([seg(0, 1)], []) == [UNKNOWN_SPEAKER]
