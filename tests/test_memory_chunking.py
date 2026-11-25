from app.features.memory.chunking import chunk_utterances
from app.schemas.memory import StoredUtterance


def utt(i: int, text: str = "가" * 50, role: str = "customer") -> StoredUtterance:
    return StoredUtterance(id=f"U{i}", start=i * 10.0, end=i * 10.0 + 5, speaker="S", role=role, text=text)


def ids(chunks) -> list[list[str]]:
    return [c.utterance_ids for c in chunks]


def test_groups_until_max_chars_with_overlap() -> None:
    # Each line is "[고객] " + 50 chars = 55 chars; three lines plus newlines fit in 170.
    chunks = chunk_utterances([utt(i) for i in range(6)], max_chars=170, overlap=1)

    assert ids(chunks) == [["U0", "U1", "U2"], ["U2", "U3", "U4"], ["U4", "U5"]]


def test_without_overlap() -> None:
    chunks = chunk_utterances([utt(i) for i in range(6)], max_chars=170, overlap=0)

    assert ids(chunks) == [["U0", "U1", "U2"], ["U3", "U4", "U5"]]


def test_long_utterance_is_its_own_chunk() -> None:
    chunks = chunk_utterances([utt(0), utt(1, "나" * 500), utt(2)], max_chars=170, overlap=1)

    assert ids(chunks) == [["U0"], ["U1"], ["U2"]]


def test_overlap_larger_than_window_still_advances() -> None:
    chunks = chunk_utterances([utt(i) for i in range(4)], max_chars=120, overlap=5)

    assert ids(chunks) == [["U0", "U1"], ["U1", "U2"], ["U2", "U3"]]


def test_keeps_time_span_and_role_labels() -> None:
    chunks = chunk_utterances([utt(0, "안녕하세요", "me"), utt(1, "네", "customer")])

    assert chunks[0].text == "[나] 안녕하세요\n[고객] 네"
    assert (chunks[0].start, chunks[0].end) == (0.0, 15.0)


def test_empty() -> None:
    assert chunk_utterances([]) == []
