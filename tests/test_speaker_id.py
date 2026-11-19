import pytest

from app.features.voice.speaker_id import (
    cosine_similarity,
    identify_roles,
    label_speakers,
    longest_spans,
    speaker_letters,
)
from app.schemas.voice import Voiceprint

VOICEPRINT = Voiceprint(engine="pyannote", vector=[1.0, 0.0])


def test_most_similar_speaker_is_me() -> None:
    embeddings = {"S0": [0.1, 1.0], "S1": [0.9, 0.1]}

    roles = identify_roles(embeddings, VOICEPRINT, engine="pyannote", threshold=0.5)

    assert roles == {"S0": "customer", "S1": "me"}


def test_unknown_when_below_threshold() -> None:
    embeddings = {"S0": [0.0, 1.0], "S1": [-1.0, 0.2]}

    roles = identify_roles(embeddings, VOICEPRINT, engine="pyannote", threshold=0.5)

    assert roles == {"S0": "unknown", "S1": "unknown"}


def test_rejects_other_engine() -> None:
    with pytest.raises(ValueError):
        identify_roles({"S0": [1.0, 0.0]}, VOICEPRINT, engine="sherpa_onnx", threshold=0.5)


def test_cosine_similarity() -> None:
    assert cosine_similarity([1, 0], [2, 0]) == pytest.approx(1.0)
    assert cosine_similarity([0, 0], [1, 0]) == 0.0
    with pytest.raises(ValueError):
        cosine_similarity([1, 0], [1, 0, 0])


def test_longest_spans_caps_total_seconds() -> None:
    spans = [(0.0, 2.0), (3.0, 13.0), (20.0, 25.0), (30.0, 31.0)]

    assert longest_spans(spans, max_seconds=12) == [(3.0, 13.0), (20.0, 22.0)]
    assert longest_spans(spans, max_seconds=100) == sorted(spans)


def test_label_speakers_names_each_person_in_order_of_appearance() -> None:
    order = ["SPEAKER_01", "SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
    roles = {"SPEAKER_00": "me", "SPEAKER_01": "customer", "SPEAKER_02": "customer"}

    assert label_speakers(order, roles) == {
        "SPEAKER_01": "A",
        "SPEAKER_00": "me",
        "SPEAKER_02": "B",
    }
    assert label_speakers(order, {}) == {
        "SPEAKER_01": "A",
        "SPEAKER_00": "B",
        "SPEAKER_02": "C",
    }


def test_speaker_letters_continue_after_z() -> None:
    assert [speaker_letters(i) for i in (0, 1, 25, 26, 27, 701, 702)] == [
        "A", "B", "Z", "AA", "AB", "ZZ", "AAA",
    ]
