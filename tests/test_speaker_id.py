import pytest

from app.features.voice.speaker_id import cosine_similarity, identify_roles
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
