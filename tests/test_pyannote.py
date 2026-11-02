import os
import sys
from types import SimpleNamespace

import pytest

from app.core.errors import AppError, InvalidAudioError
from app.providers.diarization import pyannote as diarization
from app.providers.speaker_embedding import pyannote as embedding
from app.utils.audio import SAMPLE_RATE
from tests.helpers import write_tone


@pytest.fixture(autouse=True)
def clear_model_cache():
    diarization.load_pipeline.cache_clear()
    embedding.load_inference.cache_clear()
    yield
    diarization.load_pipeline.cache_clear()
    embedding.load_inference.cache_clear()


def test_missing_pyannote_is_not_configured(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)

    with pytest.raises(AppError) as exc:
        diarization.load_pipeline("model", "", 0.5)

    assert exc.value.status_code == 503
    assert exc.value.code == "DIARIZATION_NOT_CONFIGURED"


def test_model_load_failure_is_not_configured(monkeypatch) -> None:
    def from_pretrained(*args, **kwargs):
        raise OSError("401 Client Error: gated repo")

    fake = SimpleNamespace(Pipeline=SimpleNamespace(from_pretrained=from_pretrained))
    monkeypatch.setattr(diarization, "import_pyannote", lambda: fake)

    with pytest.raises(AppError) as exc:
        diarization.load_pipeline("model", "bad-token", 0.5)

    assert exc.value.status_code == 503
    assert exc.value.code == "DIARIZATION_NOT_CONFIGURED"


def test_metrics_are_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("PYANNOTE_METRICS_ENABLED", raising=False)
    monkeypatch.setitem(sys.modules, "pyannote.audio", SimpleNamespace())

    diarization.import_pyannote()

    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "false"


class FakeAnnotation:
    def __init__(self, tracks):
        self.tracks = tracks

    def itertracks(self, yield_label=False):
        for start, end, label in self.tracks:
            yield SimpleNamespace(start=start, end=end), None, label


def test_diarize_uses_exclusive_turns(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    wav = write_tone(tmp_path / "a.wav", 1.0, rate=SAMPLE_RATE, channels=1)
    output = SimpleNamespace(
        speaker_diarization=FakeAnnotation([(0.0, 1.0, "A"), (0.5, 1.0, "B")]),
        exclusive_speaker_diarization=FakeAnnotation([(0.0, 0.5, "A"), (0.5, 1.0, "B")]),
    )
    calls = []

    def pipeline(file, num_speakers=None):
        calls.append((file, num_speakers))
        return output

    monkeypatch.setattr(diarization, "load_pipeline", lambda model, token, step: pipeline)

    turns = diarization.PyannoteDiarizer(hf_token="").diarize(wav, num_speakers=2)

    assert [(t.start, t.end, t.speaker) for t in turns] == [(0.0, 0.5, "A"), (0.5, 1.0, "B")]
    file, num_speakers = calls[0]
    assert file["sample_rate"] == SAMPLE_RATE
    assert tuple(file["waveform"].shape) == (1, SAMPLE_RATE)
    assert num_speakers == 2


def test_embed_only_given_spans(tmp_path, monkeypatch) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    wav = write_tone(tmp_path / "a.wav", 2.0, rate=SAMPLE_RATE, channels=1)
    lengths = []

    def inference(file):
        lengths.append(file["waveform"].shape[1])
        return np.array([[0.5, -0.5]], dtype=np.float32)

    monkeypatch.setattr(embedding, "load_inference", lambda model, token: inference)
    embedder = embedding.PyannoteEmbedder(hf_token="")

    assert embedder.embed(wav, [(0.0, 0.5), (1.0, 1.25)]) == [0.5, -0.5]
    assert embedder.embed(wav) == [0.5, -0.5]
    assert lengths == [int(0.75 * SAMPLE_RATE), 2 * SAMPLE_RATE]


def test_embed_rejects_empty_spans(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    wav = write_tone(tmp_path / "a.wav", 1.0, rate=SAMPLE_RATE, channels=1)
    monkeypatch.setattr(embedding, "load_inference", lambda model, token: None)

    with pytest.raises(InvalidAudioError):
        embedding.PyannoteEmbedder(hf_token="").embed(wav, [(0.5, 0.5)])


def test_load_pipeline_sets_window_step(monkeypatch) -> None:
    inference = SimpleNamespace(duration=10.0, step=1.0)
    loaded = SimpleNamespace(_segmentation=inference)
    fake = SimpleNamespace(Pipeline=SimpleNamespace(from_pretrained=lambda *args, **kwargs: loaded))
    monkeypatch.setattr(diarization, "import_pyannote", lambda: fake)

    assert diarization.load_pipeline("model", "", 0.5) is loaded
    assert inference.step == 5.0
