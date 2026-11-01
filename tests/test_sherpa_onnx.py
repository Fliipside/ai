import sys
from types import SimpleNamespace

import pytest

from app.core.errors import AppError, InvalidAudioError
from app.providers.diarization import sherpa_onnx as diarization
from app.providers.speaker_embedding import sherpa_onnx as embedding
from app.utils.audio import SAMPLE_RATE
from tests.helpers import write_tone


@pytest.fixture(autouse=True)
def clear_model_cache():
    diarization.load_diarizer.cache_clear()
    embedding.load_extractor.cache_clear()
    yield
    diarization.load_diarizer.cache_clear()
    embedding.load_extractor.cache_clear()


def test_missing_sherpa_onnx_is_not_configured(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)

    with pytest.raises(AppError) as exc:
        diarization.load_diarizer("seg.onnx", "emb.onnx", -1, 0.5, 1)

    assert exc.value.status_code == 503
    assert exc.value.code == "DIARIZATION_NOT_CONFIGURED"


def test_missing_model_file_is_not_configured(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace())

    with pytest.raises(AppError) as exc:
        embedding.load_extractor(str(tmp_path / "missing.onnx"), 1)

    assert exc.value.code == "DIARIZATION_NOT_CONFIGURED"
    assert "SHERPA_EMBEDDING_MODEL" in exc.value.message


def test_diarize_labels_speakers(tmp_path, monkeypatch) -> None:
    pytest.importorskip("numpy")
    wav = write_tone(tmp_path / "a.wav", 1.0, rate=SAMPLE_RATE, channels=1)
    segments = [SimpleNamespace(start=0.0, end=0.5, speaker=0), SimpleNamespace(start=0.5, end=1.0, speaker=1)]
    calls = []

    class FakeDiarizer:
        def process(self, samples):
            calls.append(len(samples))
            return SimpleNamespace(sort_by_start_time=lambda: segments)

    def load(seg, emb, num_clusters, threshold, threads):
        calls.append(num_clusters)
        return FakeDiarizer()

    monkeypatch.setattr(diarization, "load_diarizer", load)
    diarizer = diarization.SherpaOnnxDiarizer("seg.onnx", "emb.onnx")

    turns = diarizer.diarize(wav)
    diarizer.diarize(wav, num_speakers=2)

    assert [(t.start, t.end, t.speaker) for t in turns] == [(0.0, 0.5, "SPEAKER_00"), (0.5, 1.0, "SPEAKER_01")]
    assert calls == [-1, SAMPLE_RATE, 2, SAMPLE_RATE]


class FakeStream:
    def __init__(self) -> None:
        self.samples = None

    def accept_waveform(self, sample_rate, waveform):
        self.samples = waveform

    def input_finished(self):
        pass


class FakeExtractor:
    def __init__(self) -> None:
        self.lengths = []

    def create_stream(self):
        return FakeStream()

    def is_ready(self, stream):
        return len(stream.samples) > 0

    def compute(self, stream):
        self.lengths.append(len(stream.samples))
        return [0.5, -0.5]


def test_embed_only_given_spans(tmp_path, monkeypatch) -> None:
    pytest.importorskip("numpy")
    wav = write_tone(tmp_path / "a.wav", 2.0, rate=SAMPLE_RATE, channels=1)
    extractor = FakeExtractor()
    monkeypatch.setattr(embedding, "load_extractor", lambda model, threads: extractor)
    embedder = embedding.SherpaOnnxEmbedder("/models/wespeaker.onnx")

    assert embedder.embed(wav, [(0.0, 0.5), (1.0, 1.25)]) == [0.5, -0.5]
    assert extractor.lengths == [int(0.75 * SAMPLE_RATE)]
    assert embedder.name == "sherpa_onnx:wespeaker"


def test_embed_rejects_too_short_audio(tmp_path, monkeypatch) -> None:
    pytest.importorskip("numpy")
    wav = write_tone(tmp_path / "a.wav", 1.0, rate=SAMPLE_RATE, channels=1)
    monkeypatch.setattr(embedding, "load_extractor", lambda model, threads: FakeExtractor())

    with pytest.raises(InvalidAudioError):
        embedding.SherpaOnnxEmbedder("emb.onnx").embed(wav, [(0.5, 0.5)])
