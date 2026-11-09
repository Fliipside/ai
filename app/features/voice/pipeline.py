import asyncio
import tempfile
import wave
from pathlib import Path

from app.core.config import Settings
from app.core.errors import (
    AppError,
    AudioTooLongError,
    InvalidAudioError,
    VoiceBusyError,
    VoiceprintMismatchError,
)
from app.features.voice.merge import UNKNOWN_SPEAKER, assign_speakers
from app.features.voice.speaker_id import identify_roles, label_speakers, longest_spans
from app.providers.diarization import create_diarizer
from app.providers.diarization.base import SpeakerDiarizer
from app.providers.speaker_embedding import create_embedder
from app.providers.speaker_embedding.base import SpeakerEmbedder
from app.providers.stt.base import SpeechToText
from app.providers.stt.groq import GroqSpeechToText
from app.schemas.voice import (
    Speaker,
    SpeakerRole,
    SpeakerTurn,
    TranscriptSegment,
    Utterance,
    VoiceAnalysis,
    Voiceprint,
)
from app.utils.audio import (
    SAMPLE_RATE,
    AudioChunk,
    split_wav,
    to_wav_16k_mono,
    wav_duration,
)

# Seconds of each speaker's audio compared with the voiceprint.
SPEAKER_EMBED_SECONDS = 60


class VoicePipeline:
    """audio -> STT + diarization -> "me" identification -> merged utterances."""

    def __init__(
        self,
        stt: SpeechToText,
        diarizer: SpeakerDiarizer,
        embedder: SpeakerEmbedder,
        match_threshold: float | None = None,
        *,
        max_audio_seconds: float | None = None,
        enroll_min_seconds: float = 0.0,
        busy_retry_seconds: int = 60,
    ) -> None:
        self.stt = stt
        self.diarizer = diarizer
        self.embedder = embedder
        self.match_threshold = match_threshold
        self.max_audio_seconds = max_audio_seconds
        self.enroll_min_seconds = enroll_min_seconds
        self.busy_retry_seconds = busy_retry_seconds
        # One analysis at a time: the CPU models are the bottleneck, and a caller that
        # timed out may resend while the first run is still going.
        self._analyze_lock = asyncio.Lock()

    async def analyze(
        self,
        audio: Path,
        voiceprint: Voiceprint | None = None,
    ) -> VoiceAnalysis:
        if voiceprint is not None and self.match_threshold is None:
            raise AppError(
                status_code=503,
                code="VOICE_NOT_CONFIGURED",
                message="VOICE_MATCH_THRESHOLD가 설정되지 않았습니다.",
            )
        if voiceprint is not None and voiceprint.engine != self.embedder.name:
            raise VoiceprintMismatchError()
        if self._analyze_lock.locked():
            raise VoiceBusyError(self.busy_retry_seconds)

        async with self._analyze_lock:
            with tempfile.TemporaryDirectory() as tmp:
                wav = await asyncio.to_thread(to_wav_16k_mono, audio, Path(tmp) / "audio.wav")
                if self.max_audio_seconds and wav_duration(wav) > self.max_audio_seconds:
                    raise AudioTooLongError(self.max_audio_seconds)
                chunks = await asyncio.to_thread(split_wav, wav, Path(tmp))
                # STT first: it is quick, and a rate limit then fails before minutes of diarization.
                transcript = await self._transcribe(chunks)
                turns = await asyncio.to_thread(self.diarizer.diarize, wav)
                roles = await self._identify_roles(wav, turns, voiceprint)

        merged = assign_speakers(transcript, turns)
        # Number speakers in the order they first speak in the transcript; speakers
        # with no transcribed sentence come after.
        labels = label_speakers(
            [speaker for _, speaker in merged if speaker != UNKNOWN_SPEAKER]
            + [turn.speaker for turn in sorted(turns, key=lambda turn: turn.start)],
            roles,
        )
        speech: dict[str, float] = {}
        for turn in turns:
            speech[turn.speaker] = speech.get(turn.speaker, 0.0) + turn.end - turn.start
        return VoiceAnalysis(
            speakers=[
                Speaker(
                    speaker=label,
                    role=roles.get(speaker, "unknown"),
                    speech_seconds=round(speech[speaker], 1),
                )
                for speaker, label in labels.items()
            ],
            utterances=[
                Utterance(
                    start=segment.start,
                    end=segment.end,
                    speaker=labels.get(speaker, speaker),
                    role=roles.get(speaker, "unknown"),
                    text=segment.text,
                )
                for segment, speaker in merged
            ],
        )

    async def enroll(self, audio: Path) -> Voiceprint:
        with tempfile.TemporaryDirectory() as tmp:
            wav = await asyncio.to_thread(to_wav_16k_mono, audio, Path(tmp) / "audio.wav")
            if wav_duration(wav) < self.enroll_min_seconds:
                raise InvalidAudioError(
                    f"목소리 등록에는 {self.enroll_min_seconds:g}초 이상의 음성이 필요합니다."
                )
            vector = await asyncio.to_thread(self.embedder.embed, wav)
        return Voiceprint(engine=self.embedder.name, vector=vector)

    async def _transcribe(self, chunks: list[AudioChunk]) -> list[TranscriptSegment]:
        # Sequential on purpose: the Groq free plan is rate limited.
        transcript = []
        for chunk in chunks:
            for segment in await self.stt.transcribe(chunk.path):
                transcript.append(
                    TranscriptSegment(
                        start=segment.start + chunk.offset,
                        end=segment.end + chunk.offset,
                        text=segment.text,
                    )
                )
        return transcript

    async def _identify_roles(
        self,
        wav: Path,
        turns: list[SpeakerTurn],
        voiceprint: Voiceprint | None,
    ) -> dict[str, SpeakerRole]:
        if voiceprint is None:
            return {}

        spans: dict[str, list[tuple[float, float]]] = {}
        for turn in turns:
            spans.setdefault(turn.speaker, []).append((turn.start, turn.end))
        
        embeddings: dict[str, list[float]] = {}

        for speaker, speaker_spans in spans.items():
            try:
                embeddings[speaker] = await asyncio.to_thread(
                    self.embedder.embed,
                    wav,
                    longest_spans(speaker_spans, SPEAKER_EMBED_SECONDS),
                )
            except InvalidAudioError:
                continue
        
        if any(len(vector) != len(voiceprint.vector) for vector in embeddings.values()):
            # Same engine name but another embedding size: it was made by a different model.
            raise VoiceprintMismatchError()
        return identify_roles(
            embeddings,
            voiceprint,
            engine=self.embedder.name,
            threshold=self.match_threshold,
        )


def build_voice_pipeline(settings: Settings) -> VoicePipeline:
    # Providers load their models on first use (or at startup with VOICE_PRELOAD_MODELS).
    return VoicePipeline(
        GroqSpeechToText(settings.groq_api_key.get_secret_value(), settings.groq_stt_model),
        create_diarizer(settings),
        create_embedder(settings),
        match_threshold=settings.voice_match_threshold,
        max_audio_seconds=settings.voice_max_audio_seconds,
        enroll_min_seconds=settings.voice_enroll_min_seconds,
        busy_retry_seconds=settings.voice_busy_retry_seconds,
    )


def preload_models(pipeline: VoicePipeline) -> None:
    """Run the local models once on a second of silence so they are loaded and warm."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "silence.wav"
        with wave.open(str(wav), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(SAMPLE_RATE)
            out.writeframes(b"\x00\x00" * SAMPLE_RATE)
        pipeline.diarizer.diarize(wav)
        pipeline.embedder.embed(wav)
