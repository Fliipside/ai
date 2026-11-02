import asyncio
import tempfile
from pathlib import Path

from app.core.errors import AppError, InvalidAudioError
from app.features.voice.merge import UNKNOWN_SPEAKER, assign_speakers
from app.features.voice.speaker_id import identify_roles, label_speakers, longest_spans
from app.providers.diarization.base import SpeakerDiarizer
from app.providers.speaker_embedding.base import SpeakerEmbedder
from app.providers.stt.base import SpeechToText
from app.schemas.voice import (
    Speaker,
    SpeakerRole,
    SpeakerTurn,
    TranscriptSegment,
    Utterance,
    VoiceAnalysis,
    Voiceprint,
)
from app.utils.audio import AudioChunk, split_wav, to_wav_16k_mono

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
    ) -> None:
        self.stt = stt
        self.diarizer = diarizer
        self.embedder = embedder
        self.match_threshold = match_threshold

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
            raise AppError(
                status_code=400,
                code="VOICEPRINT_ENGINE_MISMATCH",
                message="다른 화자 분리 엔진으로 만든 목소리 정보입니다. 다시 등록해 주세요.",
            )

        with tempfile.TemporaryDirectory() as tmp:
            wav = await asyncio.to_thread(to_wav_16k_mono, audio, Path(tmp) / "audio.wav")
            chunks = await asyncio.to_thread(split_wav, wav, Path(tmp))
            transcript, turns = await asyncio.gather(
                self._transcribe(chunks),
                asyncio.to_thread(self.diarizer.diarize, wav),
            )
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
        
        return identify_roles(
            embeddings,
            voiceprint,
            engine=self.embedder.name,
            threshold=self.match_threshold,
        )
