import asyncio
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePath
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from pydantic import ValidationError

from app.core.errors import AppError
from app.features.voice.pipeline import VoicePipeline
from app.schemas.response import ApiResponse
from app.schemas.voice import VoiceAnalysis, Voiceprint

router = APIRouter()


def get_voice_pipeline(request: Request) -> VoicePipeline:
    return request.app.state.voice_pipeline


@router.post("/analyze", response_model=ApiResponse[VoiceAnalysis])
async def analyze(
    file: Annotated[UploadFile, File()],
    pipeline: Annotated[VoicePipeline, Depends(get_voice_pipeline)],
    voiceprint: Annotated[str | None, Form()] = None,
) -> ApiResponse[VoiceAnalysis]:
    # Transcribe, diarize and label "me" and everyone else (A, B, C...) per person.
    parsed = parse_voiceprint(voiceprint)
    async with saved_upload(file) as audio:
        result = await pipeline.analyze(audio, parsed)
    return ApiResponse[VoiceAnalysis].ok(data=result)


@router.post("/enroll", response_model=ApiResponse[Voiceprint])
async def enroll(
    file: Annotated[UploadFile, File()],
    pipeline: Annotated[VoicePipeline, Depends(get_voice_pipeline)],
) -> ApiResponse[Voiceprint]:
    # Build the user's voiceprint; backend stores it and sends it back with analyze.
    async with saved_upload(file) as audio:
        result = await pipeline.enroll(audio)
    return ApiResponse[Voiceprint].ok(data=result)


def parse_voiceprint(value: str | None) -> Voiceprint | None:
    # backend omits the field when the user has not enrolled.
    if value is None:
        return None
    try:
        return Voiceprint.model_validate_json(value)
    except ValidationError as exc:
        raise AppError(
            status_code=400,
            code="INVALID_REQUEST",
            message="voiceprint 형식이 올바르지 않습니다.",
        ) from exc


@asynccontextmanager
async def saved_upload(file: UploadFile) -> AsyncIterator[Path]:
    """Copy the upload to a temp file so ffmpeg can read it by path."""
    with tempfile.TemporaryDirectory() as tmp:
        # The extension is only a hint; ffmpeg detects the format from the content.
        path = Path(tmp) / f"upload{PurePath(file.filename or '').suffix[:10]}"
        await asyncio.to_thread(_copy, file.file, path)
        yield path


def _copy(source, path: Path) -> None:
    source.seek(0)
    with path.open("wb") as out:
        shutil.copyfileobj(source, out)
