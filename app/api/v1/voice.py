from fastapi import APIRouter, File, UploadFile

from app.core.errors import FeatureNotReadyError
from app.schemas.response import ApiResponse
from app.schemas.voice import VoiceAnalysis, Voiceprint

router = APIRouter()


@router.post("/analyze", response_model=ApiResponse[VoiceAnalysis])
async def analyze(file: UploadFile = File(...)) -> ApiResponse[VoiceAnalysis]:
    # Transcribe, diarize and label "me" / "customer" with the user's voiceprint.
    raise FeatureNotReadyError("voice analyze")


@router.post("/enroll", response_model=ApiResponse[Voiceprint])
async def enroll(file: UploadFile = File(...)) -> ApiResponse[Voiceprint]:
    # Build the user's voiceprint; backend stores it (pgvector).
    raise FeatureNotReadyError("voice enroll")
