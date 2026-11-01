from fastapi import APIRouter

from app.api.v1 import voice

router = APIRouter()
router.include_router(voice.router, prefix="/voice", tags=["voice"])
