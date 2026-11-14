# [API 입구] 기능별 API 주소 묶음
# voice(음성 분석)와 memory(대화 기억) 라우터를 /api/v1 아래에 연결합니다.

from fastapi import APIRouter

from app.api.v1 import memory, voice

router = APIRouter()
router.include_router(voice.router, prefix="/voice", tags=["voice"])
router.include_router(memory.router, prefix="/memory", tags=["memory"])
