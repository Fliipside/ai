# [AI 모델] LLM 고르기
# 설정(LLM_API_KEY, LLM_MODEL, LLM_BASE_URL, LLM_REASONING_EFFORT)을 보고 사용할 LLM을 만들어 줍니다.

from app.core.config import Settings
from app.providers.llm.base import StructuredLLM


def create_llm(settings: Settings) -> StructuredLLM:
    from app.providers.llm.gateway import GatewayLLM

    return GatewayLLM(
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        reasoning_effort=settings.llm_reasoning_effort,
    )
