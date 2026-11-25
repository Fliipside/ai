import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.router import router as api_router
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.features.voice.pipeline import build_voice_pipeline, preload_models
from app.schemas.response import ApiResponse


logger = logging.getLogger(__name__)

HTTP_ERROR_CODES = {
    400: "INVALID_REQUEST",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "INVALID_REQUEST",
    429: "TOO_MANY_REQUESTS",
    503: "SERVICE_UNAVAILABLE",
}


def get_http_error_code(status_code: int) -> str:
    return HTTP_ERROR_CODES.get(status_code, "HTTP_ERROR")


def create_error_content(code: str, message: str) -> dict[str, object]:
    response = ApiResponse[None].fail(
        code=code,
        message=message,
    )
    return response.model_dump(mode="json")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    pipeline = build_voice_pipeline(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if settings.voice_preload_models:
            try:
                await asyncio.to_thread(preload_models, pipeline)
            except Exception:
                # Keep serving /health; voice requests report the real error themselves.
                logger.exception("Failed to preload voice models")
        yield

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.settings = settings
    app.state.voice_pipeline = pipeline

    @app.exception_handler(AppError)
    async def app_error_handler(
        _request: Request,
        exc: AppError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=create_error_content(
                code=exc.code,
                message=exc.message,
            ),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request,
        _exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=create_error_content(
                code="INVALID_REQUEST",
                message="요청 값이 올바르지 않습니다.",
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        _request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        message = (
            exc.detail
            if isinstance(exc.detail, str)
            else "요청을 처리할 수 없습니다."
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=create_error_content(
                code=get_http_error_code(exc.status_code),
                message=message,
            ),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        _request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.error(
            "Unhandled exception",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            status_code=500,
            content=create_error_content(
                code="INTERNAL_SERVER_ERROR",
                message="서버 내부 오류가 발생했습니다.",
            ),
        )

    @app.get("/health", tags=["health"], response_model=ApiResponse[dict[str, str]])
    def health() -> ApiResponse[dict[str, str]]:
        return ApiResponse[dict[str, str]].ok(
            data={"status": "ok"}
        )

    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
