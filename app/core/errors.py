class AppError(Exception):

    def __init__(
            self,
            *,
            status_code: int,
            code: str,
            message: str,
            headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = headers
        super().__init__(message)


class FeatureNotReadyError(AppError):

    def __init__(self, feature: str) -> None:
        super().__init__(
                status_code=501,
                code="FEATURE_NOT_READY",
                message=f"{feature} AI workflow is not implemented yet",
            )


class InvalidAudioError(AppError):

    def __init__(self, message: str) -> None:
        super().__init__(
                status_code=400,
                code="INVALID_AUDIO",
                message=message,
            )


class AudioTooLongError(AppError):

    def __init__(self, max_seconds: float) -> None:
        super().__init__(
                status_code=400,
                code="AUDIO_TOO_LONG",
                message=f"녹음은 최대 {max_seconds / 60:g}분까지 분석할 수 있습니다.",
            )


class VoiceprintMismatchError(AppError):

    def __init__(self) -> None:
        super().__init__(
                status_code=400,
                code="VOICEPRINT_ENGINE_MISMATCH",
                message="다른 화자 분리 엔진으로 만든 목소리 정보입니다. 다시 등록해 주세요.",
            )


class VoiceBusyError(AppError):
    # Not a failure: the caller should come back after Retry-After without counting an attempt.

    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__(
                status_code=503,
                code="VOICE_BUSY",
                message="다른 음성 분석이 진행 중입니다. 잠시 후 다시 시도해 주세요.",
                headers={"Retry-After": str(retry_after_seconds)},
            )
