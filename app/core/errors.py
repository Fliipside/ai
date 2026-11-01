class AppError(Exception):

    def __init__(
            self,
            *,
            status_code: int,
            code: str,
            message: str,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
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
