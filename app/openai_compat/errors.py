"""OpenAI-compatible failures without coupling native API error contracts."""

from collections.abc import Mapping

from fastapi.responses import JSONResponse

from app.openai_compat.schemas import OpenAIErrorDetail, OpenAIErrorEnvelope


class OpenAIError(Exception):
    """A fully specified OpenAI error response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.type = error_type
        self.param = param
        self.code = code

    def envelope(self) -> OpenAIErrorEnvelope:
        return OpenAIErrorEnvelope(
            error=OpenAIErrorDetail(
                message=self.message,
                type=self.type,
                param=self.param,
                code=self.code,
            )
        )


def openai_error_response(
    error: OpenAIError,
    *,
    request_id: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    response_headers = dict(headers or {})
    if request_id is not None:
        response_headers["x-request-id"] = request_id
    return JSONResponse(
        status_code=error.status_code,
        content=error.envelope().model_dump(mode="json"),
        headers=response_headers,
    )
