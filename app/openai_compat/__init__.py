"""Shared OpenAI compatibility constants and errors."""

from app.openai_compat.constants import MODEL_ALIAS, MODEL_ID
from app.openai_compat.errors import OpenAIError


def validate_model(model: str) -> str:
    """Resolve the canonical model ID or its truthful short alias."""

    if model in {MODEL_ID, MODEL_ALIAS}:
        return MODEL_ID
    raise OpenAIError(
        f"The model `{model}` does not exist",
        status_code=404,
        error_type="invalid_request_error",
        param="model",
        code="model_not_found",
    )


__all__ = ["MODEL_ALIAS", "MODEL_ID", "OpenAIError", "validate_model"]
