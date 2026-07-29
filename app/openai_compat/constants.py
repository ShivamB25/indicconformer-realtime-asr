"""Stable identifiers and route predicates for the OpenAI compatibility API."""

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
MODEL_ALIAS = "indicconformer-600m"
MODEL_OWNER = "ai4bharat"
MODEL_CREATED = 0

_OPENAI_ROUTE_PREFIXES = (
    "/v1/audio/transcriptions",
    "/v1/models",
    "/v1/realtime/transcription_sessions",
)


def is_openai_route(path: str) -> bool:
    """Return whether a path belongs to an OpenAI-shaped compatibility surface."""

    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in _OPENAI_ROUTE_PREFIXES)
