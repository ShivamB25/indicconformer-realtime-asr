"""OpenAPI metadata and Swagger-specific documentation helpers."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

SERVICE_DESCRIPTION = """
AI4Bharat IndicConformer speech-to-text service for 22 supported Indic languages.

### Try a transcription in Swagger

1. Open **Authorize** and enter the bearer token configured for the service. Local
   keyless development does not require a token.
2. Check `GET /health/ready`; transcription endpoints return `503` until it reports
   `ready`.
3. Expand `POST /v1/transcribe`, select **Try it out**, upload mono 16 kHz PCM16 WAV
   (or headerless PCM16LE), choose a language and processing mode, then execute.
4. OpenAI SDK users can instead use `POST /v1/audio/transcriptions` with model
   `ai4bharat/indic-conformer-600m-multilingual`.

### Realtime transports

OpenAPI cannot execute WebSocket protocols. Two authenticated WebSocket endpoints
are available:

- `WS /v1/realtime`: binary PCM16LE, mono, 16 kHz, exact 20 ms / 640-byte
  frames; native `session.start` and `input.commit` JSON control events.
- `WS /v1/realtime/transcription_sessions`: base64 PCM16LE, mono, 24 kHz;
  OpenAI-compatible transcription-session events.

WebSocket clients send `Authorization: Bearer <token>` during the upgrade when an
API key is configured. ASR and VAD model weights are provisioned separately and
are never uploaded through this API.
"""

OPENAPI_TAGS: list[dict[str, str]] = [
    {
        "name": "transcription",
        "description": (
            "Native batch transcription. Best for direct service integrations with "
            "known mono 16 kHz PCM input."
        ),
    },
    {
        "name": "openai",
        "description": (
            "OpenAI-compatible model discovery and batch transcription for existing "
            "OpenAI Python/JavaScript SDK clients."
        ),
    },
    {
        "name": "health",
        "description": (
            "Public, inference-free liveness and readiness probes for operators and orchestrators."
        ),
    },
]

OPENAI_TRANSCRIPTION_REQUEST_BODY: dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file", "model", "language"],
                    "properties": {
                        "file": {
                            "type": "string",
                            "format": "binary",
                            "description": (
                                "Audio file supported by FFmpeg/PyAV, such as WAV, MP3, "
                                "FLAC, M4A, or OGG."
                            ),
                        },
                        "model": {
                            "type": "string",
                            "default": "ai4bharat/indic-conformer-600m-multilingual",
                            "description": "Canonical model ID or indicconformer-600m alias.",
                        },
                        "language": {
                            "type": "string",
                            "example": "hi",
                            "description": "Required supported ISO-style language code.",
                        },
                        "response_format": {
                            "type": "string",
                            "enum": ["json", "text"],
                            "default": "json",
                        },
                        "stream": {
                            "type": "boolean",
                            "default": False,
                            "description": "Must be false; streaming is a WebSocket feature.",
                        },
                        "temperature": {
                            "type": "number",
                            "enum": [0],
                            "default": 0,
                            "description": "Only deterministic temperature 0 is supported.",
                        },
                    },
                }
            }
        },
    }
}

_AUTHENTICATED_PATHS = frozenset(
    {
        "/v1/transcribe",
        "/v1/audio/transcriptions",
        "/v1/models",
        "/v1/models/{model}",
    }
)


def configure_openapi(application: FastAPI) -> None:
    """Install the documented conditional bearer scheme without changing auth runtime."""

    def build_schema() -> dict[str, Any]:
        if application.openapi_schema is not None:
            return application.openapi_schema
        schema = get_openapi(
            title=application.title,
            version=application.version,
            summary=application.summary,
            description=application.description,
            routes=application.routes,
            tags=application.openapi_tags,
            servers=application.servers,
        )
        components = schema.setdefault("components", {})
        security_schemes = components.setdefault("securitySchemes", {})
        security_schemes["BearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "service API key",
            "description": (
                "Required when ASR_API_KEY_FILE is configured; optional for keyless local "
                "development."
            ),
        }
        for path in _AUTHENTICATED_PATHS:
            for operation in schema.get("paths", {}).get(path, {}).values():
                if isinstance(operation, dict):
                    operation["security"] = [{"BearerAuth": []}, {}]
        schema["x-websocket-endpoints"] = [
            {
                "url": "/v1/realtime",
                "protocol": "native-pcm16-16khz",
                "documentation": "Service description and README realtime protocol section",
            },
            {
                "url": "/v1/realtime/transcription_sessions",
                "protocol": "openai-transcription-session-pcm16-24khz",
                "documentation": "Service description and README OpenAI realtime section",
            },
        ]
        application.openapi_schema = schema
        return schema

    application.openapi = build_schema  # type: ignore[method-assign]
