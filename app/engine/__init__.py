"""Inference engine interfaces and lightweight test implementation."""

from app.engine.base import (
    BaseEngine,
    Engine,
    EngineReadiness,
    EngineState,
    TranscriptionRequest,
    TranscriptionResult,
)
from app.engine.mock import MockEngine

__all__ = [
    "BaseEngine",
    "Engine",
    "EngineReadiness",
    "EngineState",
    "MockEngine",
    "TranscriptionRequest",
    "TranscriptionResult",
]
