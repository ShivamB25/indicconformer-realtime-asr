"""Structured logging with safe request/session context."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TextIO, cast

import structlog
from structlog.contextvars import (
    bind_contextvars,
    bound_contextvars,
    clear_contextvars,
    merge_contextvars,
    unbind_contextvars,
)
from structlog.stdlib import BoundLogger, ProcessorFormatter
from structlog.typing import EventDict, Processor, WrappedLogger

from app.core.types import LanguageCode, ProcessingMode

_REDACTED = "[REDACTED]"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SENSITIVE_KEYS = frozenset(
    {
        "audio",
        "audio_bytes",
        "binary_frame",
        "body",
        "frame",
        "hypothesis",
        "payload",
        "pcm",
        "raw_content",
        "samples",
        "text",
        "tokens",
        "transcript",
        "transcription",
        "waveform",
    }
)


def _sensitive_key(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or "transcript" in normalized


def _safe_value(key: object, value: object) -> object:
    if _sensitive_key(key) or isinstance(value, (bytes, bytearray, memoryview)):
        return _REDACTED
    if isinstance(value, Mapping):
        return {str(child_key): _safe_value(child_key, child) for child_key, child in value.items()}
    if isinstance(value, list):
        return [_safe_value("item", item) for item in value]
    if isinstance(value, tuple):
        return tuple(_safe_value("item", item) for item in value)
    return value


def redact_sensitive(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Remove raw audio/transcript fields before any renderer sees them."""

    del logger, method_name
    for key in tuple(event_dict):
        if key not in {"_record", "_from_structlog"}:
            event_dict[key] = _safe_value(key, event_dict[key])
    return event_dict


def _safe_exception(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Record only exception type, never exception text or attached values."""

    del logger, method_name
    raw = event_dict.pop("exc_info", None)
    exception_type: type[BaseException] | None = None
    if raw is True:
        exception_type = sys.exc_info()[0]
    elif isinstance(raw, BaseException):
        exception_type = type(raw)
    elif isinstance(raw, tuple) and raw and isinstance(raw[0], type):
        candidate = raw[0]
        if issubclass(candidate, BaseException):
            exception_type = candidate
    if exception_type is not None:
        event_dict["exception_type"] = exception_type.__name__
    return event_dict


def _log_level(level: str | int) -> int:
    if isinstance(level, bool):
        raise ValueError("log level must be a name or integer")
    if isinstance(level, int):
        return level
    resolved = logging.getLevelNamesMapping().get(level.upper())
    if resolved is None:
        raise ValueError(f"unknown log level: {level}")
    return resolved


def configure_logging(
    level: str | int = "INFO",
    *,
    json_output: bool = True,
    stream: TextIO | None = None,
) -> None:
    """Configure one process-wide structured handler, replacing stale handlers."""

    numeric_level = _log_level(level)
    shared: list[Processor] = [
        merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _safe_exception,
        redact_sensitive,
    ]
    renderer: Processor
    if json_output:
        renderer = structlog.processors.JSONRenderer(sort_keys=True)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    formatter = ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(numeric_level)

    root = logging.getLogger()
    for existing in tuple(root.handlers):
        root.removeHandler(existing)
        existing.close()
    root.addHandler(handler)
    root.setLevel(numeric_level)

    structlog.configure(
        processors=[*shared, ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> BoundLogger:
    """Return the configured structured logger type."""

    return cast(BoundLogger, structlog.get_logger(name))


def _validated_identifier(value: str, field: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field} must be a safe identifier of at most 128 characters")
    return value


def clear_log_context() -> None:
    clear_contextvars()


def bind_request_context(request_id: str) -> None:
    """Bind only the generated request identifier to the current context."""

    bind_contextvars(request_id=_validated_identifier(request_id, "request_id"))


def unbind_request_context() -> None:
    unbind_contextvars("request_id")


def bind_session_context(
    session_id: str,
    language: str | LanguageCode,
    mode: str | ProcessingMode,
) -> None:
    """Bind safe session metadata; raw audio/text are not accepted."""

    bind_contextvars(
        session_id=_validated_identifier(session_id, "session_id"),
        language=LanguageCode(language).value,
        mode=ProcessingMode(mode).value,
    )


def unbind_session_context() -> None:
    unbind_contextvars("session_id", "language", "mode")


@contextmanager
def request_log_context(request_id: str) -> Iterator[None]:
    """Temporarily bind request context and restore its previous value."""

    with bound_contextvars(request_id=_validated_identifier(request_id, "request_id")):
        yield


@contextmanager
def session_log_context(
    session_id: str,
    language: str | LanguageCode,
    mode: str | ProcessingMode,
) -> Iterator[None]:
    """Temporarily bind bounded session metadata."""

    with bound_contextvars(
        session_id=_validated_identifier(session_id, "session_id"),
        language=LanguageCode(language).value,
        mode=ProcessingMode(mode).value,
    ):
        yield
