"""Sanitized process logging and Uvicorn integration contracts."""

from __future__ import annotations

import io
import json
import logging

import pytest

from app.core.logging import configure_logging


def _isolated_logger_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[logging.RootLogger, dict[str, logging.Logger]]:
    root = logging.RootLogger(logging.WARNING)
    loggers: dict[str, logging.Logger] = {}

    def resolve(name: str | None = None) -> logging.Logger:
        if name is None:
            return root
        logger = loggers.get(name)
        if logger is not None:
            return logger
        logger = logging.Logger(name)
        parent_name = name.rpartition(".")[0]
        logger.parent = resolve(parent_name) if parent_name else root
        loggers[name] = logger
        return logger

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        resolve(name).addHandler(logging.StreamHandler(io.StringIO()))
    root.addHandler(logging.StreamHandler(io.StringIO()))
    monkeypatch.setattr(logging, "getLogger", resolve)
    return root, loggers


def _single_payload(stream: io.StringIO) -> tuple[str, dict[str, object]]:
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    return lines[0], json.loads(lines[0])


def test_uvicorn_exception_is_redacted_and_emitted_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    root, loggers = _isolated_logger_tree(monkeypatch)
    secret = "raw transcript must never appear"

    configure_logging(stream=stream)
    try:
        raise RuntimeError(secret)
    except RuntimeError:
        logging.getLogger("uvicorn.error").exception("ASGI failed: %s", secret)

    rendered, payload = _single_payload(stream)
    timestamp = payload.pop("timestamp")
    assert isinstance(timestamp, str)
    assert payload == {
        "event": "server_log",
        "exception_type": "RuntimeError",
        "level": "error",
        "logger": "uvicorn.error",
    }
    assert secret not in rendered
    assert len(root.handlers) == 1
    assert all(logger.handlers == [] for logger in loggers.values())
    assert all(logger.propagate is True for logger in loggers.values())


def test_uvicorn_access_keeps_bounded_metadata_without_request_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    _isolated_logger_tree(monkeypatch)
    secret_target = "/v1/audio/transcriptions?api_key=raw-secret"

    configure_logging(stream=stream)
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d',
        "127.0.0.1:1234",
        "GET",
        secret_target,
        "1.1",
        503,
    )

    rendered, payload = _single_payload(stream)
    timestamp = payload.pop("timestamp")
    assert isinstance(timestamp, str)
    assert payload == {
        "event": "http_access",
        "http_method": "GET",
        "level": "info",
        "logger": "uvicorn.access",
        "status_code": 503,
    }
    assert secret_target not in rendered
