"""Pre-parser HTTP admission contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
from starlette.types import Message, Receive, Scope, Send

from app.api.http_admission import MULTIPART_OVERHEAD_BYTES, HTTPAdmissionMiddleware
from app.observability.metrics import MetricCode


class MetricsDouble:
    def __init__(self) -> None:
        self.rejections: list[MetricCode] = []

    def record_rejection(self, code: MetricCode) -> None:
        self.rejections.append(code)


async def _consume_body(scope: Scope, receive: Receive, send: Send) -> None:
    del scope
    while True:
        message = await receive()
        if not message.get("more_body", False):
            break
    await send({"type": "http.response.start", "status": 204, "headers": []})
    await send({"type": "http.response.body", "body": b""})


def _run_request(
    *,
    path: str,
    chunks: list[bytes],
    api_key: str | None = None,
    authorization: bytes | None = None,
    content_length: int | None = None,
) -> tuple[list[dict[str, Any]], int, MetricsDouble]:
    metrics = MetricsDouble()
    settings = SimpleNamespace(
        max_upload_bytes=2,
        api_key_file=Path("configured.key") if api_key is not None else None,
    )
    application = SimpleNamespace(
        state=SimpleNamespace(settings=settings, api_key=api_key, metrics=metrics)
    )
    headers: list[tuple[bytes, bytes]] = []
    if authorization is not None:
        headers.append((b"authorization", authorization))
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode("ascii")))
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1),
        "server": ("testserver", 80),
        "app": application,
    }
    sent: list[dict[str, Any]] = []
    receive_calls = 0

    async def receive() -> Message:
        nonlocal receive_calls
        receive_calls += 1
        if receive_calls > len(chunks):
            raise AssertionError("the middleware read beyond the supplied request body")
        return {
            "type": "http.request",
            "body": chunks[receive_calls - 1],
            "more_body": receive_calls < len(chunks),
        }

    async def send(message: Message) -> None:
        sent.append(dict(message))

    async def exercise() -> None:
        middleware = HTTPAdmissionMiddleware(_consume_body)
        await middleware(scope, receive, send)

    anyio.run(exercise)
    return sent, receive_calls, metrics


def _response(sent: list[dict[str, Any]]) -> tuple[int, dict[str, str], dict[str, Any]]:
    start, body = sent
    headers = {name.decode("latin-1"): value.decode("latin-1") for name, value in start["headers"]}
    return start["status"], headers, json.loads(body["body"])


def test_invalid_authentication_returns_before_reading_any_body() -> None:
    sent, receive_calls, _ = _run_request(
        path="/v1/transcribe",
        chunks=[b"must not be consumed"],
        api_key="a" * 32,
        authorization=b"Bearer wrong",
        content_length=10_000_000,
    )

    status_code, headers, body = _response(sent)
    assert status_code == 401
    assert headers["www-authenticate"] == "Bearer"
    assert set(body) == {"error", "request_id"}
    assert receive_calls == 0


def test_openai_authentication_uses_the_safe_schema_without_reading_the_body() -> None:
    sent, receive_calls, _ = _run_request(
        path="/v1/audio/transcriptions",
        chunks=[b"must not be consumed"],
        api_key="a" * 32,
        content_length=10_000_000,
    )

    status_code, headers, body = _response(sent)
    assert status_code == 401
    assert headers["www-authenticate"] == "Bearer"
    assert headers["x-request-id"]
    assert set(body) == {"error"}
    assert set(body["error"]) == {"message", "type", "param", "code"}
    assert receive_calls == 0


def test_declared_oversize_is_rejected_without_reading_the_body() -> None:
    limit = 2 + MULTIPART_OVERHEAD_BYTES
    sent, receive_calls, metrics = _run_request(
        path="/v1/transcribe",
        chunks=[b"must not be consumed"],
        content_length=limit + 1,
    )

    status_code, _, body = _response(sent)
    assert status_code == 413
    assert set(body) == {"error", "request_id"}
    assert receive_calls == 0
    assert metrics.rejections == [MetricCode.UPLOAD_TOO_LARGE]


def test_chunked_oversize_stops_after_observing_limit_plus_one() -> None:
    limit = 2 + MULTIPART_OVERHEAD_BYTES
    sent, receive_calls, metrics = _run_request(
        path="/v1/audio/transcriptions",
        chunks=[b"a" * limit, b"b", b"must not be consumed"],
    )

    status_code, headers, body = _response(sent)
    assert status_code == 413
    assert headers["x-request-id"]
    assert body["error"]["code"] == "upload_too_large"
    assert receive_calls == 2
    assert metrics.rejections == [MetricCode.UPLOAD_TOO_LARGE]


def test_public_http_paths_are_not_authenticated_or_body_limited() -> None:
    sent, receive_calls, metrics = _run_request(
        path="/health/live",
        chunks=[b"public"],
        api_key="a" * 32,
        content_length=10_000_000,
    )

    assert sent[0]["status"] == 204
    assert receive_calls == 1
    assert metrics.rejections == []
