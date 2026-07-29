"""A thin, assertion-friendly driver for the realtime WebSocket protocol.

The driver speaks only the wire contract: JSON text for control events and raw
binary for PCM frames. It never imports the session implementation, so a
refactor inside the server cannot make these tests pass vacuously.

Every read is bounded. A server that stops talking fails a test with a
description instead of hanging a CI job, and ``expect_silence`` turns "the
server must not send anything here" into a real assertion.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import anyio
from anyio import ClosedResourceError, EndOfStream
from starlette.testclient import WebSocketTestSession

REALTIME_PATH = "/v1/realtime"
RECEIVE_TIMEOUT_SECONDS = 10.0
SILENCE_TIMEOUT_SECONDS = 0.25


@dataclass(frozen=True, slots=True)
class Received:
    """One observed server message: either a protocol event or a close."""

    kind: str
    event: dict[str, Any] | None = None
    code: int | None = None

    @property
    def type(self) -> str:
        if self.event is None:
            return "<close>"
        return cast(str, self.event.get("type", "<missing>"))


def start_payload(
    *,
    language: Any = "hi",
    mode: Any = None,
    vad: Any = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a session.start payload, with room for deliberately bad fields."""

    payload: dict[str, Any] = {"type": "session.start", "language": language}
    if mode is not None:
        payload["mode"] = mode
    if vad is not None:
        payload["vad"] = vad
    payload.update(overrides)
    return payload


async def _receive_within(session: WebSocketTestSession, timeout_seconds: float) -> dict[str, Any]:
    with anyio.fail_after(timeout_seconds):
        return cast("dict[str, Any]", await session._send_rx.receive())  # noqa: SLF001


class RealtimeDriver:
    """Drive one WebSocket session and collect what the server sent back."""

    def __init__(
        self,
        session: WebSocketTestSession,
        *,
        timeout: float = RECEIVE_TIMEOUT_SECONDS,
    ) -> None:
        self._session = session
        self._timeout = timeout
        self.received: list[Received] = []

    def send_start(self, **kwargs: Any) -> None:
        self.send_json(start_payload(**kwargs))

    def send_json(self, payload: dict[str, Any]) -> None:
        self._session.send_text(json.dumps(payload))

    def send_text(self, text: str) -> None:
        self._session.send_text(text)

    def send_commit(self) -> None:
        self.send_json({"type": "input.commit"})

    def send_frame(self, frame: bytes) -> None:
        self._session.send_bytes(frame)

    def send_frames(self, frames: Iterable[bytes]) -> None:
        for frame in frames:
            self._session.send_bytes(frame)

    def _raw_receive(self, timeout: float) -> dict[str, Any] | None:
        """Return the next ASGI message, or ``None`` once nothing arrives."""

        try:
            raw = self._session.portal.call(partial(_receive_within, self._session, timeout))
            return cast("dict[str, Any] | None", raw)
        except (TimeoutError, EndOfStream, ClosedResourceError):
            return None

    def next_message(self, timeout: float | None = None) -> Received:
        raw = self._raw_receive(self._timeout if timeout is None else timeout)
        if raw is None:
            raise AssertionError(
                f"server sent nothing; observed so far: {self.event_types(self.received)}"
            )
        message_type = raw.get("type")
        if message_type == "websocket.close":
            received = Received(kind="close", code=cast("int | None", raw.get("code")))
        elif raw.get("text") is not None:
            received = Received(kind="event", event=cast("dict[str, Any]", json.loads(raw["text"])))
        else:
            raise AssertionError(f"unexpected server message on the control channel: {raw!r}")
        self.received.append(received)
        return received

    def next_event(self, timeout: float | None = None) -> dict[str, Any]:
        message = self.next_message(timeout)
        if message.event is None:
            raise AssertionError(f"expected a protocol event, server closed with {message.code}")
        return message.event

    def expect(self, event_type: str, timeout: float | None = None) -> dict[str, Any]:
        event = self.next_event(timeout)
        assert event["type"] == event_type, f"expected {event_type}, received {event}"
        return event

    def expect_close(self, code: int, timeout: float | None = None) -> None:
        message = self.next_message(timeout)
        assert message.kind == "close", f"expected a close, received {message.event}"
        assert message.code == code, f"expected close code {code}, received {message.code}"

    def expect_error(self, code: str, timeout: float | None = None) -> dict[str, Any]:
        event = self.expect("error", timeout)
        assert event["code"] == code, f"expected error {code}, received {event}"
        return event

    def expect_silence(self, timeout: float = SILENCE_TIMEOUT_SECONDS) -> None:
        """Assert the server sends nothing at all within ``timeout``."""

        raw = self._raw_receive(timeout)
        if raw is not None:
            raise AssertionError(f"expected no server message, received {raw!r}")

    def collect_until(
        self, *stop_types: str, limit: int = 512, timeout: float | None = None
    ) -> list[Received]:
        """Read messages until one of ``stop_types`` (or a close) is observed."""

        wanted = set(stop_types)
        collected: list[Received] = []
        for _ in range(limit):
            message = self.next_message(timeout)
            collected.append(message)
            if message.kind == "close" or message.type in wanted:
                return collected
        raise AssertionError(f"never received any of {sorted(wanted)} within {limit} messages")

    def drain(self, timeout: float = SILENCE_TIMEOUT_SECONDS) -> list[Received]:
        """Read everything the server has already queued, then stop."""

        collected: list[Received] = []
        while True:
            raw = self._raw_receive(timeout)
            if raw is None:
                return collected
            if raw.get("type") == "websocket.close":
                received = Received(kind="close", code=cast("int | None", raw.get("code")))
            else:
                received = Received(
                    kind="event", event=cast("dict[str, Any]", json.loads(raw["text"]))
                )
            self.received.append(received)
            collected.append(received)

    @staticmethod
    def events_of(messages: Sequence[Received], event_type: str) -> list[dict[str, Any]]:
        return [
            message.event
            for message in messages
            if message.event is not None and message.event.get("type") == event_type
        ]

    @staticmethod
    def event_types(messages: Sequence[Received]) -> list[str]:
        return [message.type for message in messages]

    @staticmethod
    def close_code(messages: Sequence[Received]) -> int | None:
        for message in messages:
            if message.kind == "close":
                return message.code
        return None


__all__ = [
    "REALTIME_PATH",
    "RECEIVE_TIMEOUT_SECONDS",
    "RealtimeDriver",
    "Received",
    "start_payload",
]
