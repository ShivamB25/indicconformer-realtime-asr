"""Wire-level driver for the GA OpenAI transcription WebSocket."""

from __future__ import annotations

import base64
from typing import Any

from starlette.testclient import WebSocketTestSession

from tests.support.realtime import RealtimeDriver

OPENAI_TRANSCRIPTION_PATH = "/v1/realtime/transcription_sessions"


class OpenAIRealtimeDriver(RealtimeDriver):
    def __init__(self, session: WebSocketTestSession) -> None:
        super().__init__(session)

    def update(
        self,
        *,
        language: str = "hi",
        model: str = "indicconformer-600m",
        turn_detection: dict[str, Any] | None = None,
        event_id: str = "update-1",
    ) -> None:
        self.send_json(
            {
                "type": "session.update",
                "event_id": event_id,
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": {"model": model, "languages": [language]},
                            "turn_detection": turn_detection,
                        }
                    },
                },
            }
        )

    def append(self, pcm: bytes, event_id: str = "append-1") -> None:
        self.send_json(
            {
                "type": "input_audio_buffer.append",
                "event_id": event_id,
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )

    def commit(self, event_id: str = "commit-1") -> None:
        self.send_json({"type": "input_audio_buffer.commit", "event_id": event_id})

    def clear(self, event_id: str = "clear-1") -> None:
        self.send_json({"type": "input_audio_buffer.clear", "event_id": event_id})

    def expect_error(self, code: str, timeout: float | None = None) -> dict[str, Any]:
        event = self.expect("error", timeout)
        assert event["error"]["code"] == code
        return event
