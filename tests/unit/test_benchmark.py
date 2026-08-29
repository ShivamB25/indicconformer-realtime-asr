from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from scripts import benchmark


class _Response:
    status = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = io.BytesIO(json.dumps(payload).encode())

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body.read()


def test_transcribe_sends_bearer_key_without_putting_it_in_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "a" * 48
    observed: dict[str, object] = {}

    def urlopen(request: urllib.request.Request, *, timeout: float) -> _Response:
        observed["authorization"] = request.get_header("Authorization")
        observed["body"] = request.data
        observed["timeout"] = str(timeout)
        return _Response(
            {
                "text": "not retained by the benchmark",
                "language": "hi",
                "mode": "hybrid",
                "decoder": "rnnt",
                "audio_duration_ms": 100,
                "inference_ms": 4.5,
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    sample = benchmark.Sample(Path("sample.wav"), "hi", 100, b"RIFF-audio")

    measurement = benchmark._transcribe(
        endpoint="https://asr.example/v1/transcribe",
        sample=sample,
        mode="hybrid",
        decoder="rnnt",
        timeout_seconds=5,
        api_key=token,
    )

    assert observed["authorization"] == f"Bearer {token}"
    assert isinstance(observed["body"], bytes)
    assert token.encode() not in observed["body"]
    assert measurement.inference_ms == 4.5


def test_http_failure_never_copies_key_or_response_reason_into_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "secret-token-that-must-not-escape-000000000000"

    def urlopen(_request: urllib.request.Request, *, timeout: float) -> _Response:
        del timeout
        raise urllib.error.HTTPError(
            "https://asr.example/v1/transcribe",
            401,
            f"rejected {token}",
            Message(),
            None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    sample = benchmark.Sample(Path("sample.wav"), "hi", 100, b"RIFF-audio")

    with pytest.raises(benchmark.BenchmarkError) as captured:
        benchmark._transcribe(
            endpoint="https://asr.example/v1/transcribe",
            sample=sample,
            mode="hybrid",
            decoder="rnnt",
            timeout_seconds=5,
            api_key=token,
        )

    assert str(captured.value) == "transcription returned HTTP 401"
    assert token not in str(captured.value)


def test_api_key_file_requires_one_regular_bounded_ascii_token(tmp_path: Path) -> None:
    token = "b" * 48
    key_file = tmp_path / "api-key"
    key_file.write_text(token + "\n", encoding="utf-8")

    assert benchmark._read_api_key(key_file) == token
    assert benchmark._read_api_key(None) is None

    symlink = tmp_path / "api-key-link"
    symlink.symlink_to(key_file)
    with pytest.raises(benchmark.BenchmarkError, match="regular file"):
        benchmark._read_api_key(symlink)

    key_file.write_text("contains whitespace " + "x" * 32, encoding="utf-8")
    with pytest.raises(benchmark.BenchmarkError, match="ASCII token"):
        benchmark._read_api_key(key_file)
