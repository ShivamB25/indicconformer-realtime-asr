from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from app.vad.base import (
    VADCapacityError,
    VADClosedError,
    VADConfigurationError,
    VADInferenceError,
)
from app.vad.silero import SileroVADProvider, _create_onnx_session


@dataclass(slots=True)
class TensorMetadata:
    name: str
    type: str
    shape: list[int | str | None]


class FakeSession:
    def __init__(self) -> None:
        self.providers = ["CPUExecutionProvider"]
        self.inputs = [
            TensorMetadata("input", "tensor(float)", [None, None]),
            TensorMetadata("state", "tensor(float)", [2, None, 128]),
            TensorMetadata("sr", "tensor(int64)", []),
        ]
        self.outputs = [
            TensorMetadata("output", "tensor(float)", [None, 1]),
            TensorMetadata("stateN", "tensor(float)", [None, None, None]),
        ]
        self.calls: list[dict[str, np.ndarray[Any, Any]]] = []
        self.failures_remaining = 0

    def get_providers(self) -> list[str]:
        return self.providers

    def get_inputs(self) -> list[TensorMetadata]:
        return self.inputs

    def get_outputs(self) -> list[TensorMetadata]:
        return self.outputs

    def run(
        self, output_names: list[str], input_feed: dict[str, np.ndarray[Any, Any]]
    ) -> list[np.ndarray[Any, Any]]:
        assert output_names == ["output", "stateN"]
        assert input_feed["input"].shape == (1, 576)
        assert input_feed["input"].dtype == np.float32
        assert input_feed["state"].shape == (2, 1, 128)
        assert input_feed["state"].dtype == np.float32
        assert input_feed["sr"].shape == ()
        assert input_feed["sr"].dtype == np.int64
        self.calls.append({name: value.copy() for name, value in input_feed.items()})
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("injected inference failure")
        probability = np.asarray([[min(len(self.calls) / 10.0, 1.0)]], dtype=np.float32)
        return [probability, input_feed["state"] + np.float32(1.0)]


class FakeMetrics:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.ended: list[str] = []

    def set_vad_queue_depth(self, depth: int) -> None:
        del depth

    def record_vad_queue_wait(self, provider: str, seconds: float) -> None:
        del provider, seconds

    def record_vad_inference(self, provider: str, seconds: float) -> None:
        del provider, seconds

    def record_vad_runtime_error(self, provider: str, code: str) -> None:
        del provider, code

    def vad_stream_started(self, provider: str) -> None:
        self.started.append(provider)

    def vad_stream_ended(self, provider: str) -> None:
        self.ended.append(provider)


def make_provider(
    tmp_path: Path,
    session: FakeSession,
    *,
    max_streams: int = 4,
    metrics: FakeMetrics | None = None,
) -> SileroVADProvider:
    model_path = tmp_path / "silero_vad.onnx"
    model_bytes = b"deterministic fake ONNX artifact"
    model_path.write_bytes(model_bytes)
    return SileroVADProvider(
        model_path=model_path,
        model_sha256=hashlib.sha256(model_bytes).hexdigest(),
        max_streams=max_streams,
        workers=1,
        pending_capacity=4,
        deadline_seconds=1.0,
        metrics=metrics,
        session_factory=cast(Any, lambda path: session),
    )


def pcm_frame(samples: np.ndarray[Any, Any]) -> bytes:
    return np.asarray(samples, dtype="<i2").tobytes()


def constant_frame(sample_rate: int, value: int = 0) -> bytes:
    return pcm_frame(np.full(sample_rate // 50, value, dtype=np.int16))


@pytest.mark.asyncio
async def test_digest_is_verified_before_session_construction(tmp_path: Path) -> None:
    model_path = tmp_path / "silero_vad.onnx"
    model_path.write_bytes(b"wrong model")
    constructed: list[Path] = []
    provider = SileroVADProvider(
        model_path=model_path,
        model_sha256="0" * 64,
        max_streams=1,
        workers=1,
        pending_capacity=1,
        deadline_seconds=1.0,
        session_factory=lambda path: constructed.append(path),  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(VADConfigurationError, match="SHA-256"):
        await provider.startup()
    assert constructed == []
    await provider.close()


@pytest.mark.asyncio
async def test_symlink_artifact_is_rejected_before_session_construction(tmp_path: Path) -> None:
    target = tmp_path / "target.onnx"
    target.write_bytes(b"model")
    link = tmp_path / "silero_vad.onnx"
    link.symlink_to(target)
    constructed: list[Path] = []
    provider = SileroVADProvider(
        model_path=link,
        model_sha256=hashlib.sha256(b"model").hexdigest(),
        max_streams=1,
        workers=1,
        pending_capacity=1,
        deadline_seconds=1.0,
        session_factory=lambda path: constructed.append(path),  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(VADConfigurationError, match="non-symlink"):
        await provider.startup()
    assert constructed == []
    await provider.close()


def test_default_session_factory_pins_threads_and_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    class SessionOptions:
        inter_op_num_threads = 0
        intra_op_num_threads = 0

    sentinel = object()

    def inference_session(
        path: str, *, sess_options: SessionOptions, providers: list[str]
    ) -> object:
        captured.update(path=path, options=sess_options, providers=providers)
        return sentinel

    fake_ort = SimpleNamespace(SessionOptions=SessionOptions, InferenceSession=inference_session)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    model_path = tmp_path / "model.onnx"

    assert _create_onnx_session(model_path) is sentinel
    assert captured["path"] == str(model_path)
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["options"].inter_op_num_threads == 1
    assert captured["options"].intra_op_num_threads == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contract_fault", ["provider", "input_type", "state_shape", "output_shape"]
)
async def test_session_provider_and_tensor_contract_are_validated(
    tmp_path: Path, contract_fault: str
) -> None:
    session = FakeSession()
    if contract_fault == "provider":
        session.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif contract_fault == "input_type":
        session.inputs[0].type = "tensor(double)"
    elif contract_fault == "state_shape":
        session.inputs[1].shape = [1, None, 128]
    else:
        session.outputs[0].shape = [None, 2]
    provider = make_provider(tmp_path, session)

    with pytest.raises(VADConfigurationError):
        await provider.startup()
    await provider.close()


@pytest.mark.asyncio
async def test_320_sample_frames_are_consumed_once_without_padding_and_score_is_held(
    tmp_path: Path,
) -> None:
    session = FakeSession()
    provider = make_provider(tmp_path, session)
    await provider.startup()
    stream = provider.new_stream(16_000)
    source = np.arange(4 * 320, dtype=np.int16)
    frames = [pcm_frame(source[index : index + 320]) for index in range(0, source.size, 320)]

    scores = [await stream.score(frame) for frame in frames]

    assert scores == pytest.approx([0.0, 0.1, 0.1, 0.2])
    assert len(session.calls) == 2
    consumed = np.concatenate([call["input"][0, 64:] for call in session.calls])
    expected = source[:1024].astype(np.float32) / np.float32(32768.0)
    np.testing.assert_array_equal(consumed, expected)
    np.testing.assert_array_equal(session.calls[0]["input"][0, :64], np.zeros(64))
    np.testing.assert_array_equal(
        session.calls[1]["input"][0, :64], session.calls[0]["input"][0, -64:]
    )

    stream.close()
    await provider.close()


@pytest.mark.asyncio
async def test_24khz_resampling_preserves_every_frame_without_a_delayed_tail(
    tmp_path: Path,
) -> None:
    session = FakeSession()
    provider = make_provider(tmp_path, session)
    await provider.startup()
    stream = provider.new_stream(24_000)
    generator = np.random.default_rng(42)
    frames = [
        pcm_frame(generator.integers(-20_000, 20_000, 480, dtype=np.int16)) for _ in range(30)
    ]
    expected_parts: list[np.ndarray[Any, Any]] = []

    for frame in frames:
        normalized = np.frombuffer(frame, dtype="<i2").astype(np.float32) / np.float32(32768.0)
        groups = normalized.reshape(-1, 3)
        expected = np.empty(320, dtype=np.float32)
        expected[0::2] = (groups[:, 0] * 2.0 + groups[:, 1]) / 3.0
        expected[1::2] = (groups[:, 1] + groups[:, 2] * 2.0) / 3.0
        expected_parts.append(expected)
        await stream.score(frame)

    all_expected = np.concatenate(expected_parts)
    consumed = np.concatenate([call["input"][0, 64:] for call in session.calls])
    consumed_count = all_expected.size // 512 * 512
    assert consumed.size == consumed_count
    np.testing.assert_array_equal(consumed, all_expected[:consumed_count])

    stream.close()
    await provider.close()


@pytest.mark.asyncio
async def test_24khz_frame_accounting_has_no_unflushed_resampler_tail(tmp_path: Path) -> None:
    session = FakeSession()
    provider = make_provider(tmp_path, session)
    await provider.startup()
    stream = provider.new_stream(24_000)

    for _ in range(8):
        await stream.score(constant_frame(24_000))

    assert len(session.calls) == 5
    stream.close()
    await provider.close()


@pytest.mark.asyncio
async def test_recurrent_state_is_isolated_between_streams(tmp_path: Path) -> None:
    session = FakeSession()
    provider = make_provider(tmp_path, session)
    await provider.startup()
    first = provider.new_stream(16_000)
    second = provider.new_stream(16_000)
    frame = constant_frame(16_000)

    for stream in (first, second, first, second):
        await stream.score(frame)
    for stream in (first, second, first, second):
        await stream.score(frame)

    state_markers = [float(call["state"][0, 0, 0]) for call in session.calls]
    assert state_markers == [0.0, 0.0, 1.0, 1.0]

    first.close()
    second.close()
    await provider.close()


@pytest.mark.asyncio
async def test_reset_and_close_are_idempotent_and_release_one_lease(tmp_path: Path) -> None:
    session = FakeSession()
    metrics = FakeMetrics()
    provider = make_provider(tmp_path, session, metrics=metrics)
    await provider.startup()
    stream = provider.new_stream(16_000)
    frame = constant_frame(16_000)
    assert provider.active_streams == 1
    assert metrics.started == ["silero"]

    await stream.score(frame)
    await stream.score(frame)
    stream.reset()
    stream.reset()
    assert await stream.score(frame) == 0.0
    await stream.score(frame)
    assert float(session.calls[-1]["state"][0, 0, 0]) == 0.0

    stream.close()
    stream.close()
    stream.reset()
    assert provider.active_streams == 0
    assert metrics.ended == ["silero"]
    with pytest.raises(VADClosedError):
        await stream.score(frame)

    await provider.close()
    await provider.close()
    with pytest.raises(VADClosedError):
        provider.new_stream(16_000)


@pytest.mark.asyncio
async def test_stream_capacity_is_hard_bounded(tmp_path: Path) -> None:
    session = FakeSession()
    provider = make_provider(tmp_path, session, max_streams=1)
    await provider.startup()
    stream = provider.new_stream(16_000)

    with pytest.raises(VADCapacityError):
        provider.new_stream(16_000)
    stream.close()
    replacement = provider.new_stream(16_000)
    replacement.close()
    await provider.close()


@pytest.mark.asyncio
async def test_malformed_frames_are_rejected_without_poisoning_stream(tmp_path: Path) -> None:
    session = FakeSession()
    provider = make_provider(tmp_path, session)
    await provider.startup()
    stream = provider.new_stream(16_000)

    for malformed in (b"\x00" * 638, b"\x00" * 642, bytearray(640)):
        with pytest.raises(ValueError):
            await stream.score(malformed)  # type: ignore[arg-type]
    assert session.calls == []
    await stream.score(constant_frame(16_000))
    assert await stream.score(constant_frame(16_000)) == pytest.approx(0.1)

    stream.close()
    await provider.close()


@pytest.mark.asyncio
async def test_runtime_error_fails_closed_until_reset(tmp_path: Path) -> None:
    session = FakeSession()
    session.failures_remaining = 1
    provider = make_provider(tmp_path, session)
    await provider.startup()
    stream = provider.new_stream(16_000)
    frame = constant_frame(16_000)

    await stream.score(frame)
    with pytest.raises(VADInferenceError):
        await stream.score(frame)
    call_count = len(session.calls)
    with pytest.raises(VADInferenceError, match="requires reset"):
        await stream.score(frame)
    assert len(session.calls) == call_count

    stream.reset()
    await stream.score(frame)
    assert 0.0 <= await stream.score(frame) <= 1.0

    stream.close()
    await provider.close()
