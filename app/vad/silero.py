"""Streaming Silero v6.2.1 VAD backed directly by ONNX Runtime on CPU."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import math
import os
import stat
from collections import deque
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from app.vad.artifact import SILERO_VAD_MODEL_SHA256
from app.vad.base import (
    StreamCapacity,
    VADCapacityError,
    VADClosedError,
    VADConfigurationError,
    VADInferenceError,
    VADSampleRate,
    expected_frame_bytes,
)
from app.vad.runtime import BoundedVADRuntime, VADRuntimeMetrics

_MODEL_SAMPLE_RATE = 16_000
_WINDOW_SAMPLES = 512
_CONTEXT_SAMPLES = 64
_STATE_SHAPE = (2, 1, 128)
_INITIAL_SCORE = 0.0
_PCM16_SCALE = np.float32(1.0 / 32_768.0)
_SAMPLE_RATE_TENSOR = np.asarray(_MODEL_SAMPLE_RATE, dtype=np.int64)
_SAMPLE_RATE_TENSOR.flags.writeable = False

_FloatArray = npt.NDArray[np.float32]


class _TensorMetadata(Protocol):
    name: str
    type: str
    shape: Sequence[int | str | None]


class _InferenceSession(Protocol):
    def get_providers(self) -> Sequence[str]: ...

    def get_inputs(self) -> Sequence[_TensorMetadata]: ...

    def get_outputs(self) -> Sequence[_TensorMetadata]: ...

    def run(
        self, output_names: Sequence[str], input_feed: dict[str, npt.NDArray[Any]]
    ) -> Sequence[npt.NDArray[Any]]: ...


class SileroVADMetrics(VADRuntimeMetrics, Protocol):
    def vad_stream_started(self, provider: str) -> None: ...

    def vad_stream_ended(self, provider: str) -> None: ...


SessionFactory = Callable[[Path], _InferenceSession]


def _create_onnx_session(model_path: Path) -> _InferenceSession:
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError as exc:
        raise VADConfigurationError("onnxruntime is required for Silero VAD") from exc

    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    return cast(
        _InferenceSession,
        ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        ),
    )


def _verify_artifact(path: Path, expected_sha256: str) -> None:
    if (
        len(expected_sha256) != 64
        or expected_sha256.lower() != expected_sha256
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise VADConfigurationError("Silero model SHA-256 must be 64 lowercase hex characters")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise VADConfigurationError("Silero model must be a readable non-symlink file") from exc

    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise VADConfigurationError("Silero model must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as artifact:
            while chunk := artifact.read(1024 * 1024):
                digest.update(chunk)
    except VADConfigurationError:
        raise
    except OSError as exc:
        raise VADConfigurationError("Silero model could not be read") from exc
    finally:
        os.close(descriptor)

    if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
        raise VADConfigurationError("Silero model SHA-256 does not match configuration")


def _validate_session(session: _InferenceSession) -> None:
    if list(session.get_providers()) != ["CPUExecutionProvider"]:
        raise VADConfigurationError("Silero VAD session must use only CPUExecutionProvider")

    inputs = list(session.get_inputs())
    outputs = list(session.get_outputs())
    if len(inputs) != 3 or len(outputs) != 2:
        raise VADConfigurationError("Silero VAD model has an unexpected tensor count")

    _validate_tensor(inputs[0], "input", "tensor(float)", rank=2)
    _validate_tensor(inputs[1], "state", "tensor(float)", rank=3, fixed={0: 2, 2: 128})
    _validate_tensor(inputs[2], "sr", "tensor(int64)", rank=0)
    _validate_tensor(outputs[0], "output", "tensor(float)", rank=2, fixed={1: 1})
    _validate_tensor(outputs[1], "stateN", "tensor(float)", rank=3)


def _validate_tensor(
    tensor: _TensorMetadata,
    name: str,
    element_type: str,
    *,
    rank: int,
    fixed: dict[int, int] | None = None,
) -> None:
    shape = list(tensor.shape)
    if tensor.name != name or tensor.type != element_type or len(shape) != rank:
        raise VADConfigurationError(f"Silero VAD tensor contract mismatch for {name}")
    for axis, expected in (fixed or {}).items():
        if shape[axis] != expected:
            raise VADConfigurationError(f"Silero VAD tensor shape mismatch for {name}")


class SileroVADProvider:
    """Process-owned immutable ONNX session and bounded CPU inference runtime."""

    __slots__ = (
        "_capacity",
        "_closed",
        "_metrics",
        "_model_path",
        "_model_sha256",
        "_runtime",
        "_session",
        "_session_factory",
        "_started",
    )

    def __init__(
        self,
        *,
        model_path: Path,
        model_sha256: str = SILERO_VAD_MODEL_SHA256,
        max_streams: int,
        workers: int,
        pending_capacity: int,
        deadline_seconds: float,
        metrics: SileroVADMetrics | None = None,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._model_sha256 = model_sha256
        self._capacity = StreamCapacity(max_streams)
        self._metrics = metrics
        self._runtime = BoundedVADRuntime(
            provider="silero",
            workers=workers,
            pending_capacity=pending_capacity,
            deadline_seconds=deadline_seconds,
            metrics=metrics,
        )
        self._session_factory = session_factory or _create_onnx_session
        self._session: _InferenceSession | None = None
        self._started = False
        self._closed = False

    @property
    def name(self) -> str:
        return "silero"

    @property
    def default_threshold(self) -> float:
        return 0.5

    @property
    def active_streams(self) -> int:
        return self._capacity.active

    async def startup(self) -> None:
        if self._closed:
            raise VADClosedError("Silero VAD provider is closed")
        if self._started:
            return

        _verify_artifact(self._model_path, self._model_sha256)
        try:
            session = self._session_factory(self._model_path)
        except VADConfigurationError:
            raise
        except Exception as exc:
            raise VADConfigurationError("Silero VAD model could not be loaded") from exc
        try:
            _validate_session(session)
        except VADConfigurationError:
            raise
        except Exception as exc:
            raise VADConfigurationError(
                "Silero VAD tensor contract could not be inspected"
            ) from exc

        await self._runtime.start()
        self._session = session
        self._started = True

    def new_stream(self, input_sample_rate: VADSampleRate) -> SileroVADStream:
        if self._closed or not self._started or self._session is None:
            raise VADClosedError("Silero VAD provider is not running")
        if isinstance(input_sample_rate, bool) or input_sample_rate not in (16_000, 24_000):
            raise ValueError("Silero VAD input sample rate must be 16000 or 24000 Hz")

        self._capacity.acquire()
        try:
            stream = SileroVADStream(
                input_sample_rate=input_sample_rate,
                session=self._session,
                runtime=self._runtime,
                release_lease=self._release_stream,
            )
            if self._metrics is not None:
                self._metrics.vad_stream_started("silero")
            return stream
        except Exception:
            self._capacity.release()
            raise

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._started = False
        await self._runtime.close()
        self._session = None

    def _release_stream(self) -> None:
        self._capacity.release()
        if self._metrics is not None:
            self._metrics.vad_stream_ended("silero")


class SileroVADStream:
    """Connection-owned recurrent, resampling, framing, and score state."""

    __slots__ = (
        "_closed",
        "_context",
        "_epoch",
        "_failed",
        "_expected_frame_bytes",
        "_fifo",
        "_fifo_samples",
        "_held_score",
        "_input_sample_rate",
        "_release_lease",
        "_runtime",
        "_scoring",
        "_session",
        "_state",
    )

    def __init__(
        self,
        *,
        input_sample_rate: VADSampleRate,
        session: _InferenceSession,
        runtime: BoundedVADRuntime,
        release_lease: Callable[[], None],
    ) -> None:
        self._input_sample_rate = input_sample_rate
        self._expected_frame_bytes = expected_frame_bytes(input_sample_rate)
        self._session = session
        self._runtime = runtime
        self._release_lease = release_lease
        self._closed = False
        self._scoring = False
        self._failed = False
        self._epoch = 0
        self._fifo: deque[_FloatArray] = deque()
        self._fifo_samples = 0
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)
        self._held_score = _INITIAL_SCORE

    async def score(self, pcm16_20ms: bytes) -> float:
        samples = self._decode_frame(pcm16_20ms)
        if self._closed:
            raise VADClosedError("Silero VAD stream is closed")
        if self._failed:
            raise VADInferenceError("Silero VAD stream requires reset after an error")
        if self._scoring:
            raise VADInferenceError("Silero VAD stream score calls must be sequential")

        self._scoring = True
        epoch = self._epoch
        try:
            model_samples = self._resample(samples)
            if model_samples.size:
                self._fifo.append(model_samples)
                self._fifo_samples += int(model_samples.size)

            while self._fifo_samples >= _WINDOW_SAMPLES:
                window = self._peek_window()
                model_input = np.empty((1, _CONTEXT_SAMPLES + _WINDOW_SAMPLES), dtype=np.float32)
                model_input[:, :_CONTEXT_SAMPLES] = self._context
                model_input[:, _CONTEXT_SAMPLES:] = window
                state_input = self._state.copy()

                score, next_state, next_context = await self._runtime.submit(
                    partial(_run_inference, self._session, model_input, state_input)
                )
                if self._epoch != epoch:
                    if self._closed:
                        raise VADClosedError("Silero VAD stream closed during classification")
                    raise VADInferenceError("Silero VAD stream reset during classification")

                self._discard_window()
                self._state = next_state
                self._context = next_context
                self._held_score = score
            return self._held_score
        except asyncio.CancelledError:
            self._failed = True
            raise
        except (VADCapacityError, VADClosedError, VADInferenceError):
            self._failed = True
            raise
        except Exception as exc:
            self._failed = True
            raise VADInferenceError("Silero VAD stream processing failed") from exc
        finally:
            self._scoring = False

    def reset(self) -> None:
        if self._closed:
            return
        self._epoch += 1
        self._failed = False
        self._fifo.clear()
        self._fifo_samples = 0
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)
        self._held_score = _INITIAL_SCORE

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._epoch += 1
        self._fifo.clear()
        self._fifo_samples = 0
        self._release_lease()

    def _decode_frame(self, pcm16_20ms: bytes) -> _FloatArray:
        if not isinstance(pcm16_20ms, bytes):
            raise ValueError("Silero VAD frame must be immutable bytes")
        if len(pcm16_20ms) != self._expected_frame_bytes:
            raise ValueError(
                f"Silero VAD requires exactly {self._expected_frame_bytes} PCM16LE bytes"
            )
        integers = np.frombuffer(pcm16_20ms, dtype="<i2")
        return np.multiply(integers, _PCM16_SCALE, dtype=np.float32)

    def _resample(self, samples: _FloatArray) -> _FloatArray:
        if self._input_sample_rate == _MODEL_SAMPLE_RATE:
            return samples
        # Each 20 ms 24 kHz frame has 160 complete groups of three samples.
        # Area-weighted 3:2 conversion emits two samples per group immediately,
        # so every real input sample contributes and no filter tail is lost on reset.
        groups = samples.reshape(-1, 3)
        output = np.empty(groups.shape[0] * 2, dtype=np.float32)
        output[0::2] = (groups[:, 0] * 2.0 + groups[:, 1]) / 3.0
        output[1::2] = (groups[:, 1] + groups[:, 2] * 2.0) / 3.0
        return output

    def _peek_window(self) -> _FloatArray:
        window = np.empty(_WINDOW_SAMPLES, dtype=np.float32)
        written = 0
        for chunk in self._fifo:
            count = min(chunk.size, _WINDOW_SAMPLES - written)
            window[written : written + count] = chunk[:count]
            written += count
            if written == _WINDOW_SAMPLES:
                break
        if written != _WINDOW_SAMPLES:
            raise VADInferenceError("Silero VAD FIFO accounting is inconsistent")
        return window

    def _discard_window(self) -> None:
        remaining = _WINDOW_SAMPLES
        while remaining:
            chunk = self._fifo[0]
            if chunk.size <= remaining:
                remaining -= int(chunk.size)
                self._fifo.popleft()
            else:
                self._fifo[0] = chunk[remaining:]
                remaining = 0
        self._fifo_samples -= _WINDOW_SAMPLES


def _run_inference(
    session: _InferenceSession, model_input: _FloatArray, state_input: _FloatArray
) -> tuple[float, _FloatArray, _FloatArray]:
    outputs = session.run(
        ["output", "stateN"],
        {"input": model_input, "state": state_input, "sr": _SAMPLE_RATE_TENSOR},
    )
    if len(outputs) != 2:
        raise ValueError("Silero VAD inference returned an unexpected output count")

    probability = np.asarray(outputs[0])
    next_state_raw = np.asarray(outputs[1])
    if probability.shape != (1, 1) or next_state_raw.shape != _STATE_SHAPE:
        raise ValueError("Silero VAD inference returned malformed output shapes")
    if not np.issubdtype(probability.dtype, np.floating) or not np.issubdtype(
        next_state_raw.dtype, np.floating
    ):
        raise ValueError("Silero VAD inference returned non-floating tensors")

    score = float(probability[0, 0])
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("Silero VAD inference returned an invalid probability")
    if not np.all(np.isfinite(next_state_raw)):
        raise ValueError("Silero VAD inference returned non-finite recurrent state")

    next_state = np.asarray(next_state_raw, dtype=np.float32).copy(order="C")
    next_context = model_input[:, -_CONTEXT_SAMPLES:].copy(order="C")
    return score, next_state, next_context
