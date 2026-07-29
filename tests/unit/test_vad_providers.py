from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import ClassVar

import numpy as np
import pytest
import webrtcvad  # type: ignore[import-untyped]

from app.vad.base import (
    VADCapacityError,
    VADClosedError,
    VADConfigurationError,
    VADInferenceError,
    VADProvider,
)
from app.vad.energy import EnergyVADProvider
from app.vad.webrtc import WebRTCVADProvider


def energy_provider(*, max_streams: int = 4, threshold: float = 0.015) -> EnergyVADProvider:
    return EnergyVADProvider(
        max_streams=max_streams,
        workers=1,
        pending_capacity=4,
        deadline_seconds=1.0,
        threshold=threshold,
    )


def webrtc_provider(*, max_streams: int = 4, mode: int = 1) -> WebRTCVADProvider:
    return WebRTCVADProvider(
        max_streams=max_streams,
        workers=1,
        pending_capacity=4,
        deadline_seconds=1.0,
        mode=mode,
    )


def pcm_frame(sample_rate: int, value: int = 0) -> bytes:
    return np.full(sample_rate // 50, value, dtype="<i2").tobytes()


class RecordingVad:
    instances: ClassVar[list[RecordingVad]] = []

    def __init__(self, mode: int | None = None) -> None:
        self.mode = mode
        self.calls: list[tuple[bytes, int]] = []
        type(self).instances.append(self)

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        self.calls.append((frame, sample_rate))
        return bool(np.max(np.abs(np.frombuffer(frame, dtype="<i2"))) > 100)


@pytest.mark.parametrize("sample_rate", [16_000, 24_000])
async def test_energy_scores_normalized_rms_at_both_rates(sample_rate: int) -> None:
    provider = energy_provider()
    await provider.startup()
    stream = provider.new_stream(sample_rate)  # type: ignore[arg-type]

    assert await stream.score(pcm_frame(sample_rate, 16_384)) == pytest.approx(0.5)
    assert await stream.score(pcm_frame(sample_rate, -32_768)) == pytest.approx(1.0)
    assert await stream.score(pcm_frame(sample_rate)) == 0.0

    stream.close()
    await provider.close()


async def test_energy_threshold_is_provider_default_without_quantizing_scores() -> None:
    provider = energy_provider(threshold=0.3)
    await provider.startup()
    stream = provider.new_stream(16_000)

    assert provider.default_threshold == pytest.approx(0.3)
    assert await stream.score(pcm_frame(16_000, 8_192)) == pytest.approx(0.25)

    stream.close()
    await provider.close()


@pytest.mark.parametrize("bad_threshold", [0.0, -0.1, 1.1, float("inf"), float("nan"), True])
def test_energy_rejects_invalid_thresholds(bad_threshold: float) -> None:
    with pytest.raises(ValueError):
        energy_provider(threshold=bad_threshold)


@pytest.mark.parametrize("sample_rate", [16_000, 24_000])
async def test_webrtc_scores_binary_decisions_at_both_rates(
    monkeypatch: pytest.MonkeyPatch, sample_rate: int
) -> None:
    RecordingVad.instances = []
    monkeypatch.setattr(webrtcvad, "Vad", RecordingVad)
    provider = webrtc_provider()
    await provider.startup()
    stream = provider.new_stream(sample_rate)  # type: ignore[arg-type]

    assert provider.default_threshold == 0.5
    assert await stream.score(pcm_frame(sample_rate)) == 0.0
    assert await stream.score(pcm_frame(sample_rate, 12_000)) == 1.0
    classifier = RecordingVad.instances[-1]
    assert [rate for _, rate in classifier.calls] == [16_000, 16_000]
    assert [len(frame) for frame, _ in classifier.calls] == [640, 640]

    stream.close()
    await provider.close()


async def test_webrtc_24khz_resampling_keeps_digital_silence_silent() -> None:
    provider = webrtc_provider()
    await provider.startup()
    stream = provider.new_stream(24_000)

    scores = [await stream.score(pcm_frame(24_000)) for _ in range(10)]

    assert scores == [0.0] * 10
    stream.close()
    await provider.close()


async def test_webrtc_forwards_mode_to_startup_stream_and_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    RecordingVad.instances = []
    monkeypatch.setattr(webrtcvad, "Vad", RecordingVad)
    provider = webrtc_provider(mode=3)
    await provider.startup()
    stream = provider.new_stream(16_000)
    stream.reset()

    assert [instance.mode for instance in RecordingVad.instances] == [3, 3, 3]

    stream.close()
    await provider.close()


async def test_webrtc_sessions_have_isolated_handles_and_reset_only_its_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    RecordingVad.instances = []
    monkeypatch.setattr(webrtcvad, "Vad", RecordingVad)
    provider = webrtc_provider()
    await provider.startup()
    first = provider.new_stream(16_000)
    second = provider.new_stream(16_000)
    first_handle, second_handle = RecordingVad.instances[1:3]

    await first.score(pcm_frame(16_000, 1_000))
    await second.score(pcm_frame(16_000))
    first.reset()

    assert first_handle is not second_handle
    assert len(first_handle.calls) == 1
    assert len(second_handle.calls) == 1
    assert RecordingVad.instances[-1] not in (first_handle, second_handle)
    assert len(RecordingVad.instances) == 4

    first.close()
    second.close()
    await provider.close()


ProviderFactory = Callable[[int], VADProvider]


def make_energy_provider(max_streams: int) -> VADProvider:
    return energy_provider(max_streams=max_streams)


def make_webrtc_provider(max_streams: int) -> VADProvider:
    return webrtc_provider(max_streams=max_streams)


@pytest.mark.parametrize("factory", [make_energy_provider, make_webrtc_provider])
async def test_stream_capacity_is_leased_and_released(factory: ProviderFactory) -> None:
    provider = factory(1)
    await provider.startup()
    first = provider.new_stream(16_000)

    assert provider.active_streams == 1
    with pytest.raises(VADCapacityError):
        provider.new_stream(16_000)

    first.close()
    replacement = provider.new_stream(24_000)
    assert provider.active_streams == 1
    replacement.close()
    assert provider.active_streams == 0
    await provider.close()


@pytest.mark.parametrize("factory", [make_energy_provider, make_webrtc_provider])
@pytest.mark.parametrize("sample_rate", [16_000, 24_000])
async def test_malformed_frames_are_rejected_before_stream_state_changes(
    factory: ProviderFactory, sample_rate: int
) -> None:
    provider = factory(1)
    await provider.startup()
    stream = provider.new_stream(sample_rate)  # type: ignore[arg-type]
    expected_bytes = sample_rate * 2 // 50

    for malformed in (
        b"",
        b"\x00" * (expected_bytes - 1),
        b"\x00" * (expected_bytes + 1),
        bytearray(expected_bytes),
    ):
        with pytest.raises(ValueError):
            await stream.score(malformed)  # type: ignore[arg-type]
    assert 0.0 <= await stream.score(bytes(expected_bytes)) <= 1.0

    stream.close()
    await provider.close()


@pytest.mark.parametrize("factory", [make_energy_provider, make_webrtc_provider])
async def test_stream_and_provider_close_are_idempotent(factory: ProviderFactory) -> None:
    provider = factory(1)
    await provider.startup()
    stream = provider.new_stream(16_000)

    stream.close()
    stream.close()
    assert provider.active_streams == 0
    with pytest.raises(VADClosedError):
        await stream.score(pcm_frame(16_000))

    await provider.close()
    await provider.close()
    with pytest.raises(VADClosedError):
        provider.new_stream(16_000)


async def test_webrtc_classifier_failure_is_reported_without_energy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingVad(RecordingVad):
        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            self.calls.append((frame, sample_rate))
            raise RuntimeError("classifier failed")

    FailingVad.instances = []
    monkeypatch.setattr(webrtcvad, "Vad", FailingVad)
    provider = webrtc_provider()
    await provider.startup()
    stream = provider.new_stream(16_000)
    classifier = FailingVad.instances[-1]

    with pytest.raises(VADInferenceError, match="classifier"):
        await stream.score(pcm_frame(16_000, 10_000))
    with pytest.raises(VADInferenceError, match="reset"):
        await stream.score(pcm_frame(16_000, 10_000))
    assert len(classifier.calls) == 1

    stream.close()
    await provider.close()


async def test_webrtc_startup_failure_is_not_replaced_by_another_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_initialize(mode: int) -> RecordingVad:
        raise OSError(f"mode {mode} unavailable")

    monkeypatch.setattr(webrtcvad, "Vad", fail_to_initialize)
    provider = webrtc_provider(mode=2)

    with pytest.raises(VADConfigurationError, match="could not be initialized"):
        await provider.startup()
    with pytest.raises(VADClosedError):
        provider.new_stream(16_000)

    await provider.close()


async def test_overlapping_webrtc_scores_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingVad(RecordingVad):
        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            self.calls.append((frame, sample_rate))
            started.set()
            if not release.wait(timeout=1.0):
                raise RuntimeError("test did not release classifier")
            return True

    BlockingVad.instances = []
    monkeypatch.setattr(webrtcvad, "Vad", BlockingVad)
    provider = webrtc_provider()
    await provider.startup()
    stream = provider.new_stream(16_000)
    first = asyncio.create_task(stream.score(pcm_frame(16_000, 1_000)))
    assert await asyncio.to_thread(started.wait, 1.0)

    with pytest.raises(VADInferenceError, match="sequential"):
        await stream.score(pcm_frame(16_000, 1_000))
    release.set()
    assert await first == 1.0

    stream.close()
    await provider.close()
