#!/usr/bin/env python3
"""Deterministic local benchmark for the production Silero streaming pipeline."""

from __future__ import annotations

import asyncio
import math
import statistics
import time
from pathlib import Path

import numpy as np

from app.vad.base import VADSampleRate
from app.vad.silero import SileroVADProvider

_MODEL_PATH = Path(".models/vad/silero-v6.2.1/silero_vad.onnx")
_FRAME_DURATION_MS = 20
_MEASURED_FRAMES = 4_000
_REPETITIONS = 5
_WARMUP_FRAMES = 80
_SAMPLE_RATES: tuple[VADSampleRate, ...] = (16_000, 24_000)


def _pcm_frames(sample_rate: VADSampleRate, frame_count: int) -> tuple[bytes, ...]:
    samples_per_frame = sample_rate * _FRAME_DURATION_MS // 1_000
    sample_count = samples_per_frame * frame_count
    positions = np.arange(sample_count, dtype=np.float32)
    phase = positions * np.float32(2.0 * math.pi / sample_rate)
    waveform = np.sin(phase * np.float32(220.0)) * np.float32(0.18) + np.sin(
        phase * np.float32(711.0)
    ) * np.float32(0.035)

    frame_envelope = np.ones(frame_count, dtype=np.float32)
    for silent_start in range(0, frame_count, 150):
        frame_envelope[silent_start : silent_start + 50] = 0.0
    waveform *= np.repeat(frame_envelope, samples_per_frame)
    pcm = np.clip(np.rint(waveform * np.float32(32_767.0)), -32_768, 32_767).astype("<i2")
    return tuple(
        pcm[offset : offset + samples_per_frame].tobytes()
        for offset in range(0, sample_count, samples_per_frame)
    )


async def _run_stream(
    provider: SileroVADProvider, sample_rate: VADSampleRate, frames: tuple[bytes, ...]
) -> tuple[float, tuple[float, ...]]:
    stream = provider.new_stream(sample_rate)
    scores: list[float] = []
    started = time.perf_counter_ns()
    try:
        for frame in frames:
            scores.append(await stream.score(frame))
    finally:
        stream.close()
    elapsed_ns = time.perf_counter_ns() - started
    return elapsed_ns / len(frames) / 1_000.0, tuple(scores)


async def _benchmark() -> None:
    provider = SileroVADProvider(
        model_path=_MODEL_PATH,
        max_streams=1,
        workers=1,
        pending_capacity=2,
        deadline_seconds=5.0,
    )
    await provider.startup()
    try:
        workloads: dict[VADSampleRate, tuple[bytes, ...]] = {
            sample_rate: _pcm_frames(sample_rate, _MEASURED_FRAMES) for sample_rate in _SAMPLE_RATES
        }
        for sample_rate in workloads:
            _ = await _run_stream(
                provider,
                sample_rate,
                workloads[sample_rate][:_WARMUP_FRAMES],
            )

        timings: dict[VADSampleRate, list[float]] = {16_000: [], 24_000: []}
        reference_scores: dict[VADSampleRate, tuple[float, ...]] = {}
        for _ in range(_REPETITIONS):
            for sample_rate, frames in workloads.items():
                microseconds, scores = await _run_stream(provider, sample_rate, frames)
                timings[sample_rate].append(microseconds)
                previous = reference_scores.setdefault(sample_rate, scores)
                if scores != previous:
                    raise RuntimeError(
                        f"Silero scores changed between {sample_rate} Hz repetitions"
                    )
    finally:
        await provider.close()

    rate_metrics = {
        sample_rate: statistics.median(values) for sample_rate, values in timings.items()
    }
    primary = statistics.fmean(rate_metrics.values())
    score_sum = sum(sum(scores) for scores in reference_scores.values())
    print(f"METRIC silero_pipeline_us_per_frame={primary:.3f}")
    print(f"METRIC silero_16k_us_per_frame={rate_metrics[16_000]:.3f}")
    print(f"METRIC silero_24k_us_per_frame={rate_metrics[24_000]:.3f}")
    print(f"METRIC silero_score_sum={score_sum:.9f}")


def main() -> int:
    asyncio.run(_benchmark())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
