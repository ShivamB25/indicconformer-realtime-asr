"""Small deterministic mono resampler used at ingestion boundaries."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.audio.pcm import SAMPLE_RATE


class ResampleError(ValueError):
    """Raised for unsupported audio shape, dtype, or sample rate."""


def _as_float32_mono(audio: np.ndarray) -> NDArray[np.float32]:
    if audio.ndim != 1:
        raise ResampleError("audio must be a one-dimensional mono array")
    if not np.issubdtype(audio.dtype, np.number):
        raise ResampleError("audio must have a numeric dtype")
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), info.max))
        result = audio.astype(np.float32) / np.float32(scale)
    else:
        result = audio.astype(np.float32, copy=True)
    if not np.all(np.isfinite(result)):
        raise ResampleError("audio contains non-finite samples")
    return result


def resample_audio(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int = SAMPLE_RATE,
) -> NDArray[np.float32]:
    """Convert mono audio to float32 and linearly resample it.

    Linear interpolation is intentional: it is dependency-free and exactly
    deterministic across the REST and streaming ingestion paths. Production
    PCM websocket traffic already arrives at 16 kHz and therefore does not
    pass through interpolation.
    """

    if source_rate <= 0 or target_rate <= 0:
        raise ResampleError("sample rates must be positive")
    samples = _as_float32_mono(audio)
    if samples.size == 0 or source_rate == target_rate:
        return np.ascontiguousarray(samples)

    output_size = max(1, round(samples.size * target_rate / source_rate))
    source_positions = (
        np.arange(output_size, dtype=np.float64) * float(source_rate) / float(target_rate)
    )
    np.minimum(source_positions, samples.size - 1, out=source_positions)
    left = source_positions.astype(np.int64)
    right = np.minimum(left + 1, samples.size - 1)
    weight = (source_positions - left).astype(np.float32)
    output = samples[left] + (samples[right] - samples[left]) * weight
    return np.ascontiguousarray(output, dtype=np.float32)
