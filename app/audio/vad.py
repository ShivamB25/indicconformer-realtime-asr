"""Deterministic frame-level energy voice activity detection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.audio.pcm import SAMPLES_PER_FRAME


@dataclass(frozen=True, slots=True)
class EnergyVADConfig:
    """Configuration for normalized RMS energy classification."""

    speech_threshold: float = 0.015

    def __post_init__(self) -> None:
        if not 0.0 < self.speech_threshold <= 1.0:
            raise ValueError("speech_threshold must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class VADDecision:
    is_speech: bool
    rms: float
    dbfs: float


class EnergyVAD:
    """Classify each complete 20 ms frame solely by normalized RMS energy."""

    __slots__ = ("config",)

    def __init__(self, config: EnergyVADConfig | None = None) -> None:
        self.config = config or EnergyVADConfig()

    def classify(self, frame: np.ndarray) -> VADDecision:
        if frame.ndim != 1 or frame.size != SAMPLES_PER_FRAME:
            raise ValueError(f"VAD requires exactly {SAMPLES_PER_FRAME} mono samples")
        if np.issubdtype(frame.dtype, np.integer):
            info = np.iinfo(frame.dtype)
            peak = float(max(abs(info.min), info.max))
            normalized = frame.astype(np.float64) / peak
        elif np.issubdtype(frame.dtype, np.floating):
            normalized = frame.astype(np.float64, copy=False)
        else:
            raise ValueError("VAD frame must have a numeric dtype")
        if not np.all(np.isfinite(normalized)):
            raise ValueError("VAD frame contains non-finite samples")

        rms = float(np.sqrt(np.mean(np.square(normalized), dtype=np.float64)))
        dbfs = 20.0 * math.log10(max(rms, 1e-12))
        return VADDecision(
            is_speech=rms >= self.config.speech_threshold
            or math.isclose(rms, self.config.speech_threshold, rel_tol=1e-7),
            rms=rms,
            dbfs=dbfs,
        )

    def is_speech(self, frame: np.ndarray) -> bool:
        return self.classify(frame).is_speech
