"""Configuration-driven construction of voice activity detection providers."""

from __future__ import annotations

from app.core.config import Settings
from app.core.types import VADKind
from app.vad.base import VADConfigurationError, VADProvider
from app.vad.runtime import VADRuntimeMetrics


def build_vad_provider(
    settings: Settings,
    metrics: VADRuntimeMetrics | None,
) -> VADProvider:
    """Construct exactly the configured provider without loading model artifacts."""

    if settings.vad_provider is VADKind.DISABLED:
        from app.vad.disabled import DisabledVADProvider

        return DisabledVADProvider(
            max_streams=settings.vad_max_streams,
            metrics=metrics,
        )

    if settings.vad_provider is VADKind.ENERGY:
        from app.vad.energy import EnergyVADProvider

        return EnergyVADProvider(
            max_streams=settings.vad_max_streams,
            workers=settings.vad_cpu_workers,
            pending_capacity=settings.vad_pending_capacity,
            deadline_seconds=settings.vad_classification_deadline_seconds,
            metrics=metrics,
        )

    if settings.vad_provider is VADKind.WEBRTC:
        from app.vad.webrtc import WebRTCVADProvider

        return WebRTCVADProvider(
            max_streams=settings.vad_max_streams,
            workers=settings.vad_cpu_workers,
            pending_capacity=settings.vad_pending_capacity,
            deadline_seconds=settings.vad_classification_deadline_seconds,
            metrics=metrics,
            mode=settings.vad_webrtc_mode,
        )

    if settings.vad_provider is VADKind.SILERO:
        if settings.vad_model_path is None or settings.vad_model_sha256 is None:
            raise VADConfigurationError(
                "Silero VAD requires a model path and pinned SHA-256 digest"
            )
        from app.vad.silero import SileroVADProvider

        return SileroVADProvider(
            model_path=settings.vad_model_path,
            model_sha256=settings.vad_model_sha256,
            max_streams=settings.vad_max_streams,
            workers=settings.vad_cpu_workers,
            pending_capacity=settings.vad_pending_capacity,
            deadline_seconds=settings.vad_classification_deadline_seconds,
            metrics=metrics,
        )

    raise VADConfigurationError(f"unsupported VAD provider: {settings.vad_provider}")
