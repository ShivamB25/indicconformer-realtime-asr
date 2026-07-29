"""Deterministic in-memory WAV containers for the REST upload path.

Only the REST endpoint accepts a container; the realtime socket takes raw PCM.
Everything here is synthetic silence or a constant tone: no recorded audio and no
claim about what a real model would transcribe from it.
"""

from __future__ import annotations

import io
import wave

from app.audio.pcm import BYTES_PER_SAMPLE, SAMPLE_RATE
from tests.support.audio import PCM16_DTYPE, SPEECH_AMPLITUDE, int16_frame

__all__ = [
    "PCM_WAV_HEADER",
    "pcm_bytes_for_ms",
    "wav_bytes",
]

PCM_WAV_HEADER = b"RIFF"


def pcm_bytes_for_ms(duration_ms: int, amplitude: int = SPEECH_AMPLITUDE) -> bytes:
    """Headerless little-endian PCM16 covering ``duration_ms`` of 20 ms frames."""

    if duration_ms % 20:
        raise ValueError("duration_ms must be a multiple of 20 ms")
    frame = int16_frame(amplitude).astype(PCM16_DTYPE, copy=False).tobytes()
    return frame * (duration_ms // 20)


def wav_bytes(
    pcm: bytes,
    *,
    channels: int = 1,
    sample_rate: int = SAMPLE_RATE,
    sample_width: int = BYTES_PER_SAMPLE,
) -> bytes:
    """Wrap PCM payload bytes in a RIFF/WAVE container, valid or deliberately not.

    The channel count, rate, and width are caller-controlled so tests can present
    containers the service must refuse.
    """

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as sink:
        sink.setnchannels(channels)
        sink.setsampwidth(sample_width)
        sink.setframerate(sample_rate)
        sink.writeframes(pcm)
    return buffer.getvalue()
