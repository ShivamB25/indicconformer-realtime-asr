"""Bounded decoding for containerized OpenAI transcription uploads."""

from __future__ import annotations

import io
import wave

import av
import numpy as np
from numpy.typing import NDArray

TARGET_SAMPLE_RATE = 16_000
_AV_TIME_BASE = 1_000_000
# The outer upload uses caller-owned BytesIO. No FFmpeg URL protocol is needed,
# so every nested protocol open is denied.
_DEMUX_OPTIONS = {
    "protocol_whitelist": "",
    "max_streams": "8",
    "probesize": str(1024 * 1024),
    "analyzeduration": str(5 * _AV_TIME_BASE),
}


class InvalidAudioError(ValueError):
    """The upload is not a decodable, non-empty supported audio container."""


class AudioDurationExceeded(ValueError):
    """Decoded audio would exceed the configured duration bound."""


def _demuxer(payload: bytes) -> str:
    """Select only the five documented container families from their signatures."""

    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WAVE":
        return "wav"
    if payload.startswith(b"fLaC"):
        return "flac"
    if payload.startswith(b"OggS"):
        return "ogg"
    if len(payload) >= 12 and payload[4:8] == b"ftyp":
        return "mov"
    if payload.startswith(b"ID3") or (
        len(payload) >= 2
        and payload[0] == 0xFF
        and payload[1] & 0xE0 == 0xE0
        and payload[1] & 0x18 != 0x08
        and payload[1] & 0x06 != 0
    ):
        return "mp3"
    raise InvalidAudioError(
        "The uploaded audio file is not a supported WAV, MP3, FLAC, M4A, or OGG file"
    )


def _reject_oversize_pcm_wav(payload: bytes, max_audio_seconds: int) -> None:
    """Reject declared PCM WAV duration without allocating decoded float samples."""

    try:
        with wave.open(io.BytesIO(payload), "rb") as source:
            if source.getcomptype() != "NONE" or source.getframerate() <= 0:
                return
            if source.getnframes() > max_audio_seconds * source.getframerate():
                raise AudioDurationExceeded(
                    "The audio file exceeds the configured maximum duration"
                )
    except AudioDurationExceeded:
        raise
    except (EOFError, wave.Error):
        # FFmpeg supplies the stable public error for malformed containers.
        return


def decode_audio_file(payload: bytes, *, max_audio_seconds: int) -> NDArray[np.float32]:
    """Decode, downmix, and resample one supported upload to bounded mono float32."""

    if not payload:
        raise InvalidAudioError("The uploaded audio file is empty")
    if max_audio_seconds <= 0:
        raise ValueError("max_audio_seconds must be positive")

    demuxer = _demuxer(payload)
    if demuxer == "wav":
        _reject_oversize_pcm_wav(payload, max_audio_seconds)

    max_samples = max_audio_seconds * TARGET_SAMPLE_RATE
    chunks: list[NDArray[np.float32]] = []
    sample_count = 0
    frame_count = 0

    def append_frames(frames: list[av.AudioFrame]) -> None:
        nonlocal frame_count, sample_count
        for frame in frames:
            frame_count += 1
            # A non-empty audio frame contributes at least one sample. This also
            # bounds adversarial streams made from enormous numbers of tiny frames.
            if frame_count > max_samples or frame.samples > max_samples - sample_count:
                raise AudioDurationExceeded(
                    "The audio file exceeds the configured maximum duration"
                )
            if not frame.samples:
                continue
            values = frame.to_ndarray().reshape(-1)
            if values.size > max_samples - sample_count:
                raise AudioDurationExceeded(
                    "The audio file exceeds the configured maximum duration"
                )
            chunk = np.ascontiguousarray(values, dtype=np.float32)
            chunks.append(chunk)
            sample_count += chunk.size

    try:
        with av.open(
            io.BytesIO(payload),
            mode="r",
            format=demuxer,
            options=_DEMUX_OPTIONS,
        ) as container:
            if (
                container.duration is not None
                and container.duration > max_audio_seconds * _AV_TIME_BASE
            ):
                raise AudioDurationExceeded(
                    "The audio file exceeds the configured maximum duration"
                )
            if not container.streams.audio:
                raise InvalidAudioError("The uploaded file does not contain an audio stream")
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=TARGET_SAMPLE_RATE)
            for frame in container.decode(stream):
                append_frames(resampler.resample(frame))
            append_frames(resampler.resample(None))
    except (InvalidAudioError, AudioDurationExceeded):
        raise
    except Exception as exc:
        raise InvalidAudioError("The uploaded audio file could not be decoded") from exc

    if not chunks:
        raise InvalidAudioError("The uploaded audio file contains no audio samples")
    waveform = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    if not np.isfinite(waveform).all():
        raise InvalidAudioError("The uploaded audio file contains invalid samples")
    return waveform
