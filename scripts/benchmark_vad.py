#!/usr/bin/env python3
"""Benchmark labeled 20 ms streaming VAD without exposing audio or transcripts.

The input contract is ``scripts/vad_benchmark_manifest.schema.json``. Reported
accuracy numbers describe only the supplied labeled corpus; this tool does not
make a general model-accuracy claim.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import importlib.metadata
import io
import json
import math
import os
import pathlib
import platform
import sys
import time
import wave
from array import array
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from app.vad.base import VADProvider, VADStream, expected_frame_bytes
else:
    from app.vad.base import VADProvider, VADStream, expected_frame_bytes

SCHEMA_VERSION: Final[str] = "1.0"
MANIFEST_SCHEMA_VERSION: Final[str] = "1.0"
FRAME_DURATION_MS: Final[int] = 20
SILERO_VERSION: Final[str] = "6.2.1"
SILERO_COMMIT: Final[str] = "7e30209a3e901f9842f81b225f3e93d8199902b1"
SILERO_ARTIFACT: Final[str] = "src/silero_vad/data/silero_vad.onnx"
SILERO_SHA256: Final[str] = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
LANGUAGES: Final[frozenset[str]] = frozenset(
    {
        "as",
        "bn",
        "brx",
        "doi",
        "gu",
        "hi",
        "kn",
        "kok",
        "ks",
        "mai",
        "ml",
        "mni",
        "mr",
        "ne",
        "or",
        "pa",
        "sa",
        "sat",
        "sd",
        "ta",
        "te",
        "ur",
    }
)
PROVIDER_IDS: Final[tuple[str, ...]] = (
    "energy",
    "webrtc_mode_0",
    "webrtc_mode_1",
    "webrtc_mode_2",
    "webrtc_mode_3",
    "silero",
)


class BenchmarkError(RuntimeError):
    """The benchmark input, provider, or result violated its contract."""


@dataclasses.dataclass(frozen=True, slots=True)
class Variant:
    clip_id: str
    variant_id: str
    language: str
    path: pathlib.Path
    sha256: str
    condition: dict[str, str | float]
    speech_segments: tuple[tuple[int, int], ...]

    @property
    def condition_key(self) -> str:
        if self.condition["type"] == "clean":
            return "clean"
        return f"noise:{self.condition['noise']}:{cast(float, self.condition['snr_db']):g}dB"


@dataclasses.dataclass(frozen=True, slots=True)
class Manifest:
    path: pathlib.Path
    digest: str
    corpus_name: str
    corpus_version: str
    sample_rate_hz: int
    variants: tuple[Variant, ...]


@dataclasses.dataclass(slots=True)
class Aggregate:
    true_positive_frames: int = 0
    false_positive_frames: int = 0
    true_negative_frames: int = 0
    false_negative_frames: int = 0
    false_activations: int = 0
    reference_segments: int = 0
    matched_reference_segments: int = 0
    frame_count: int = 0
    cpu_seconds: float = 0.0
    classification_latencies_ms: list[float] = dataclasses.field(default_factory=list)
    onset_delays_ms: list[float] = dataclasses.field(default_factory=list)
    endpoint_delays_ms: list[float] = dataclasses.field(default_factory=list)

    def add(
        self,
        reference: Sequence[bool],
        predicted: Sequence[bool],
        latencies_ms: Sequence[float],
        cpu_seconds: float,
    ) -> None:
        if len(reference) != len(predicted) or len(reference) != len(latencies_ms):
            raise BenchmarkError("reference, prediction, and latency lengths differ")
        counts = confusion_counts(reference, predicted)
        self.true_positive_frames += counts["true_positive_frames"]
        self.false_positive_frames += counts["false_positive_frames"]
        self.true_negative_frames += counts["true_negative_frames"]
        self.false_negative_frames += counts["false_negative_frames"]
        self.false_activations += false_activation_count(reference, predicted)
        delays = segment_delays(reference, predicted)
        self.reference_segments += delays["reference_segment_count"]
        self.matched_reference_segments += delays["matched_reference_segment_count"]
        self.onset_delays_ms.extend(delays["onset_delays_ms"])
        self.endpoint_delays_ms.extend(delays["endpoint_delays_ms"])
        self.classification_latencies_ms.extend(latencies_ms)
        self.frame_count += len(reference)
        self.cpu_seconds += cpu_seconds

    def summary(self) -> dict[str, Any]:
        counts = {
            "true_positive_frames": self.true_positive_frames,
            "false_positive_frames": self.false_positive_frames,
            "true_negative_frames": self.true_negative_frames,
            "false_negative_frames": self.false_negative_frames,
        }
        rates = metric_rates(**counts)
        nonspeech_hours = (
            (self.false_positive_frames + self.true_negative_frames) * FRAME_DURATION_MS / 3_600_000
        )
        audio_seconds = self.frame_count * FRAME_DURATION_MS / 1_000
        return {
            "frames": self.frame_count,
            "audio_seconds": audio_seconds,
            "confusion": counts,
            **rates,
            "false_activation_segments": self.false_activations,
            "false_activations_per_hour": safe_ratio(
                float(self.false_activations), nonspeech_hours
            ),
            "reference_segments": self.reference_segments,
            "matched_reference_segments": self.matched_reference_segments,
            "onset_delay_ms": percentile_summary(self.onset_delays_ms),
            "endpoint_delay_ms": percentile_summary(self.endpoint_delays_ms),
            "cpu_rtf": safe_ratio(self.cpu_seconds, audio_seconds),
            "classification_latency_ms": {
                "p50": percentile(self.classification_latencies_ms, 50),
                "p95": percentile(self.classification_latencies_ms, 95),
            },
        }


class NullMetrics:
    """Structurally satisfies the public VAD metrics contract without side effects."""

    def set_vad_queue_depth(self, depth: int) -> None:
        del depth

    def record_vad_queue_wait(self, provider: str, seconds: float) -> None:
        del provider, seconds

    def record_vad_inference(self, provider: str, seconds: float) -> None:
        del provider, seconds

    def record_vad_runtime_error(self, provider: str, code: str) -> None:
        del provider, code

    def vad_stream_started(self, provider: str) -> None:
        del provider

    def vad_stream_ended(self, provider: str) -> None:
        del provider


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """Return a finite ratio, or JSON ``null`` when its denominator is zero."""

    if denominator == 0:
        return None
    result = numerator / denominator
    if not math.isfinite(result):
        raise BenchmarkError("metric calculation produced a non-finite value")
    return result


def percentile(values: Sequence[float], requested: float) -> float | None:
    """Return the R-7 linearly interpolated percentile, or null for no observations."""

    if not 0 <= requested <= 100:
        raise ValueError("percentile must be in [0, 100]")
    if not values:
        return None
    if any(not math.isfinite(value) for value in values):
        raise BenchmarkError("percentile input contains a non-finite value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * requested / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def percentile_summary(values: Sequence[float]) -> dict[str, float | None]:
    return {"p50": percentile(values, 50), "p95": percentile(values, 95)}


def true_runs(flags: Sequence[bool]) -> list[tuple[int, int]]:
    """Return maximal half-open ``[start_frame, end_frame)`` true runs."""

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def confusion_counts(reference: Sequence[bool], predicted: Sequence[bool]) -> dict[str, int]:
    if len(reference) != len(predicted):
        raise ValueError("reference and prediction lengths differ")
    tp = fp = tn = fn = 0
    for expected, actual in zip(reference, predicted, strict=True):
        if expected and actual:
            tp += 1
        elif expected:
            fn += 1
        elif actual:
            fp += 1
        else:
            tn += 1
    return {
        "true_positive_frames": tp,
        "false_positive_frames": fp,
        "true_negative_frames": tn,
        "false_negative_frames": fn,
    }


def metric_rates(
    *,
    true_positive_frames: int,
    false_positive_frames: int,
    true_negative_frames: int,
    false_negative_frames: int,
) -> dict[str, float | None]:
    tp = true_positive_frames
    fp = false_positive_frames
    tn = true_negative_frames
    fn = false_negative_frames
    return {
        "frame_f1": safe_ratio(2.0 * tp, 2.0 * tp + fp + fn),
        "miss_rate": safe_ratio(float(fn), tp + fn),
        "false_positive_time_rate": safe_ratio(float(fp), fp + tn),
    }


def false_activation_count(reference: Sequence[bool], predicted: Sequence[bool]) -> int:
    """Count maximal predicted-speech runs containing no reference-speech frame."""

    if len(reference) != len(predicted):
        raise ValueError("reference and prediction lengths differ")
    return sum(not any(reference[start:end]) for start, end in true_runs(predicted))


def segment_delays(reference: Sequence[bool], predicted: Sequence[bool]) -> dict[str, Any]:
    """Match each reference run to all overlapping predicted runs.

    The first overlapping prediction supplies onset and the last supplies endpoint.
    Undetected reference runs are counted but excluded from delay percentiles.
    """

    if len(reference) != len(predicted):
        raise ValueError("reference and prediction lengths differ")
    predicted_runs = true_runs(predicted)
    onset: list[float] = []
    endpoint: list[float] = []
    references = true_runs(reference)
    for reference_start, reference_end in references:
        overlaps = [
            (start, end)
            for start, end in predicted_runs
            if start < reference_end and end > reference_start
        ]
        if not overlaps:
            continue
        onset.append((overlaps[0][0] - reference_start) * FRAME_DURATION_MS)
        endpoint.append((overlaps[-1][1] - reference_end) * FRAME_DURATION_MS)
    return {
        "reference_segment_count": len(references),
        "matched_reference_segment_count": len(onset),
        "onset_delays_ms": onset,
        "endpoint_delays_ms": endpoint,
    }


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkError(f"manifest contains duplicate object key {key!r}")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> None:
    raise BenchmarkError(f"manifest contains invalid JSON number {value}")


def canonical_manifest_hash(payload: Any) -> str:
    """Hash canonical UTF-8 JSON; insignificant formatting and key order do not matter."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_keys(value: Mapping[str, Any], allowed: set[str], context: str) -> None:
    extra = set(value) - allowed
    if extra:
        raise BenchmarkError(f"{context} has unknown fields: {', '.join(sorted(extra))}")


def _required_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkError(f"{context} must be a non-empty string")
    return value


def _required_int(value: Any, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchmarkError(f"{context} must be an integer >= {minimum}")
    return cast(int, value)


def load_manifest(path: pathlib.Path) -> Manifest:
    resolved = path.expanduser().resolve(strict=True)
    try:
        payload = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_invalid_json_constant,
        )
    except BenchmarkError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError("manifest is not readable strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise BenchmarkError("manifest root must be an object")
    _exact_keys(
        payload,
        {"schema_version", "corpus", "sample_rate_hz", "frame_duration_ms", "clips"},
        "manifest",
    )
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise BenchmarkError(f"manifest schema_version must be {MANIFEST_SCHEMA_VERSION!r}")
    if payload.get("frame_duration_ms") != FRAME_DURATION_MS:
        raise BenchmarkError("manifest frame_duration_ms must be exactly 20")
    sample_rate = payload.get("sample_rate_hz")
    if isinstance(sample_rate, bool) or sample_rate not in (16_000, 24_000):
        raise BenchmarkError("manifest sample_rate_hz must be 16000 or 24000")
    corpus = payload.get("corpus")
    if not isinstance(corpus, dict):
        raise BenchmarkError("manifest corpus must be an object")
    _exact_keys(corpus, {"name", "version"}, "corpus")
    corpus_name = _required_string(corpus.get("name"), "corpus.name")
    corpus_version = _required_string(corpus.get("version"), "corpus.version")
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        raise BenchmarkError("manifest clips must be a non-empty array")

    variants: list[Variant] = []
    clip_ids: set[str] = set()
    for clip_index, raw_clip in enumerate(clips):
        context = f"clips[{clip_index}]"
        if not isinstance(raw_clip, dict):
            raise BenchmarkError(f"{context} must be an object")
        _exact_keys(raw_clip, {"id", "language", "speech_segments", "variants"}, context)
        clip_id = _required_string(raw_clip.get("id"), f"{context}.id")
        if clip_id in clip_ids:
            raise BenchmarkError(f"duplicate clip id {clip_id!r}")
        clip_ids.add(clip_id)
        language = raw_clip.get("language")
        if language not in LANGUAGES:
            raise BenchmarkError(f"{context}.language is not a supported language code")
        raw_segments = raw_clip.get("speech_segments")
        if not isinstance(raw_segments, list):
            raise BenchmarkError(f"{context}.speech_segments must be an array")
        segments: list[tuple[int, int]] = []
        previous_end = 0
        for segment_index, raw_segment in enumerate(raw_segments):
            segment_context = f"{context}.speech_segments[{segment_index}]"
            if not isinstance(raw_segment, dict):
                raise BenchmarkError(f"{segment_context} must be an object")
            _exact_keys(raw_segment, {"start_frame", "end_frame"}, segment_context)
            start = _required_int(raw_segment.get("start_frame"), f"{segment_context}.start_frame")
            end = _required_int(raw_segment.get("end_frame"), f"{segment_context}.end_frame", 1)
            if end <= start:
                raise BenchmarkError(f"{segment_context} must have end_frame > start_frame")
            if segments and start < previous_end:
                raise BenchmarkError(
                    f"{context}.speech_segments must be sorted and non-overlapping"
                )
            segments.append((start, end))
            previous_end = end
        raw_variants = raw_clip.get("variants")
        if not isinstance(raw_variants, list) or not raw_variants:
            raise BenchmarkError(f"{context}.variants must be a non-empty array")
        variant_ids: set[str] = set()
        clean_count = 0
        for variant_index, raw_variant in enumerate(raw_variants):
            variant_context = f"{context}.variants[{variant_index}]"
            if not isinstance(raw_variant, dict):
                raise BenchmarkError(f"{variant_context} must be an object")
            _exact_keys(raw_variant, {"id", "path", "sha256", "condition"}, variant_context)
            variant_id = _required_string(raw_variant.get("id"), f"{variant_context}.id")
            if variant_id in variant_ids:
                raise BenchmarkError(f"duplicate variant id {variant_id!r} in clip {clip_id!r}")
            variant_ids.add(variant_id)
            raw_path = _required_string(raw_variant.get("path"), f"{variant_context}.path")
            digest = raw_variant.get("sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise BenchmarkError(f"{variant_context}.sha256 must be lowercase SHA-256")
            condition = raw_variant.get("condition")
            if not isinstance(condition, dict):
                raise BenchmarkError(f"{variant_context}.condition must be an object")
            condition_type = condition.get("type")
            normalized_condition: dict[str, str | float]
            if condition_type == "clean":
                _exact_keys(condition, {"type"}, f"{variant_context}.condition")
                normalized_condition = {"type": "clean"}
                clean_count += 1
            elif condition_type == "noise":
                _exact_keys(
                    condition,
                    {"type", "noise", "snr_db"},
                    f"{variant_context}.condition",
                )
                noise = _required_string(
                    condition.get("noise"), f"{variant_context}.condition.noise"
                )
                snr = condition.get("snr_db")
                if (
                    isinstance(snr, bool)
                    or not isinstance(snr, (int, float))
                    or not math.isfinite(snr)
                ):
                    raise BenchmarkError(f"{variant_context}.condition.snr_db must be finite")
                normalized_condition = {"type": "noise", "noise": noise, "snr_db": float(snr)}
            else:
                raise BenchmarkError(f"{variant_context}.condition.type must be clean or noise")
            audio_path = pathlib.Path(raw_path)
            if not audio_path.is_absolute():
                audio_path = resolved.parent / audio_path
            variants.append(
                Variant(
                    clip_id=clip_id,
                    variant_id=variant_id,
                    language=cast(str, language),
                    path=audio_path,
                    sha256=digest,
                    condition=normalized_condition,
                    speech_segments=tuple(segments),
                )
            )
        if clean_count == 0:
            raise BenchmarkError(f"{context} must include at least one clean variant")
    return Manifest(
        path=resolved,
        digest=canonical_manifest_hash(payload),
        corpus_name=corpus_name,
        corpus_version=corpus_version,
        sample_rate_hz=cast(int, sample_rate),
        variants=tuple(variants),
    )


def read_frames(variant: Variant, sample_rate_hz: int) -> tuple[list[bytes], list[bool]]:
    try:
        audio_bytes = variant.path.read_bytes()
    except OSError as exc:
        raise BenchmarkError(
            f"audio artifact for {variant.clip_id}/{variant.variant_id} is unreadable"
        ) from exc
    if hashlib.sha256(audio_bytes).hexdigest() != variant.sha256:
        raise BenchmarkError(
            f"audio artifact hash mismatch for {variant.clip_id}/{variant.variant_id}"
        )
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as audio:
            if (
                audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or audio.getframerate() != sample_rate_hz
                or audio.getcomptype() != "NONE"
            ):
                raise BenchmarkError(
                    f"audio artifact {variant.clip_id}/{variant.variant_id} must be mono "
                    f"uncompressed PCM16LE at {sample_rate_hz} Hz"
                )
            sample_count = audio.getnframes()
            pcm = audio.readframes(sample_count)
            if len(pcm) != sample_count * 2:
                raise BenchmarkError(
                    f"audio artifact {variant.clip_id}/{variant.variant_id} is truncated"
                )
    except (wave.Error, EOFError) as exc:
        raise BenchmarkError(
            f"audio artifact {variant.clip_id}/{variant.variant_id} is not a PCM WAV"
        ) from exc
    frame_size = expected_frame_bytes(cast(Any, sample_rate_hz))
    if not pcm or len(pcm) % frame_size:
        raise BenchmarkError(
            f"audio artifact {variant.clip_id}/{variant.variant_id} must contain an exact "
            "positive number of 20 ms frames"
        )
    frames = [pcm[offset : offset + frame_size] for offset in range(0, len(pcm), frame_size)]
    if variant.speech_segments and variant.speech_segments[-1][1] > len(frames):
        raise BenchmarkError(
            f"speech labels exceed audio duration for {variant.clip_id}/{variant.variant_id}"
        )
    reference = [False] * len(frames)
    for start, end in variant.speech_segments:
        reference[start:end] = [True] * (end - start)
    return frames, reference


async def score_frames(
    stream: VADStream,
    frames: Sequence[bytes],
    expected_size: int,
) -> tuple[list[float], list[float], float]:
    """Submit each frame once, sequentially, and measure only classification work."""

    scores: list[float] = []
    latencies_ms: list[float] = []
    cpu_seconds = 0.0
    for frame in frames:
        if len(frame) != expected_size:
            raise BenchmarkError(
                f"benchmark frame has {len(frame)} bytes; expected {expected_size}"
            )
        wall_started = time.perf_counter_ns()
        cpu_started = time.process_time_ns()
        score = await stream.score(frame)
        cpu_seconds += (time.process_time_ns() - cpu_started) / 1_000_000_000
        latencies_ms.append((time.perf_counter_ns() - wall_started) / 1_000_000)
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise BenchmarkError("provider returned a non-finite or non-numeric score")
        if not 0 <= score <= 1:
            raise BenchmarkError("provider returned a score outside [0, 1]")
        scores.append(float(score))
    return scores, latencies_ms, cpu_seconds


def _rss_bytes() -> tuple[int, str]:
    try:
        statm = pathlib.Path("/proc/self/statm").read_text(encoding="ascii").split()
        return int(statm[1]) * os.sysconf("SC_PAGE_SIZE"), "procfs-current-rss"
    except (OSError, ValueError, IndexError):
        import resource

        scale = 1 if sys.platform == "darwin" else 1024
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale, "getrusage-max-rss"


async def measure_rss_per_stream(
    provider: VADProvider,
    sample_rate_hz: int,
    stream_count: int,
) -> dict[str, Any]:
    before, source = _rss_bytes()
    streams: list[VADStream] = []
    try:
        for _ in range(stream_count):
            streams.append(provider.new_stream(cast(Any, sample_rate_hz)))
        await asyncio.sleep(0)
        after, after_source = _rss_bytes()
        if after_source != source:
            source = f"{source}+{after_source}"
    finally:
        for stream in streams:
            stream.close()
    delta = max(0, after - before)
    return {
        "live_streams": len(streams),
        "baseline_rss_bytes": before,
        "live_rss_bytes": after,
        "incremental_rss_bytes": delta,
        "rss_bytes_per_live_stream": safe_ratio(float(delta), len(streams)),
        "source": source,
    }


async def _capacity_trial(
    provider: VADProvider,
    sample_rate_hz: int,
    concurrency: int,
    frames_per_stream: int,
    deadline_seconds: float,
) -> dict[str, Any]:
    streams: list[VADStream] = []
    latencies: list[float] = []
    frame = _synthetic_tone_frame(sample_rate_hz)
    passed = False
    error: str | None = None
    try:
        for _ in range(concurrency):
            streams.append(provider.new_stream(cast(Any, sample_rate_hz)))
        for _ in range(frames_per_stream):

            async def classify(stream: VADStream) -> float:
                started = time.perf_counter()
                await stream.score(frame)
                return time.perf_counter() - started

            round_latencies = await asyncio.gather(*(classify(stream) for stream in streams))
            latencies.extend(round_latencies)
        passed = bool(latencies) and max(latencies) <= deadline_seconds
    except Exception as exc:
        error = type(exc).__name__
    finally:
        for stream in streams:
            stream.close()
    return {
        "concurrency": concurrency,
        "passed": passed,
        "classifications": len(latencies),
        "p95_classification_latency_ms": (
            None if not latencies else percentile([value * 1_000 for value in latencies], 95)
        ),
        "maximum_classification_latency_ms": (None if not latencies else max(latencies) * 1_000),
        "error_type": error,
    }


async def maximum_sustainable_concurrency(
    provider: VADProvider,
    sample_rate_hz: int,
    maximum: int,
    frames_per_stream: int,
    deadline_seconds: float,
) -> dict[str, Any]:
    """Binary-search the largest observed concurrency whose every call meets deadline."""

    low = 0
    high = maximum
    attempts: list[dict[str, Any]] = []
    while low < high:
        candidate = (low + high + 1) // 2
        trial = await _capacity_trial(
            provider,
            sample_rate_hz,
            candidate,
            frames_per_stream,
            deadline_seconds,
        )
        attempts.append(trial)
        if trial["passed"]:
            low = candidate
        else:
            high = candidate - 1
    return {
        "maximum_sustainable_concurrent_streams": low,
        "search_upper_bound": maximum,
        "deadline_ms": deadline_seconds * 1_000,
        "frames_per_stream_per_trial": frames_per_stream,
        "criterion": "every score call in a trial completed within deadline_ms without error",
        "search": "monotonic binary search",
        "attempts": sorted(attempts, key=lambda item: item["concurrency"]),
    }


def _distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def provider_metadata(
    benchmark_id: str,
    provider: VADProvider,
    threshold: float,
    webrtc_mode: int | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "benchmark_id": benchmark_id,
        "provider_name": provider.name,
        "threshold": threshold,
        "provider_default_threshold": provider.default_threshold,
        "service_package_version": _distribution_version("indicconformer-realtime-asr"),
    }
    if benchmark_id == "energy":
        metadata["algorithm_version"] = "normalized-rms-energy-v1"
    elif benchmark_id.startswith("webrtc_mode_"):
        metadata.update(
            webrtc_mode=webrtc_mode,
            webrtcvad_wheels_version=_distribution_version("webrtcvad-wheels"),
        )
    elif benchmark_id == "silero":
        metadata.update(
            silero_version=SILERO_VERSION,
            silero_commit=SILERO_COMMIT,
            artifact_path_in_upstream=SILERO_ARTIFACT,
            artifact_sha256=SILERO_SHA256,
            onnxruntime_version=_distribution_version("onnxruntime"),
        )
    return metadata


def parse_thresholds(raw: Sequence[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for assignment in raw:
        name, separator, raw_value = assignment.partition("=")
        if not separator or name not in PROVIDER_IDS or name in thresholds:
            raise BenchmarkError(
                "thresholds must be unique PROVIDER=VALUE assignments for "
                + ", ".join(PROVIDER_IDS)
            )
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise BenchmarkError(f"invalid threshold for {name}") from exc
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise BenchmarkError(f"threshold for {name} must be finite and in [0, 1]")
        thresholds[name] = value
    return thresholds


def _provider_settings(
    benchmark_id: str,
    args: argparse.Namespace,
    sample_rate_hz: int,
) -> Any:
    from app.core.config import Settings
    from app.core.types import VADKind

    common: dict[str, Any] = {
        "environment": "development",
        "engine": "mock",
        "require_cuda": False,
        "offline": True,
        "sample_rate": 16_000,
        "vad_max_streams": max(args.max_concurrency, args.rss_streams),
        "vad_cpu_workers": args.workers,
        "vad_pending_capacity": args.pending_capacity,
        "vad_classification_deadline_seconds": args.deadline_seconds,
    }
    del sample_rate_hz  # VAD input rate is selected when each stream is created.
    if benchmark_id == "energy":
        return Settings(vad_provider=VADKind.ENERGY, **common)
    if benchmark_id.startswith("webrtc_mode_"):
        return Settings(
            vad_provider=VADKind.WEBRTC,
            vad_webrtc_mode=int(benchmark_id.rsplit("_", 1)[1]),
            **common,
        )
    return Settings(
        vad_provider=VADKind.SILERO,
        vad_model_path=args.silero_model,
        vad_model_sha256=SILERO_SHA256,
        **common,
    )


def build_provider(benchmark_id: str, args: argparse.Namespace, sample_rate_hz: int) -> VADProvider:
    from app.vad.factory import build_vad_provider

    return build_vad_provider(_provider_settings(benchmark_id, args, sample_rate_hz), NullMetrics())


async def benchmark_provider(
    benchmark_id: str,
    manifest: Manifest,
    args: argparse.Namespace,
    threshold_overrides: Mapping[str, float],
) -> dict[str, Any]:
    provider = build_provider(benchmark_id, args, manifest.sample_rate_hz)
    await provider.startup()
    try:
        threshold = threshold_overrides.get(benchmark_id, provider.default_threshold)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise BenchmarkError(f"provider {benchmark_id} has an invalid default threshold")
        overall = Aggregate()
        languages: dict[str, Aggregate] = {}
        conditions: dict[str, Aggregate] = {}
        variant_results: list[dict[str, Any]] = []
        expected_size = expected_frame_bytes(cast(Any, manifest.sample_rate_hz))
        for variant in manifest.variants:
            frames, reference = read_frames(variant, manifest.sample_rate_hz)
            stream = provider.new_stream(cast(Any, manifest.sample_rate_hz))
            try:
                scores, latencies, cpu_seconds = await score_frames(stream, frames, expected_size)
            finally:
                stream.close()
            predicted = [score >= threshold for score in scores]
            variant_aggregate = Aggregate()
            variant_aggregate.add(reference, predicted, latencies, cpu_seconds)
            overall.add(reference, predicted, latencies, cpu_seconds)
            languages.setdefault(variant.language, Aggregate()).add(
                reference, predicted, latencies, cpu_seconds
            )
            conditions.setdefault(variant.condition_key, Aggregate()).add(
                reference, predicted, latencies, cpu_seconds
            )
            variant_results.append(
                {
                    "clip_id": variant.clip_id,
                    "variant_id": variant.variant_id,
                    "language": variant.language,
                    "condition": variant.condition,
                    "metrics": variant_aggregate.summary(),
                }
            )
        rss = await measure_rss_per_stream(provider, manifest.sample_rate_hz, args.rss_streams)
        capacity = await maximum_sustainable_concurrency(
            provider,
            manifest.sample_rate_hz,
            args.max_concurrency,
            args.capacity_probe_frames,
            args.deadline_seconds,
        )
        mode = int(benchmark_id.rsplit("_", 1)[1]) if benchmark_id.startswith("webrtc") else None
        return {
            "metadata": provider_metadata(benchmark_id, provider, threshold, mode),
            "overall": overall.summary(),
            "by_language": {key: languages[key].summary() for key in sorted(languages)},
            "by_condition": {key: conditions[key].summary() for key in sorted(conditions)},
            "variants": variant_results,
            "resource_measurements": {
                "rss_per_live_stream": rss,
                "concurrency": capacity,
            },
        }
    finally:
        await provider.close()


def result_definitions() -> dict[str, str]:
    return {
        "frame": "one exact 20 ms mono PCM16LE frame",
        "frame_f1": "2*TP/(2*TP+FP+FN); null when the denominator is zero",
        "miss_rate": "FN/(TP+FN); null when there are no reference-speech frames",
        "false_positive_time_rate": (
            "FP/(FP+TN), equal to false-positive non-speech time divided by all "
            "labeled non-speech time; null when there is no non-speech time"
        ),
        "activation_segment": "a maximal contiguous run of predicted-speech frames",
        "false_activation_segment": ("an activation segment containing no reference-speech frame"),
        "false_activations_per_hour": (
            "false activation segments divided by labeled non-speech hours; "
            "null when there is no non-speech time"
        ),
        "reference_segment": "a maximal contiguous run of reference-speech frames",
        "onset_delay_ms": (
            "first overlapping predicted activation start minus reference segment "
            "start; undetected segments are excluded and counted via "
            "matched_reference_segments"
        ),
        "endpoint_delay_ms": (
            "last overlapping predicted activation end minus reference segment end; "
            "undetected segments are excluded and counted via matched_reference_segments"
        ),
        "percentile": ("R-7 linear interpolation (the NumPy default); null for no observations"),
        "cpu_rtf": (
            "process CPU seconds spent awaiting score calls divided by evaluated audio seconds"
        ),
        "rss_bytes_per_live_stream": (
            "non-negative current-RSS increase while holding N new live streams "
            "divided by N; allocator granularity can produce zero"
        ),
        "maximum_sustainable_concurrent_streams": (
            "largest concurrency observed by the reported bounded binary search where "
            "every score call completed without error within the configured deadline"
        ),
    }


def validate_result(result: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "benchmark",
        "manifest",
        "corpus",
        "environment",
        "definitions",
        "providers",
    }
    if set(result) != required:
        raise BenchmarkError("result root does not match output schema")
    if result["schema_version"] != SCHEMA_VERSION:
        raise BenchmarkError("result schema_version is invalid")
    if not isinstance(result["benchmark"], dict) or set(result["benchmark"]) != {
        "name",
        "started_unix_seconds",
        "accuracy_scope",
        "audio_or_transcripts_logged",
        "configuration",
    }:
        raise BenchmarkError("benchmark metadata does not match output schema")
    if not isinstance(result["manifest"], dict) or set(result["manifest"]) != {
        "schema_version",
        "hash_algorithm",
        "sha256",
    }:
        raise BenchmarkError("manifest metadata does not match output schema")
    if not isinstance(result["corpus"], dict) or set(result["corpus"]) != {"name", "version"}:
        raise BenchmarkError("corpus metadata does not match output schema")
    if not isinstance(result["environment"], dict) or set(result["environment"]) != {
        "python",
        "platform",
        "logical_cpu_count",
    }:
        raise BenchmarkError("environment metadata does not match output schema")
    if not isinstance(result["definitions"], dict):
        raise BenchmarkError("metric definitions must be an object")
    providers = result["providers"]
    if not isinstance(providers, dict) or set(providers) != set(PROVIDER_IDS):
        raise BenchmarkError("result must contain energy, four WebRTC modes, and Silero")
    summary_keys = {
        "frames",
        "audio_seconds",
        "confusion",
        "frame_f1",
        "miss_rate",
        "false_positive_time_rate",
        "false_activation_segments",
        "false_activations_per_hour",
        "reference_segments",
        "matched_reference_segments",
        "onset_delay_ms",
        "endpoint_delay_ms",
        "cpu_rtf",
        "classification_latency_ms",
    }
    for provider_id, provider_result in providers.items():
        if not isinstance(provider_result, dict) or set(provider_result) != {
            "metadata",
            "overall",
            "by_language",
            "by_condition",
            "variants",
            "resource_measurements",
        }:
            raise BenchmarkError(f"provider result {provider_id} does not match output schema")
        metadata = provider_result["metadata"]
        if not isinstance(metadata, dict) or not {
            "benchmark_id",
            "provider_name",
            "threshold",
            "provider_default_threshold",
            "service_package_version",
        }.issubset(metadata):
            raise BenchmarkError(f"provider metadata {provider_id} does not match output schema")
        summaries: list[Any] = [provider_result["overall"]]
        by_language = provider_result["by_language"]
        by_condition = provider_result["by_condition"]
        if not isinstance(by_language, dict) or not isinstance(by_condition, dict):
            raise BenchmarkError(f"provider aggregates {provider_id} must be objects")
        summaries.extend(by_language.values())
        summaries.extend(by_condition.values())
        variants = provider_result["variants"]
        if not isinstance(variants, list):
            raise BenchmarkError(f"provider variants {provider_id} must be an array")
        for variant in variants:
            if not isinstance(variant, dict) or set(variant) != {
                "clip_id",
                "variant_id",
                "language",
                "condition",
                "metrics",
            }:
                raise BenchmarkError(f"provider variant {provider_id} does not match output schema")
            summaries.append(variant["metrics"])
        if any(
            not isinstance(summary, dict) or set(summary) != summary_keys for summary in summaries
        ):
            raise BenchmarkError(f"provider metrics {provider_id} do not match output schema")
        resources = provider_result["resource_measurements"]
        if not isinstance(resources, dict) or set(resources) != {
            "rss_per_live_stream",
            "concurrency",
        }:
            raise BenchmarkError(f"provider resources {provider_id} do not match output schema")
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError("result is not finite JSON data") from exc


def _synthetic_tone_frame(sample_rate_hz: int) -> bytes:
    sample_count = sample_rate_hz * FRAME_DURATION_MS // 1_000
    samples = array(
        "h",
        (
            round(12_000 * math.sin(2 * math.pi * 440 * index / sample_rate_hz))
            for index in range(sample_count)
        ),
    )
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def _self_check_args() -> argparse.Namespace:
    return argparse.Namespace(
        max_concurrency=2,
        rss_streams=2,
        workers=1,
        pending_capacity=2,
        deadline_seconds=0.1,
        silero_model=None,
    )


async def self_check() -> dict[str, Any]:
    """Exercise the real Energy provider with exact synthetic silence/tone frames."""

    args = _self_check_args()
    provider = build_provider("energy", args, 16_000)
    await provider.startup()
    stream = provider.new_stream(16_000)
    try:
        silence = bytes(expected_frame_bytes(16_000))
        frames = [silence, _synthetic_tone_frame(16_000), silence]
        scores, _, _ = await score_frames(stream, frames, expected_frame_bytes(16_000))
        decisions = [score >= provider.default_threshold for score in scores]
        if decisions != [False, True, False]:
            raise BenchmarkError("Energy self-check did not separate synthetic tone from silence")
        return {
            "schema_version": SCHEMA_VERSION,
            "self_check": {
                "status": "ok",
                "provider": provider.name,
                "frame_duration_ms": FRAME_DURATION_MS,
                "frame_bytes": expected_frame_bytes(16_000),
                "frames_classified": len(frames),
            },
        }
    finally:
        stream.close()
        await provider.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, help="labeled JSON manifest")
    parser.add_argument("--silero-model", type=pathlib.Path, help="pinned Silero v6.2.1 ONNX file")
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="PROVIDER=VALUE",
        help="deterministic score threshold override; repeat per provider",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pending-capacity", type=int, default=128)
    parser.add_argument("--deadline-seconds", type=float, default=0.1)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--capacity-probe-frames", type=int, default=25)
    parser.add_argument("--rss-streams", type=int, default=16)
    parser.add_argument(
        "--output", type=pathlib.Path, help="write JSON atomically instead of stdout"
    )
    parser.add_argument(
        "--self-check", action="store_true", help="run synthetic Energy smoke check"
    )
    return parser


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise BenchmarkError(f"{name} must be positive")


def _validate_args(args: argparse.Namespace) -> None:
    if args.self_check:
        return
    if args.manifest is None or args.silero_model is None:
        raise BenchmarkError(
            "--manifest and --silero-model are required unless --self-check is used"
        )
    for value, name in (
        (args.workers, "--workers"),
        (args.pending_capacity, "--pending-capacity"),
        (args.max_concurrency, "--max-concurrency"),
        (args.capacity_probe_frames, "--capacity-probe-frames"),
        (args.rss_streams, "--rss-streams"),
    ):
        _positive_int(value, name)
    if not math.isfinite(args.deadline_seconds) or args.deadline_seconds <= 0:
        raise BenchmarkError("--deadline-seconds must be finite and positive")
    model_path = args.silero_model.expanduser().resolve(strict=True)
    try:
        observed = hashlib.sha256(model_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise BenchmarkError("Silero model is unreadable") from exc
    if observed != SILERO_SHA256:
        raise BenchmarkError("Silero model does not match the pinned v6.2.1 artifact SHA-256")
    args.silero_model = model_path


def _write_output(payload: Mapping[str, Any], output: pathlib.Path | None) -> None:
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(encoded)
        return
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


async def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    if args.self_check:
        return await self_check()
    manifest = load_manifest(cast(pathlib.Path, args.manifest))
    thresholds = parse_thresholds(args.threshold)
    provider_results: dict[str, Any] = {}
    started = time.time()
    for benchmark_id in PROVIDER_IDS:
        provider_results[benchmark_id] = await benchmark_provider(
            benchmark_id, manifest, args, thresholds
        )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": {
            "name": "labeled-streaming-vad-comparison",
            "started_unix_seconds": started,
            "accuracy_scope": "measurements apply only to the identified labeled corpus",
            "audio_or_transcripts_logged": False,
            "configuration": {
                "sample_rate_hz": manifest.sample_rate_hz,
                "frame_duration_ms": FRAME_DURATION_MS,
                "cpu_workers": args.workers,
                "pending_capacity": args.pending_capacity,
                "classification_deadline_seconds": args.deadline_seconds,
                "concurrency_search_upper_bound": args.max_concurrency,
                "capacity_probe_frames_per_stream": args.capacity_probe_frames,
                "rss_live_streams": args.rss_streams,
            },
        },
        "manifest": {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "hash_algorithm": "sha256-canonical-json-v1",
            "sha256": manifest.digest,
        },
        "corpus": {"name": manifest.corpus_name, "version": manifest.corpus_version},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "definitions": result_definitions(),
        "providers": provider_results,
    }
    validate_result(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(run(args))
        _write_output(result, args.output)
    except (BenchmarkError, OSError, ValueError) as exc:
        parser.exit(2, f"benchmark_vad.py: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
