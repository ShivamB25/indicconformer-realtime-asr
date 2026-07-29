"""Bounded-cardinality Prometheus metrics for the ASR service."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from threading import Lock
from typing import Self, cast
from weakref import WeakKeyDictionary

from fastapi import APIRouter, FastAPI, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from app.core.types import LanguageCode, ProcessingMode, VADKind

_QUEUE_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)
_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)
_RTF_BUCKETS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)


class MetricCode(StrEnum):
    """Closed error/rejection reasons exposed as label values."""

    BAD_REQUEST = "bad_request"
    EMPTY_UTTERANCE = "empty_utterance"
    FRAME_TOO_LARGE = "frame_too_large"
    IDLE_TIMEOUT = "idle_timeout"
    INFERENCE_ERROR = "inference_error"
    INTERNAL_ERROR = "internal_error"
    INVALID_AUDIO = "invalid_audio"
    INVALID_FRAME_SIZE = "invalid_frame_size"
    INVALID_SESSION = "invalid_session"
    MALFORMED_EVENT = "malformed_event"
    SERVER_BUSY = "server_busy"
    SERVICE_UNAVAILABLE = "service_unavailable"
    SESSION_ALREADY_STARTED = "session_already_started"
    SESSION_LIMIT = "session_limit"
    SESSION_REQUIRED = "session_required"
    TELEMETRY_ERROR = "telemetry_error"
    TIMEOUT = "timeout"
    UNKNOWN_EVENT = "unknown_event"
    UPLOAD_TOO_LARGE = "upload_too_large"
    UTTERANCE_TOO_LONG = "utterance_too_long"
    VALIDATION_ERROR = "validation_error"
    OTHER = "other"


_LANGUAGES = frozenset(item.value for item in LanguageCode)
_MODES = frozenset(item.value for item in ProcessingMode)
_CODES = frozenset(item.value for item in MetricCode)
_VAD_PROVIDERS = frozenset(item.value for item in VADKind)
_VAD_PROTOCOLS = frozenset(("native", "openai"))
_VAD_RESULTS = frozenset(("speech", "silence"))
_VAD_ERRORS = frozenset(("capacity", "deadline", "inference"))

# Every code is classified exactly once: a server error means the service
# accepted work and failed it, a refusal means the input or state was declined.
# A new MetricCode must join one of these sets or import fails below.
_SERVER_ERRORS = frozenset(
    (
        MetricCode.INFERENCE_ERROR,
        MetricCode.INTERNAL_ERROR,
        MetricCode.TELEMETRY_ERROR,
        MetricCode.TIMEOUT,
    )
)
_REFUSALS = frozenset(
    (
        MetricCode.BAD_REQUEST,
        MetricCode.EMPTY_UTTERANCE,
        MetricCode.FRAME_TOO_LARGE,
        MetricCode.IDLE_TIMEOUT,
        MetricCode.INVALID_AUDIO,
        MetricCode.INVALID_FRAME_SIZE,
        MetricCode.INVALID_SESSION,
        MetricCode.MALFORMED_EVENT,
        MetricCode.OTHER,
        MetricCode.SERVER_BUSY,
        MetricCode.SERVICE_UNAVAILABLE,
        MetricCode.SESSION_ALREADY_STARTED,
        MetricCode.SESSION_LIMIT,
        MetricCode.SESSION_REQUIRED,
        MetricCode.UNKNOWN_EVENT,
        MetricCode.UPLOAD_TOO_LARGE,
        MetricCode.UTTERANCE_TOO_LONG,
        MetricCode.VALIDATION_ERROR,
    )
)

if frozenset(MetricCode) != _SERVER_ERRORS | _REFUSALS or _SERVER_ERRORS & _REFUSALS:
    raise RuntimeError("every MetricCode must be either a server error or a refusal")

_ERROR_CODES = frozenset(code.value for code in _SERVER_ERRORS)


def normalize_language(value: str | LanguageCode) -> str:
    """Collapse unsupported language values into one bounded fallback."""

    raw = value.value if isinstance(value, LanguageCode) else value
    return raw if raw in _LANGUAGES else "unknown"


def normalize_mode(value: str | ProcessingMode) -> str:
    """Collapse unsupported mode values into one bounded fallback."""

    raw = value.value if isinstance(value, ProcessingMode) else value
    return raw if raw in _MODES else "unknown"


def normalize_code(value: str | MetricCode) -> str:
    """Collapse unsupported reason codes into the bounded ``other`` value."""

    raw = value.value if isinstance(value, MetricCode) else value.strip().lower().replace("-", "_")
    return raw if raw in _CODES else MetricCode.OTHER.value


def is_server_error(code: str | MetricCode) -> bool:
    """Report whether a code counts against the service rather than the caller."""

    return normalize_code(code) in _ERROR_CODES


def _nonnegative(value: float, name: str) -> float:
    message = f"{name} must be finite and non-negative"
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(message)
    return result


def _normalize_vad_label(value: str, allowed: frozenset[str], name: str) -> str:
    """Normalize one closed VAD label and reject values outside its domain."""

    if not isinstance(value, str):
        raise ValueError(f"VAD {name} must be one of {sorted(allowed)}")
    result = value.strip().lower().replace("-", "_")
    if result not in allowed:
        raise ValueError(f"VAD {name} must be one of {sorted(allowed)}")
    return result


def _normalize_vad_provider(provider: str) -> str:
    return _normalize_vad_label(provider, _VAD_PROVIDERS, "provider")


def _normalize_vad_protocol(protocol: str) -> str:
    return _normalize_vad_label(protocol, _VAD_PROTOCOLS, "protocol")


def _normalize_vad_result(result: str | bool) -> str:
    if isinstance(result, bool):
        return "speech" if result else "silence"
    return _normalize_vad_label(result, _VAD_RESULTS, "result")


def _normalize_vad_error(error: str) -> str:
    return _normalize_vad_label(error, _VAD_ERRORS, "error")


class Metrics:
    """Collector facade that bounds all labels and is unique per registry."""

    def __new__(cls, registry: CollectorRegistry = REGISTRY) -> Self:
        with _LOCK:
            current = _INSTANCES.get(registry)
            if current is not None:
                return cast(Self, current)
            instance = super().__new__(cls)
            instance._initialize(registry)
            _INSTANCES[registry] = instance
            return instance

    def __init__(self, registry: CollectorRegistry = REGISTRY) -> None:
        # Initialization is performed once, under the lock, in __new__.
        del registry

    def _initialize(self, registry: CollectorRegistry) -> None:
        self.registry = registry
        self._session_lock = Lock()
        self._active_sessions = 0
        self._sessions = Gauge(
            "active_sessions",
            "Active realtime sessions.",
            namespace="asr",
            registry=registry,
        )
        self._rejections = Counter(
            "rejections",
            "Rejected requests or sessions.",
            ("code",),
            namespace="asr",
            registry=registry,
        )
        self._errors = Counter(
            "errors",
            "Operational errors.",
            ("code",),
            namespace="asr",
            registry=registry,
        )
        self._transcriptions = Counter(
            "transcriptions",
            "Successful transcriptions.",
            ("language", "mode"),
            namespace="asr",
            registry=registry,
        )
        self._queue_depth = Gauge(
            "queue_depth",
            "Queued inference jobs.",
            namespace="asr",
            registry=registry,
        )
        self._queue_wait = Histogram(
            "queue_wait_seconds",
            "Inference queue wait.",
            ("mode",),
            namespace="asr",
            registry=registry,
            buckets=_QUEUE_BUCKETS,
        )
        self._audio = Counter(
            "audio_seconds",
            "Accepted audio duration.",
            ("language", "mode"),
            namespace="asr",
            registry=registry,
        )
        self._partial = self._latency_histogram(registry, "partial_latency_seconds")
        self._final = self._latency_histogram(registry, "final_latency_seconds")
        self._encoder = self._latency_histogram(registry, "encoder_latency_seconds")
        self._ctc = self._latency_histogram(registry, "ctc_latency_seconds")
        self._rnnt = self._latency_histogram(registry, "rnnt_latency_seconds")
        self._rtf = Histogram(
            "realtime_factor",
            "Processing seconds divided by audio seconds.",
            ("language", "mode"),
            namespace="asr",
            registry=registry,
            buckets=_RTF_BUCKETS,
        )
        self._vad_stream_lock = Lock()
        self._active_vad_streams: dict[str, int] = {}
        self._selected_vad_provider: str | None = None
        self._vad_provider = Gauge(
            "vad_provider_info",
            "Explicitly selected VAD provider.",
            ("provider",),
            namespace="asr",
            registry=registry,
        )
        self._vad_live_streams = Gauge(
            "vad_live_streams",
            "Live VAD streams.",
            ("provider",),
            namespace="asr",
            registry=registry,
        )
        self._vad_queue_depth = Gauge(
            "vad_queue_depth",
            "Queued VAD inference jobs.",
            namespace="asr",
            registry=registry,
        )
        self._vad_queue_wait = Histogram(
            "vad_queue_wait_seconds",
            "VAD inference queue wait.",
            ("provider",),
            namespace="asr",
            registry=registry,
            buckets=_QUEUE_BUCKETS,
        )
        self._vad_inference = Histogram(
            "vad_inference_seconds",
            "VAD inference latency.",
            ("provider",),
            namespace="asr",
            registry=registry,
            buckets=_LATENCY_BUCKETS,
        )
        self._vad_decisions = Counter(
            "vad_decisions",
            "VAD frame decisions.",
            ("provider", "result"),
            namespace="asr",
            registry=registry,
        )
        self._vad_endpoint_events = Counter(
            "vad_endpoint_events",
            "VAD endpoint events.",
            ("protocol",),
            namespace="asr",
            registry=registry,
        )
        self._vad_errors = Counter(
            "vad_errors",
            "VAD runtime errors.",
            ("provider", "error"),
            namespace="asr",
            registry=registry,
        )

    @staticmethod
    def _latency_histogram(registry: CollectorRegistry, name: str) -> Histogram:
        return Histogram(
            name,
            f"ASR {name.replace('_', ' ')}.",
            ("language", "mode"),
            namespace="asr",
            registry=registry,
            buckets=_LATENCY_BUCKETS,
        )

    def session_started(self) -> None:
        with self._session_lock:
            self._sessions.inc()
            self._active_sessions += 1

    def session_ended(self) -> None:
        # A caller whose start failed still runs its cleanup, so an unmatched end is
        # dropped rather than driving the gauge below zero for the process lifetime.
        with self._session_lock:
            if self._active_sessions == 0:
                return
            self._active_sessions -= 1
            self._sessions.dec()

    @contextmanager
    def active_session(self) -> Iterator[None]:
        """Track a realtime session and release the gauge slot on every exit."""

        self.session_started()
        try:
            yield
        finally:
            self.session_ended()

    def record_rejection(self, code: str | MetricCode) -> None:
        self._rejections.labels(normalize_code(code)).inc()

    def record_error(self, code: str | MetricCode) -> None:
        self._errors.labels(normalize_code(code)).inc()

    def record_telemetry_failure(self) -> None:
        """Count a failure of the recording path itself, never raising in turn.

        Callers reach this from an exception handler guarding work that already
        succeeded, so a second failure here must stay silent rather than convert
        degraded observability into a failed request.
        """

        try:
            self._errors.labels(MetricCode.TELEMETRY_ERROR.value).inc()
        except Exception:
            pass

    def record_protocol_failure(self, code: str | MetricCode) -> None:
        """Route one closed protocol code, failing loudly on programmer drift."""

        normalized = normalize_code(code)
        explicit_other = code is MetricCode.OTHER or (
            isinstance(code, str) and code.strip().lower() == MetricCode.OTHER.value
        )
        if normalized == MetricCode.OTHER.value and not explicit_other:
            raise ValueError(f"unclassified protocol metric code: {code!r}")
        counter = self._errors if is_server_error(normalized) else self._rejections
        counter.labels(normalized).inc()

    def record_transcription(
        self,
        language: str | LanguageCode,
        mode: str | ProcessingMode,
    ) -> None:
        self._transcriptions.labels(
            normalize_language(language),
            normalize_mode(mode),
        ).inc()

    def set_queue_depth(self, depth: int) -> None:
        if isinstance(depth, bool) or depth < 0:
            raise ValueError("queue depth must be non-negative")
        self._queue_depth.set(depth)

    def record_queue_wait(
        self,
        mode: str | ProcessingMode,
        seconds: float,
    ) -> None:
        self._queue_wait.labels(normalize_mode(mode)).observe(_nonnegative(seconds, "queue wait"))

    def record_audio_seconds(
        self,
        language: str | LanguageCode,
        mode: str | ProcessingMode,
        seconds: float,
    ) -> None:
        self._audio.labels(
            normalize_language(language),
            normalize_mode(mode),
        ).inc(_nonnegative(seconds, "audio duration"))

    def _observe_latency(
        self,
        metric: Histogram,
        language: str | LanguageCode,
        mode: str | ProcessingMode,
        seconds: float,
    ) -> None:
        metric.labels(
            normalize_language(language),
            normalize_mode(mode),
        ).observe(_nonnegative(seconds, "latency"))

    def record_partial_latency(
        self,
        language: str | LanguageCode,
        mode: str | ProcessingMode,
        seconds: float,
    ) -> None:
        self._observe_latency(self._partial, language, mode, seconds)

    def record_final_latency(
        self,
        language: str | LanguageCode,
        mode: str | ProcessingMode,
        seconds: float,
    ) -> None:
        self._observe_latency(self._final, language, mode, seconds)

    def record_encoder_latency(
        self,
        language: str | LanguageCode,
        mode: str | ProcessingMode,
        seconds: float,
    ) -> None:
        self._observe_latency(self._encoder, language, mode, seconds)

    def record_ctc_latency(
        self,
        language: str | LanguageCode,
        mode: str | ProcessingMode,
        seconds: float,
    ) -> None:
        self._observe_latency(self._ctc, language, mode, seconds)

    def record_rnnt_latency(
        self,
        language: str | LanguageCode,
        mode: str | ProcessingMode,
        seconds: float,
    ) -> None:
        self._observe_latency(self._rnnt, language, mode, seconds)

    def record_realtime_factor(
        self,
        language: str | LanguageCode,
        mode: str | ProcessingMode,
        factor: float,
    ) -> None:
        self._rtf.labels(
            normalize_language(language),
            normalize_mode(mode),
        ).observe(_nonnegative(factor, "realtime factor"))

    def set_vad_provider(self, provider: str) -> None:
        """Expose the process-selected provider as a one-hot bounded gauge."""

        normalized = _normalize_vad_provider(provider)
        with self._vad_stream_lock:
            previous = self._selected_vad_provider
            if previous == normalized:
                return
            if previous is not None:
                self._vad_provider.labels(previous).set(0)
            self._vad_provider.labels(normalized).set(1)
            self._selected_vad_provider = normalized

    def vad_stream_started(self, provider: str) -> None:
        """Increment the provider's live-stream gauge."""

        normalized = _normalize_vad_provider(provider)
        with self._vad_stream_lock:
            active = self._active_vad_streams.get(normalized, 0) + 1
            self._active_vad_streams[normalized] = active
            self._vad_live_streams.labels(normalized).set(active)

    def vad_stream_ended(self, provider: str) -> None:
        """Decrement a live-stream gauge without allowing unmatched cleanup below zero."""

        normalized = _normalize_vad_provider(provider)
        with self._vad_stream_lock:
            active = self._active_vad_streams.get(normalized, 0)
            if active == 0:
                self._vad_live_streams.labels(normalized).set(0)
                return
            active -= 1
            self._active_vad_streams[normalized] = active
            self._vad_live_streams.labels(normalized).set(active)

    def set_vad_queue_depth(self, depth: int) -> None:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError("VAD queue depth must be a non-negative integer")
        self._vad_queue_depth.set(depth)

    def record_vad_queue_wait(self, provider: str, seconds: float) -> None:
        self._vad_queue_wait.labels(_normalize_vad_provider(provider)).observe(
            _nonnegative(seconds, "VAD queue wait")
        )

    def record_vad_inference(self, provider: str, seconds: float) -> None:
        self._vad_inference.labels(_normalize_vad_provider(provider)).observe(
            _nonnegative(seconds, "VAD inference latency")
        )

    def record_vad_decision(self, provider: str, result: str | bool) -> None:
        self._vad_decisions.labels(
            _normalize_vad_provider(provider),
            _normalize_vad_result(result),
        ).inc()

    def record_vad_endpoint_event(self, protocol: str) -> None:
        self._vad_endpoint_events.labels(_normalize_vad_protocol(protocol)).inc()

    def record_vad_runtime_error(self, provider: str, code: str) -> None:
        self._vad_errors.labels(
            _normalize_vad_provider(provider),
            _normalize_vad_error(code),
        ).inc()


_LOCK = Lock()
_INSTANCES: WeakKeyDictionary[CollectorRegistry, Metrics] = WeakKeyDictionary()


def get_metrics(registry: CollectorRegistry = REGISTRY) -> Metrics:
    """Return the default metrics or an isolated set for a test registry."""

    return Metrics(registry)


def create_metrics_router(registry: CollectorRegistry = REGISTRY) -> APIRouter:
    """Create a Prometheus exposition router backed by ``registry``."""

    get_metrics(registry)
    result = APIRouter(tags=["observability"])

    @result.get("/metrics", include_in_schema=False, response_class=Response)
    def prometheus_metrics() -> Response:
        return Response(
            generate_latest(registry),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    return result


def install_metrics(app: FastAPI) -> None:
    """Install an isolated registry and endpoint exactly once per app."""
    if getattr(app.state, "_asr_metrics_installed", False):
        return
    registry = CollectorRegistry(auto_describe=True)
    app.state.metrics = get_metrics(registry)
    app.include_router(create_metrics_router(registry))
    app.state._asr_metrics_installed = True
