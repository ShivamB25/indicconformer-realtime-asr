"""Closed error/rejection classification contracts for bounded metrics."""

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from app.main import create_app
from app.observability import MetricCode, Metrics, is_server_error

SERVER_ERRORS = {
    MetricCode.INFERENCE_ERROR,
    MetricCode.INTERNAL_ERROR,
    MetricCode.TELEMETRY_ERROR,
    MetricCode.TIMEOUT,
}


def test_every_metric_code_has_the_expected_classification() -> None:
    assert {code for code in MetricCode if is_server_error(code)} == SERVER_ERRORS
    assert all(not is_server_error(code) for code in set(MetricCode) - SERVER_ERRORS)


def test_protocol_failures_route_to_exactly_one_counter() -> None:
    registry = CollectorRegistry()
    metrics = Metrics(registry)

    metrics.record_protocol_failure(MetricCode.INFERENCE_ERROR)
    metrics.record_protocol_failure(MetricCode.INVALID_FRAME_SIZE)
    payload = generate_latest(registry).decode("utf-8")

    assert 'asr_errors_total{code="inference_error"} 1.0' in payload
    assert 'asr_rejections_total{code="invalid_frame_size"} 1.0' in payload
    assert 'asr_rejections_total{code="inference_error"}' not in payload
    assert 'asr_errors_total{code="invalid_frame_size"}' not in payload


def test_unclassified_protocol_code_fails_instead_of_becoming_a_refusal() -> None:
    metrics = Metrics(CollectorRegistry())

    with pytest.raises(ValueError, match="unclassified protocol metric code"):
        metrics.record_protocol_failure("NEW_SERVER_FAILURE")


def test_an_unmatched_session_end_cannot_make_the_gauge_negative() -> None:
    registry = CollectorRegistry()
    metrics = Metrics(registry)

    metrics.session_ended()
    metrics.session_ended()
    assert "asr_active_sessions 0.0" in generate_latest(registry).decode("utf-8")

    metrics.session_started()
    assert "asr_active_sessions 1.0" in generate_latest(registry).decode("utf-8")
    metrics.session_ended()
    assert "asr_active_sessions 0.0" in generate_latest(registry).decode("utf-8")


def test_recording_a_telemetry_failure_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = Metrics(CollectorRegistry())

    def fail(_: str) -> None:
        raise RuntimeError("counter backend failed")

    monkeypatch.setattr(metrics._errors, "labels", fail)  # noqa: SLF001
    metrics.record_telemetry_failure()


def test_each_application_owns_an_isolated_collector_registry() -> None:
    first = create_app()
    second = create_app()

    assert first.state.metrics is not second.state.metrics
    assert first.state.metrics.registry is not second.state.metrics.registry


def _vad_samples(registry: CollectorRegistry) -> set[str]:
    return {
        line
        for line in generate_latest(registry).decode("utf-8").splitlines()
        if line.startswith("asr_vad_")
        and not line.startswith("asr_vad_queue_wait_seconds_bucket")
        and not line.startswith("asr_vad_inference_seconds_bucket")
        and not line.startswith("asr_vad_queue_wait_seconds_created")
        and not line.startswith("asr_vad_inference_seconds_created")
        and not line.startswith("asr_vad_decisions_created")
        and not line.startswith("asr_vad_endpoint_events_created")
        and not line.startswith("asr_vad_errors_created")
    }


def test_vad_metrics_have_exact_bounded_exposition() -> None:
    registry = CollectorRegistry()
    metrics = Metrics(registry)

    metrics.set_vad_provider(" SILERO ")
    metrics.vad_stream_started("silero")
    metrics.set_vad_queue_depth(2)
    metrics.record_vad_queue_wait("silero", 0.0)
    metrics.record_vad_inference("silero", 0.0)
    metrics.record_vad_decision("silero", "speech")
    metrics.record_vad_decision("silero", False)
    metrics.record_vad_endpoint_event(" OPENAI ")
    metrics.record_vad_runtime_error("silero", "deadline")

    assert _vad_samples(registry) == {
        'asr_vad_provider_info{provider="silero"} 1.0',
        'asr_vad_live_streams{provider="silero"} 1.0',
        "asr_vad_queue_depth 2.0",
        'asr_vad_queue_wait_seconds_count{provider="silero"} 1.0',
        'asr_vad_queue_wait_seconds_sum{provider="silero"} 0.0',
        'asr_vad_inference_seconds_count{provider="silero"} 1.0',
        'asr_vad_inference_seconds_sum{provider="silero"} 0.0',
        'asr_vad_decisions_total{provider="silero",result="speech"} 1.0',
        'asr_vad_decisions_total{provider="silero",result="silence"} 1.0',
        'asr_vad_endpoint_events_total{protocol="openai"} 1.0',
        'asr_vad_errors_total{error="deadline",provider="silero"} 1.0',
    }


@pytest.mark.parametrize("depth", [-1, True, 1.5])
def test_vad_queue_depth_requires_a_nonnegative_integer(depth: object) -> None:
    metrics = Metrics(CollectorRegistry())

    with pytest.raises(ValueError, match="non-negative integer"):
        metrics.set_vad_queue_depth(depth)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-0.001, float("nan"), float("inf"), True])
@pytest.mark.parametrize("method_name", ["record_vad_queue_wait", "record_vad_inference"])
def test_vad_durations_must_be_finite_and_nonnegative(
    method_name: str,
    value: float,
) -> None:
    metrics = Metrics(CollectorRegistry())

    with pytest.raises(ValueError, match="finite and non-negative"):
        getattr(metrics, method_name)("energy", value)


@pytest.mark.parametrize(
    ("method_name", "args", "label_name"),
    [
        ("set_vad_provider", ("unknown",), "provider"),
        ("vad_stream_started", ("unknown",), "provider"),
        ("record_vad_queue_wait", ("unknown", 0.0), "provider"),
        ("record_vad_inference", ("unknown", 0.0), "provider"),
        ("record_vad_decision", ("energy", "maybe"), "result"),
        ("record_vad_endpoint_event", ("grpc",), "protocol"),
        ("record_vad_runtime_error", ("energy", "timeout"), "error"),
    ],
)
def test_vad_labels_reject_values_outside_closed_sets(
    method_name: str,
    args: tuple[object, ...],
    label_name: str,
) -> None:
    registry = CollectorRegistry()
    metrics = Metrics(registry)

    with pytest.raises(ValueError, match=f"VAD {label_name} must be one of"):
        getattr(metrics, method_name)(*args)
    assert _vad_samples(registry) == {"asr_vad_queue_depth 0.0"}


def test_vad_labels_are_normalized_within_their_closed_sets() -> None:
    registry = CollectorRegistry()
    metrics = Metrics(registry)

    metrics.set_vad_provider(" WEBRTC ")
    metrics.record_vad_decision("webRTC", " SILENCE ")
    metrics.record_vad_endpoint_event(" NATIVE ")
    metrics.record_vad_runtime_error("WEBRTC", " INFERENCE ")
    payload = generate_latest(registry).decode("utf-8")

    assert 'asr_vad_provider_info{provider="webrtc"} 1.0' in payload
    assert 'asr_vad_decisions_total{provider="webrtc",result="silence"} 1.0' in payload
    assert 'asr_vad_endpoint_events_total{protocol="native"} 1.0' in payload
    assert 'asr_vad_errors_total{error="inference",provider="webrtc"} 1.0' in payload


def test_vad_live_stream_gauges_are_balanced_per_provider() -> None:
    registry = CollectorRegistry()
    metrics = Metrics(registry)

    metrics.vad_stream_ended("energy")
    metrics.vad_stream_started("energy")
    metrics.vad_stream_started("energy")
    metrics.vad_stream_started("silero")
    metrics.vad_stream_ended("energy")
    metrics.vad_stream_ended("energy")
    metrics.vad_stream_ended("energy")
    payload = generate_latest(registry).decode("utf-8")

    assert 'asr_vad_live_streams{provider="energy"} 0.0' in payload
    assert 'asr_vad_live_streams{provider="silero"} 1.0' in payload


def test_selected_vad_provider_is_a_one_hot_gauge() -> None:
    registry = CollectorRegistry()
    metrics = Metrics(registry)

    metrics.set_vad_provider("silero")
    metrics.set_vad_provider("energy")
    payload = generate_latest(registry).decode("utf-8")

    assert 'asr_vad_provider_info{provider="silero"} 0.0' in payload
    assert 'asr_vad_provider_info{provider="energy"} 1.0' in payload


def test_vad_collectors_are_isolated_between_registries() -> None:
    first_registry = CollectorRegistry()
    second_registry = CollectorRegistry()
    first = Metrics(first_registry)
    Metrics(second_registry)

    first.set_vad_provider("silero")
    first.vad_stream_started("silero")
    first.record_vad_decision("silero", True)

    assert 'provider="silero"' in generate_latest(first_registry).decode("utf-8")
    assert 'provider="silero"' not in generate_latest(second_registry).decode("utf-8")
    assert "asr_vad_queue_depth 0.0" in generate_latest(second_registry).decode("utf-8")
