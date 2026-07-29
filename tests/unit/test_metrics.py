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
