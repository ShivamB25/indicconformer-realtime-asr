from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import benchmark_vad


class RecordingStream:
    def __init__(self, scores: list[float]) -> None:
        self.scores = iter(scores)
        self.frames: list[bytes] = []

    async def score(self, pcm16_20ms: bytes) -> float:
        self.frames.append(pcm16_20ms)
        return next(self.scores)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_metric_formulas_use_frame_confusion_and_labeled_time() -> None:
    reference = [True, True, False, False]
    predicted = [True, False, True, False]

    counts = benchmark_vad.confusion_counts(reference, predicted)
    rates = benchmark_vad.metric_rates(**counts)

    assert counts == {
        "true_positive_frames": 1,
        "false_positive_frames": 1,
        "true_negative_frames": 1,
        "false_negative_frames": 1,
    }
    assert rates == {
        "frame_f1": 0.5,
        "miss_rate": 0.5,
        "false_positive_time_rate": 0.5,
    }


def test_zero_denominators_are_json_null_not_zero_or_nan() -> None:
    no_speech = benchmark_vad.metric_rates(
        true_positive_frames=0,
        false_positive_frames=0,
        true_negative_frames=4,
        false_negative_frames=0,
    )
    no_nonspeech = benchmark_vad.metric_rates(
        true_positive_frames=4,
        false_positive_frames=0,
        true_negative_frames=0,
        false_negative_frames=0,
    )
    empty = benchmark_vad.Aggregate().summary()

    assert no_speech["frame_f1"] is None
    assert no_speech["miss_rate"] is None
    assert no_speech["false_positive_time_rate"] == 0.0
    assert no_nonspeech["false_positive_time_rate"] is None
    assert empty["false_activations_per_hour"] is None
    assert empty["cpu_rtf"] is None
    assert empty["classification_latency_ms"] == {"p50": None, "p95": None}
    json.dumps(empty, allow_nan=False)


def test_activation_segments_are_maximal_runs_and_false_runs_have_no_reference_overlap() -> None:
    reference = [False, False, False, True, True, False, False, False]
    predicted = [True, True, False, True, False, False, True, False]
    assert benchmark_vad.true_runs(predicted) == [(0, 2), (3, 4), (6, 7)]
    assert benchmark_vad.false_activation_count(reference, predicted) == 2
    aggregate = benchmark_vad.Aggregate()
    aggregate.add([False] * 50, [True] + [False] * 49, [0.1] * 50, 0.01)
    assert aggregate.summary()["false_activations_per_hour"] == 3_600.0


def test_segment_delay_uses_first_and_last_overlapping_activation() -> None:
    reference = [False, False, True, True, True, False, False, True, True, False]
    predicted = [False, True, True, False, True, True, False, False, True, True]

    assert benchmark_vad.segment_delays(reference, predicted) == {
        "reference_segment_count": 2,
        "matched_reference_segment_count": 2,
        "onset_delays_ms": [-20, 20],
        "endpoint_delays_ms": [20, 20],
    }


def test_percentiles_use_r7_interpolation_and_empty_is_null() -> None:
    assert benchmark_vad.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert benchmark_vad.percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)
    assert benchmark_vad.percentile([], 95) is None


def test_manifest_hash_is_independent_of_whitespace_and_key_order() -> None:
    first = json.loads('{"schema_version":"1.0","corpus":{"name":"x","version":"1"}}')
    second = json.loads(
        """{
          "corpus": {"version": "1", "name": "x"},
          "schema_version": "1.0"
        }"""
    )

    assert benchmark_vad.canonical_manifest_hash(first) == benchmark_vad.canonical_manifest_hash(
        second
    )
    assert benchmark_vad.canonical_manifest_hash(first) == (
        "2c2f6e2e80dc94861af1fcbdb199cd0f5e7f1861008b4b53d0eab843a2e763e9"
    )


@pytest.mark.asyncio
async def test_score_frames_submits_each_exact_frame_once_in_order() -> None:
    frames = [bytes([index]) * 640 for index in range(3)]
    stream = RecordingStream([0.0, 0.5, 1.0])

    scores, latencies, cpu_seconds = await benchmark_vad.score_frames(stream, frames, 640)

    assert stream.frames == frames
    assert all(len(frame) == 640 for frame in stream.frames)
    assert scores == [0.0, 0.5, 1.0]
    assert len(latencies) == 3
    assert all(latency >= 0 for latency in latencies)
    assert cpu_seconds >= 0


@pytest.mark.asyncio
async def test_score_frames_rejects_non_twenty_millisecond_input_before_provider_call() -> None:
    stream = RecordingStream([0.5])

    with pytest.raises(benchmark_vad.BenchmarkError, match="expected 640"):
        await benchmark_vad.score_frames(stream, [bytes(639)], 640)

    assert stream.frames == []


def _provider_result() -> dict[str, Any]:
    summary = {
        "frames": 0,
        "audio_seconds": 0.0,
        "confusion": {},
        "frame_f1": None,
        "miss_rate": None,
        "false_positive_time_rate": None,
        "false_activation_segments": 0,
        "false_activations_per_hour": None,
        "reference_segments": 0,
        "matched_reference_segments": 0,
        "onset_delay_ms": {"p50": None, "p95": None},
        "endpoint_delay_ms": {"p50": None, "p95": None},
        "cpu_rtf": None,
        "classification_latency_ms": {"p50": None, "p95": None},
    }
    return {
        "metadata": {
            "benchmark_id": "energy",
            "provider_name": "energy",
            "threshold": 0.5,
            "provider_default_threshold": 0.5,
            "service_package_version": None,
        },
        "overall": summary,
        "by_language": {},
        "by_condition": {},
        "variants": [],
        "resource_measurements": {"rss_per_live_stream": {}, "concurrency": {}},
    }


def _valid_result() -> dict[str, Any]:
    return {
        "schema_version": benchmark_vad.SCHEMA_VERSION,
        "benchmark": {
            "name": "test",
            "started_unix_seconds": 0.0,
            "accuracy_scope": "test corpus only",
            "audio_or_transcripts_logged": False,
            "configuration": {},
        },
        "manifest": {
            "schema_version": "1.0",
            "hash_algorithm": "sha256-canonical-json-v1",
            "sha256": "0" * 64,
        },
        "corpus": {"name": "test", "version": "1"},
        "environment": {
            "python": "3.11",
            "platform": "test",
            "logical_cpu_count": 1,
        },
        "definitions": {},
        "providers": {
            provider_id: _provider_result() for provider_id in benchmark_vad.PROVIDER_IDS
        },
    }


def test_output_schema_requires_every_baseline_and_provisional_provider() -> None:
    result = _valid_result()
    benchmark_vad.validate_result(result)

    del result["providers"]["webrtc_mode_3"]
    with pytest.raises(benchmark_vad.BenchmarkError, match="four WebRTC"):
        benchmark_vad.validate_result(result)


def test_output_schema_rejects_non_finite_json_values() -> None:
    result = _valid_result()
    result["benchmark"]["started_unix_seconds"] = float("nan")

    with pytest.raises(benchmark_vad.BenchmarkError, match="finite JSON"):
        benchmark_vad.validate_result(result)


def test_manifest_schema_is_machine_readable_draft_2020_12() -> None:
    schema_path = Path("scripts/vad_benchmark_manifest.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["frame_duration_ms"] == {"const": 20}
    assert schema["$defs"]["variant"]["required"] == ["id", "path", "sha256", "condition"]


@pytest.mark.asyncio
async def test_self_check_exercises_synthetic_energy_frames() -> None:
    result = await benchmark_vad.self_check()

    assert result == {
        "schema_version": benchmark_vad.SCHEMA_VERSION,
        "self_check": {
            "status": "ok",
            "provider": "energy",
            "frame_duration_ms": 20,
            "frame_bytes": 640,
            "frames_classified": 3,
        },
    }
