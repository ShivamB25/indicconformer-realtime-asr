#!/usr/bin/env python3
"""End-to-end HTTP benchmark for the IndicConformer transcription API.

The benchmark intentionally uses only Python's standard library so the exact
same client runs in CPU CI and inside a release image on a GPU runner. It never
logs audio bytes or transcripts. GPU metadata comes only from ``nvidia-smi``;
when that command is absent or fails, GPU fields are JSON ``null`` rather than
invented values.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import itertools
import json
import math
import pathlib
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from collections.abc import Callable, Sequence
from typing import Any, Final

SCHEMA_VERSION: Final[str] = "1.0"
DEFAULT_TIMEOUT_SECONDS: Final[float] = 120.0
MODES: Final[tuple[str, ...]] = ("latency", "hybrid", "accuracy")
DECODERS: Final[tuple[str, ...]] = ("ctc", "rnnt")
FINAL_DECODER_BY_MODE: Final[dict[str, str]] = {
    "latency": "ctc",
    "hybrid": "rnnt",
    "accuracy": "rnnt",
}
LANGUAGES: Final[tuple[str, ...]] = (
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
)


class BenchmarkError(RuntimeError):
    """A workload or response violated the benchmark contract."""


@dataclasses.dataclass(frozen=True, slots=True)
class Sample:
    path: pathlib.Path
    language: str
    audio_duration_ms: int
    audio_bytes: bytes


@dataclasses.dataclass(frozen=True, slots=True)
class Measurement:
    latency_ms: float
    inference_ms: float
    audio_duration_ms: int

    @property
    def end_to_end_rtf(self) -> float:
        return self.latency_ms / self.audio_duration_ms

    @property
    def inference_rtf(self) -> float:
        return self.inference_ms / self.audio_duration_ms


def _csv_values(
    raw_values: Sequence[str],
    *,
    cast: type[str] | type[int] | type[float],
) -> list[Any]:
    values: list[Any] = []
    for raw in raw_values:
        for item in raw.split(","):
            item = item.strip()
            if item:
                values.append(cast(item))
    if not values:
        raise BenchmarkError("matrix axis cannot be empty")
    return values


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a linearly interpolated percentile (R-7 / NumPy default)."""
    if not values:
        raise BenchmarkError("cannot compute a percentile of no measurements")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise BenchmarkError("cannot summarise no measurements")
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values),
    }


def _read_api_key(path: pathlib.Path | None) -> str | None:
    """Read one bounded bearer token without ever including it in diagnostics."""
    if path is None:
        return None
    expanded = path.expanduser()
    try:
        if expanded.is_symlink() or not expanded.is_file():
            raise BenchmarkError("API key path must be a regular file")
        key = expanded.read_text(encoding="utf-8")
    except BenchmarkError:
        raise
    except (OSError, UnicodeError):
        raise BenchmarkError("API key file cannot be read") from None
    if key.endswith("\r\n"):
        key = key[:-2]
    elif key.endswith("\n"):
        key = key[:-1]
    if (
        len(key) < 32
        or len(key) > 4096
        or not key.isascii()
        or any(character.isspace() for character in key)
    ):
        raise BenchmarkError("API key must contain one 32-4096 character ASCII token")
    return key


def _read_sample(path: pathlib.Path, language: str) -> Sample:
    if language not in LANGUAGES:
        raise BenchmarkError(f"unsupported language {language!r}")
    resolved = path.expanduser().resolve(strict=True)
    try:
        with wave.open(str(resolved), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
    except (wave.Error, EOFError) as exc:
        raise BenchmarkError(f"{resolved}: not a readable PCM WAV file") from exc
    if (channels, sample_width, sample_rate) != (1, 2, 16_000):
        raise BenchmarkError(
            f"{resolved}: expected mono signed PCM16 at 16 kHz; got "
            f"channels={channels}, sample_width={sample_width}, sample_rate={sample_rate}"
        )
    if frame_count <= 0:
        raise BenchmarkError(f"{resolved}: audio is empty")
    duration_ms = round(frame_count * 1000 / sample_rate)
    return Sample(resolved, language, duration_ms, resolved.read_bytes())


def _load_samples(args: argparse.Namespace) -> list[Sample]:
    descriptors: list[tuple[pathlib.Path, str]] = []
    if args.manifest is not None:
        manifest_path = args.manifest.expanduser().resolve(strict=True)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BenchmarkError(f"{manifest_path}: invalid JSON manifest") from exc
        records = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(records, list) or not records:
            raise BenchmarkError(
                'manifest must be a non-empty JSON array or {"samples": [...]} object'
            )
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise BenchmarkError(f"manifest sample {index} must be an object")
            raw_path, language = record.get("path"), record.get("language")
            if not isinstance(raw_path, str) or not isinstance(language, str):
                raise BenchmarkError(
                    f"manifest sample {index} needs string path and language fields"
                )
            sample_path = pathlib.Path(raw_path)
            if not sample_path.is_absolute():
                sample_path = manifest_path.parent / sample_path
            descriptors.append((sample_path, language))
    elif args.audio is not None and args.language is not None:
        descriptors.append((args.audio, args.language))
    else:
        raise BenchmarkError("provide --manifest, or both --audio and --language")
    return [_read_sample(path, language) for path, language in descriptors]


def _multipart_body(sample: Sample, mode: str) -> tuple[bytes, str]:
    boundary = f"asr-benchmark-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    def add_text(name: str, value: str) -> None:
        chunks.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            )
        )

    add_text("language", sample.language)
    add_text("mode", mode)
    chunks.extend(
        (
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="audio"; filename="sample.wav"\r\n',
            b"Content-Type: audio/wav\r\n\r\n",
            sample.audio_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        )
    )
    return b"".join(chunks), boundary


def _transcribe(
    *,
    endpoint: str,
    sample: Sample,
    mode: str,
    decoder: str,
    timeout_seconds: float,
    api_key: str | None,
) -> Measurement:
    body, boundary = _multipart_body(sample, mode)
    headers = {
        "Accept": "application/json",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "User-Agent": "indicconformer-realtime-asr-benchmark/1",
    }
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers=headers,
    )
    started = time.perf_counter_ns()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_bytes = response.read()
    except urllib.error.HTTPError as exc:
        # Never read/log the response body: an error may contain a transcript.
        raise BenchmarkError(f"transcription returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise BenchmarkError("transcription endpoint unavailable") from exc
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000

    try:
        payload = json.loads(response_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BenchmarkError("transcription response was not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise BenchmarkError("transcription response must be a JSON object")
    # Validate metadata without retaining or reporting the transcript itself.
    if not isinstance(payload.get("text"), str):
        raise BenchmarkError("transcription response has no string text field")
    if payload.get("language") != sample.language:
        raise BenchmarkError("transcription response language does not match the request")
    if payload.get("mode") != mode or payload.get("decoder") != decoder:
        raise BenchmarkError("transcription response does not match mode-derived decoder")
    server_duration = payload.get("audio_duration_ms")
    inference_ms = payload.get("inference_ms")
    if not isinstance(server_duration, int) or server_duration <= 0:
        raise BenchmarkError("transcription response has invalid audio_duration_ms")
    if abs(server_duration - sample.audio_duration_ms) > max(2, sample.audio_duration_ms * 0.001):
        raise BenchmarkError("server audio_duration_ms does not match the WAV duration")
    if (
        not isinstance(inference_ms, (int, float))
        or isinstance(inference_ms, bool)
        or inference_ms < 0
    ):
        raise BenchmarkError("transcription response has invalid inference_ms")
    return Measurement(latency_ms, float(inference_ms), server_duration)


def _wait_until_ready(base_url: str, timeout_seconds: float) -> None:
    endpoint = urllib.parse.urljoin(base_url.rstrip("/") + "/", "health/ready")
    deadline = time.monotonic() + timeout_seconds
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(endpoint, timeout=min(5.0, timeout_seconds)) as response:
                payload = json.load(response)
            if (
                response.status == 200
                and isinstance(payload, dict)
                and payload.get("status") == "ready"
                and payload.get("checks", {}).get("engine") == "ready"
                and payload.get("checks", {}).get("scheduler") == "ready"
            ):
                return
            last_error = "readiness response did not report ready engine and scheduler"
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = type(exc).__name__
        time.sleep(1.0)
    raise BenchmarkError(f"server did not become ready within {timeout_seconds:g}s ({last_error})")


def _gpu_metadata() -> dict[str, Any]:
    """Read observed GPU data; return explicit nulls when unavailable."""
    metadata: dict[str, Any] = {
        "source": "nvidia-smi",
        "available": False,
        "driver_version": None,
        "gpus": None,
        "error": None,
    }
    executable = shutil.which("nvidia-smi")
    if executable is None:
        metadata["error"] = "nvidia-smi not found"
        return metadata
    command = [
        executable,
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        metadata["error"] = "nvidia-smi query failed"
        return metadata
    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            metadata["error"] = "nvidia-smi returned an unexpected row"
            return metadata
        index, name, memory_mib, driver = fields
        try:
            gpus.append({"index": int(index), "name": name, "memory_total_mib": int(memory_mib)})
        except ValueError:
            metadata["error"] = "nvidia-smi returned invalid numeric data"
            return metadata
        metadata["driver_version"] = driver
    if not gpus:
        metadata["error"] = "nvidia-smi reported no GPUs"
        return metadata
    metadata.update(available=True, gpus=gpus, error=None)
    return metadata


def _run_cell(
    *,
    endpoint: str,
    samples: Sequence[Sample],
    concurrency: int,
    duration_seconds: float,
    mode: str,
    decoder: str,
    warmup_requests: int,
    timeout_seconds: float,
    api_key: str | None,
) -> dict[str, Any]:
    if warmup_requests:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(concurrency, warmup_requests)
        ) as pool:
            futures = [
                pool.submit(
                    _transcribe,
                    endpoint=endpoint,
                    sample=samples[index % len(samples)],
                    mode=mode,
                    decoder=decoder,
                    timeout_seconds=timeout_seconds,
                    api_key=api_key,
                )
                for index in range(warmup_requests)
            ]
            for future in futures:
                future.result()

    stop_at = time.monotonic() + duration_seconds
    sample_counter = itertools.count()
    counter_lock = threading.Lock()

    def worker() -> list[Measurement]:
        measurements: list[Measurement] = []
        while time.monotonic() < stop_at:
            with counter_lock:
                sample = samples[next(sample_counter) % len(samples)]
            measurements.append(
                _transcribe(
                    endpoint=endpoint,
                    sample=sample,
                    mode=mode,
                    decoder=decoder,
                    timeout_seconds=timeout_seconds,
                    api_key=api_key,
                )
            )
        return measurements

    measured_at = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        batches = [pool.submit(worker) for _ in range(concurrency)]
        measurements = [item for batch in batches for item in batch.result()]
    elapsed_seconds = time.perf_counter() - measured_at
    if not measurements:
        raise BenchmarkError("benchmark cell completed without a measurement")

    latencies = [item.latency_ms for item in measurements]
    inference = [item.inference_ms for item in measurements]
    end_to_end_rtf = [item.end_to_end_rtf for item in measurements]
    inference_rtf = [item.inference_rtf for item in measurements]
    audio_seconds = sum(item.audio_duration_ms for item in measurements) / 1000.0
    request_count = len(measurements)
    return {
        "mode": mode,
        "decoder": decoder,
        "concurrency": concurrency,
        "requested_duration_seconds": duration_seconds,
        "elapsed_seconds": elapsed_seconds,
        "request_count": request_count,
        "audio_seconds": audio_seconds,
        "latency_ms": _distribution(latencies),
        "inference_ms": _distribution(inference),
        "end_to_end_rtf": _distribution(end_to_end_rtf),
        "inference_rtf": _distribution(inference_rtf),
        "throughput": {
            "requests_per_second": request_count / elapsed_seconds,
            "audio_seconds_per_second": audio_seconds / elapsed_seconds,
            "realtime_capacity": audio_seconds / elapsed_seconds,
        },
    }


def _threshold_failures(cell: dict[str, Any], args: argparse.Namespace) -> list[str]:
    label = f"mode={cell['mode']},decoder={cell['decoder']},concurrency={cell['concurrency']}"
    checks: tuple[tuple[str, float, float | None, Callable[[float, float], bool], str], ...] = (
        (
            "max_p95_latency_ms",
            cell["latency_ms"]["p95"],
            args.max_p95_latency_ms,
            lambda a, n: a <= n,
            "<=",
        ),
        (
            "max_mean_end_to_end_rtf",
            cell["end_to_end_rtf"]["mean"],
            args.max_mean_rtf,
            lambda a, n: a <= n,
            "<=",
        ),
        (
            "min_throughput_rps",
            cell["throughput"]["requests_per_second"],
            args.min_throughput_rps,
            lambda a, n: a >= n,
            ">=",
        ),
    )
    failures: list[str] = []
    for name, actual, limit, predicate, operator in checks:
        if limit is not None and not predicate(actual, limit):
            failures.append(f"{label}: {name} requires {operator} {limit:g}, observed {actual:.6g}")
    return failures


def _self_check() -> int:
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert _distribution([2.0, 4.0])["mean"] == 3.0
    assert _gpu_metadata()["source"] == "nvidia-smi"
    print(json.dumps({"self_check": "ok", "schema_version": SCHEMA_VERSION}, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="ASR API base URL")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=pathlib.Path, help="JSON workload manifest")
    source.add_argument("--audio", type=pathlib.Path, help="single mono PCM16 16 kHz WAV")
    parser.add_argument("--language", choices=LANGUAGES, help="language for --audio")
    parser.add_argument(
        "--api-key-file",
        type=pathlib.Path,
        help="regular file containing the bearer key; the value is never logged",
    )
    parser.add_argument(
        "--concurrency",
        action="append",
        default=[],
        metavar="N[,N...]",
        help="concurrency matrix",
    )
    parser.add_argument(
        "--duration-seconds",
        action="append",
        default=[],
        metavar="S[,S...]",
        help="per-cell duration matrix",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        metavar="MODE[,MODE...]",
        help="mode matrix",
    )
    parser.add_argument(
        "--decoder",
        action="append",
        default=[],
        metavar="DECODER[,DECODER...]",
        help="decoder matrix",
    )
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--request-timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--ready-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--max-p95-latency-ms",
        type=float,
        default=None,
        help="fail if any cell exceeds this p95",
    )
    parser.add_argument(
        "--max-mean-rtf",
        type=float,
        default=None,
        help="fail if any cell exceeds this mean end-to-end RTF",
    )
    parser.add_argument(
        "--min-throughput-rps",
        type=float,
        default=None,
        help="fail if any cell is below this throughput",
    )
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="exercise offline calculations and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.self_check:
        return _self_check()
    try:
        concurrencies = _csv_values(args.concurrency or ["1"], cast=int)
        durations = _csv_values(args.duration_seconds or ["30"], cast=float)
        modes = _csv_values(args.mode or ["hybrid"], cast=str)
        decoders = _csv_values(args.decoder or list(DECODERS), cast=str)
        if any(value <= 0 for value in concurrencies):
            raise BenchmarkError("concurrency values must be positive integers")
        if any(value <= 0 for value in durations):
            raise BenchmarkError("duration values must be positive numbers")
        if any(value not in MODES for value in modes):
            raise BenchmarkError(f"mode must be one of {', '.join(MODES)}")
        if any(value not in DECODERS for value in decoders):
            raise BenchmarkError(f"decoder must be one of {', '.join(DECODERS)}")
        if args.warmup_requests < 0:
            raise BenchmarkError("warmup requests cannot be negative")
        if args.request_timeout_seconds <= 0 or args.ready_timeout_seconds <= 0:
            raise BenchmarkError("timeouts must be positive")

        api_key = _read_api_key(args.api_key_file)
        samples = _load_samples(args)
        base_url = args.base_url.rstrip("/") + "/"
        endpoint = urllib.parse.urljoin(base_url, "v1/transcribe")
        _wait_until_ready(base_url, args.ready_timeout_seconds)

        cells: list[dict[str, Any]] = []
        failures: list[str] = []
        for concurrency, duration, mode in itertools.product(concurrencies, durations, modes):
            decoder = FINAL_DECODER_BY_MODE[mode]
            if decoder not in decoders:
                continue
            print(
                f"benchmarking mode={mode} decoder={decoder} "
                f"concurrency={concurrency} duration={duration:g}s",
                file=sys.stderr,
            )
            cell = _run_cell(
                endpoint=endpoint,
                samples=samples,
                concurrency=concurrency,
                duration_seconds=duration,
                mode=mode,
                decoder=decoder,
                warmup_requests=args.warmup_requests,
                timeout_seconds=args.request_timeout_seconds,
                api_key=api_key,
            )
            cells.append(cell)
            failures.extend(_threshold_failures(cell, args))

        if not cells:
            raise BenchmarkError("mode and decoder selections have no valid combinations")
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
            "target": {"base_url": base_url, "endpoint": "/v1/transcribe"},
            "workload": {
                "sample_count": len(samples),
                "languages": sorted({sample.language for sample in samples}),
                "audio_seconds": (sum(sample.audio_duration_ms for sample in samples) / 1000.0),
                "concurrency_matrix": concurrencies,
                "duration_seconds_matrix": durations,
                "mode_matrix": modes,
                "decoder_matrix": sorted({cell["decoder"] for cell in cells}),
                "warmup_requests_per_cell": args.warmup_requests,
            },
            "hardware": {"gpu": _gpu_metadata()},
            "thresholds": {
                "max_p95_latency_ms": args.max_p95_latency_ms,
                "max_mean_end_to_end_rtf": args.max_mean_rtf,
                "min_throughput_requests_per_second": args.min_throughput_rps,
            },
            "fields": {
                "latency_ms": "client wall time from request start through complete response body",
                "inference_ms": "server-reported model inference time",
                "end_to_end_rtf": "latency_ms / audio_duration_ms; lower is better",
                "inference_rtf": "inference_ms / audio_duration_ms; lower is better",
                "throughput.requests_per_second": "completed requests / measured wall time",
                "throughput.audio_seconds_per_second": (
                    "completed audio seconds / measured wall time"
                ),
                "throughput.realtime_capacity": (
                    "completed audio seconds / wall time; higher is better"
                ),
            },
            "cells": cells,
            "failures": failures,
            "passed": not failures,
        }
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        for failure in failures:
            print(f"THRESHOLD FAILURE: {failure}", file=sys.stderr)
        return 2 if failures else 0
    except (BenchmarkError, FileNotFoundError, PermissionError, ValueError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
