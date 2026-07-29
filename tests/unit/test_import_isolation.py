"""The service must import and serve without any GPU or model runtime present.

Each case runs in a fresh interpreter, because import side effects cannot be
undone inside the current process once pytest has already imported the package.
The probe reports which heavy modules ended up in ``sys.modules``; a positive
control asserts the probe itself still detects modules that are genuinely
imported, so a silently broken probe cannot make these tests vacuous.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Runtimes that must never be pulled in by importing or serving the application.
# The official and ORT engines import them inside startup, never at module scope.
HEAVY_MODULES = (
    "torch",
    "torchaudio",
    "transformers",
    "onnx",
    "onnxruntime",
    "huggingface_hub",
)

PROBE_PREAMBLE = """
import json, sys

HEAVY = {heavy!r}

def report():
    return sorted(name for name in HEAVY if name in sys.modules)
"""

CPU_ONLY_ENVIRONMENT = {
    "ASR_ENGINE": "mock",
    "ASR_ENVIRONMENT": "test",
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def run_probe(body: str) -> dict[str, object]:
    """Run ``body`` in a clean interpreter and return its JSON report."""

    program = PROBE_PREAMBLE.format(heavy=HEAVY_MODULES) + body
    environment = {**os.environ, **CPU_ONLY_ENVIRONMENT}
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert isinstance(payload, dict)
    return payload


class TestProbeItself:
    def test_the_probe_detects_a_module_that_is_actually_imported(self) -> None:
        report = run_probe(
            """
import onnx  # noqa: F401
print(json.dumps({"imported": report()}))
"""
        )

        assert report["imported"] == ["onnx"]

    def test_the_probe_reports_nothing_for_a_bare_interpreter(self) -> None:
        report = run_probe('print(json.dumps({"imported": report()}))')

        assert report["imported"] == []


class TestApplicationImport:
    def test_importing_the_application_module_pulls_in_no_heavy_runtime(self) -> None:
        report = run_probe(
            """
import app.main
print(json.dumps({"imported": report(), "app": type(app.main.app).__name__}))
"""
        )

        assert report["imported"] == []
        assert report["app"] == "FastAPI"

    def test_the_application_factory_pulls_in_no_heavy_runtime(self) -> None:
        report = run_probe(
            """
from app.core.config import Settings
from app.main import create_app

application = create_app(Settings(environment="test"))
print(json.dumps({"imported": report(), "routes": len(application.routes)}))
"""
        )

        assert report["imported"] == []
        assert isinstance(report["routes"], int)
        assert report["routes"] > 0

    def test_a_full_startup_and_shutdown_pulls_in_no_heavy_runtime(self) -> None:
        report = run_probe(
            """
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.engine.mock import MockEngine
from app.main import create_app

application = create_app(Settings(environment="test"), engine=MockEngine())
with TestClient(application) as client:
    ready = client.get("/health/ready")
print(json.dumps({"imported": report(), "ready": ready.status_code}))
"""
        )

        assert report["imported"] == []
        assert report["ready"] == 200

    def test_serving_a_transcription_pulls_in_no_heavy_runtime(self) -> None:
        report = run_probe(
            """
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.engine.mock import MockEngine
from app.main import create_app

application = create_app(Settings(environment="test"), engine=MockEngine())
with TestClient(application) as client:
    response = client.post(
        "/v1/transcribe",
        data={"language": "hi"},
        files={"audio": ("a.pcm", b"\\x00\\x01" * 320, "application/octet-stream")},
    )
print(json.dumps({"imported": report(), "status": response.status_code}))
"""
        )

        assert report["imported"] == []
        assert report["status"] == 200


class TestEngineModulesAreLazy:
    @pytest.mark.parametrize(
        "module",
        [
            "app.engine.official_engine",
            "app.engine.ort_engine",
            "app.engine.scheduler",
            "app.engine.manifest",
            "app.api.websocket",
            "app.core.lifespan",
        ],
    )
    def test_importing_an_engine_module_pulls_in_no_heavy_runtime(self, module: str) -> None:
        report = run_probe(
            f"""
import importlib

importlib.import_module({module!r})
print(json.dumps({{"imported": report()}}))
"""
        )

        assert report["imported"] == []

    def test_the_downloader_does_not_import_the_hub_client_at_module_scope(self) -> None:
        report = run_probe(
            """
import sys
sys.path.insert(0, "scripts")
import download_model  # noqa: F401
print(json.dumps({"imported": report(), "assets": len(download_model.REQUIRED_ASSETS)}))
"""
        )

        assert report["imported"] == []
        assert isinstance(report["assets"], int)
        assert report["assets"] > 0


class TestNoCudaRequirement:
    def test_the_mock_engine_path_never_asks_for_cuda(self) -> None:
        """``ASR_REQUIRE_CUDA`` is irrelevant to a mock-engine service."""

        report = run_probe(
            """
import asyncio

from app.core.config import Settings
from app.engine.mock import MockEngine

settings = Settings(environment="test", require_cuda=True)
engine = MockEngine()
asyncio.run(engine.startup())
print(json.dumps({"imported": report(), "ready": engine.readiness.ready}))
"""
        )

        assert report["imported"] == []
        assert report["ready"] is True
