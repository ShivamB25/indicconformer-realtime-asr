"""Suite-wide isolation and shared fixtures.

Isolation rules enforced here for every test:

* ``ASR_*`` environment variables are cleared, so a developer shell can never
  change an assertion, and the settings cache is dropped around each test.
* the mock engine is selected and the Hugging Face/CUDA escape hatches are
  pinned to their offline values.
* outbound network calls raise immediately, so a test that grows an accidental
  dependency on the network fails loudly instead of hanging in CI.
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from pathlib import Path
from typing import Any

import pytest

from app.core.config import get_settings
from app.engine.mock import MockEngine
from app.engine.scheduler import InferenceScheduler, SchedulerConfig
from tests.support.engines import GatedMockEngine, RecordingMockEngine
from tests.support.model_snapshot import publish_snapshot

OFFLINE_ENVIRONMENT = {
    "ASR_ENGINE": "mock",
    "ASR_ENVIRONMENT": "test",
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


@pytest.fixture(autouse=True)
def isolated_settings_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give every test the same clean, offline, mock-engine environment."""

    for name in list(os.environ):
        if name.startswith("ASR_"):
            monkeypatch.delenv(name, raising=False)
    for name, value in OFFLINE_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def blocked_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast on any outbound connection attempt.

    In-process ASGI transports, thread portals, and asyncio self-pipes do not
    open outbound sockets, so nothing legitimate in this suite is affected.
    """

    def deny(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)


@pytest.fixture
async def mock_engine() -> AsyncIterator[MockEngine]:
    """A started MockEngine, shut down again at teardown."""

    engine = MockEngine()
    await engine.startup()
    yield engine
    await engine.shutdown()


@pytest.fixture
async def recording_engine() -> AsyncIterator[RecordingMockEngine]:
    engine = RecordingMockEngine()
    await engine.startup()
    yield engine
    await engine.shutdown()


@pytest.fixture
async def gated_engine() -> AsyncIterator[GatedMockEngine]:
    """A MockEngine whose blocking calls park until the test opens the gate."""

    engine = GatedMockEngine()
    await engine.startup()
    yield engine
    engine.open_gate()
    await engine.shutdown()


SchedulerFactory = Callable[..., Coroutine[Any, Any, InferenceScheduler]]


@pytest.fixture
async def scheduler_factory() -> AsyncIterator[SchedulerFactory]:
    """Create started schedulers and guarantee they are closed at teardown."""

    created: list[InferenceScheduler] = []

    async def create(engine: Any, config: SchedulerConfig | None = None) -> InferenceScheduler:
        scheduler = InferenceScheduler(engine, config)
        await scheduler.start()
        created.append(scheduler)
        return scheduler

    yield create
    for scheduler in reversed(created):
        await scheduler.close()


@pytest.fixture
def published_snapshot(tmp_path: Path) -> Path:
    """A verifiable local model snapshot of deterministic placeholder files."""

    root = tmp_path / "model"
    publish_snapshot(root)
    return root
