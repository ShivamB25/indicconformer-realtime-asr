"""Explicit no-VAD streaming behavior."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.types import VADKind
from app.vad.base import VADCapacityError, VADClosedError
from app.vad.disabled import DisabledVADProvider
from app.vad.factory import build_vad_provider


@pytest.mark.asyncio
async def test_disabled_provider_marks_valid_frames_as_speech_until_manual_commit() -> None:
    provider = DisabledVADProvider(max_streams=1)
    await provider.startup()
    stream = provider.new_stream(16_000)

    assert provider.name == "disabled"
    assert provider.active_streams == 1
    assert await stream.score(bytes(640)) == 1.0
    with pytest.raises(VADCapacityError):
        provider.new_stream(16_000)

    stream.reset()
    stream.close()
    assert provider.active_streams == 0
    with pytest.raises(VADClosedError):
        await stream.score(bytes(640))
    await provider.close()


@pytest.mark.asyncio
async def test_factory_builds_disabled_provider_without_model_artifacts() -> None:
    settings = Settings(vad_provider=VADKind.DISABLED, vad_max_streams=1)
    provider = build_vad_provider(settings, metrics=None)

    await provider.startup()
    assert provider.name == "disabled"
    await provider.close()
