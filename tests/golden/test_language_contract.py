"""Golden metadata contracts; these do not claim model accuracy or GPU output."""

import numpy as np
import pytest

from app.core.types import SUPPORTED_LANGUAGE_CODES, ProcessingMode
from app.engine.base import TranscriptionRequest
from app.engine.mock import MockEngine
from tests.support.golden import load_golden


def test_golden_language_set_matches_runtime_contract() -> None:
    golden = load_golden("languages.json")
    assert golden["count"] == 22
    assert set(golden["languages"]) == SUPPORTED_LANGUAGE_CODES
    assert set(golden["final_decoder_by_mode"]) == {mode.value for mode in ProcessingMode}


@pytest.mark.asyncio
@pytest.mark.parametrize("language", sorted(SUPPORTED_LANGUAGE_CODES))
async def test_mock_contract_covers_every_supported_language(language: str) -> None:
    engine = MockEngine()
    await engine.startup()
    try:
        result = engine.transcribe(
            TranscriptionRequest(
                audio=np.zeros(320, dtype=np.float32),
                sample_rate=16_000,
                language=language,
                decoder="ctc",
            )
        )
    finally:
        await engine.shutdown()
    assert result.language == language
    assert result.decoder == "ctc"
    assert result.audio_duration_ms == 20
