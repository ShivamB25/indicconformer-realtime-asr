"""Golden text for the deterministic mock engine.

These strings come from a formatted echo of request metadata, not from a model.
They are pinned so a change to the mock's output shape is a deliberate edit:
the whole suite reads MockEngine text as its stand-in for a transcript.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.engine.mock import MockEngine
from tests.support.engines import make_request
from tests.support.golden import load_golden

GOLDEN = load_golden("mock_transcript.json")
EXAMPLES: list[dict[str, Any]] = GOLDEN["examples"]


@pytest.fixture
async def engine() -> MockEngine:
    instance = MockEngine()
    await instance.startup()
    return instance


@pytest.mark.parametrize(
    "example", EXAMPLES, ids=[f"{item['language']}-{item['duration_ms']}ms" for item in EXAMPLES]
)
async def test_recorded_examples_are_reproduced_exactly(
    example: dict[str, Any], engine: MockEngine
) -> None:
    result = engine.transcribe(
        make_request(
            duration_ms=example["duration_ms"],
            language=example["language"],
            decoder=example["decoder"],
        )
    )
    assert result.text == example["text"]
    assert result.language == example["language"]
    assert result.decoder == example["decoder"]
    assert result.audio_duration_ms == example["duration_ms"]


@pytest.mark.parametrize("example", EXAMPLES, ids=[item["language"] for item in EXAMPLES])
def test_recorded_examples_agree_with_the_recorded_template(example: dict[str, Any]) -> None:
    assert (
        GOLDEN["template"].format(
            **{
                "language": example["language"],
                "decoder": example["decoder"],
                "duration_ms": example["duration_ms"],
            }
        )
        == example["text"]
    )


async def test_the_template_still_describes_live_output(engine: MockEngine) -> None:
    result = engine.transcribe(make_request(duration_ms=140, language="kn", decoder="rnnt"))
    assert result.text == GOLDEN["template"].format(language="kn", decoder="rnnt", duration_ms=140)


async def test_the_text_carries_no_audio_dependent_content(engine: MockEngine) -> None:
    quiet = engine.transcribe(make_request(duration_ms=200, level=0.0))
    loud = engine.transcribe(make_request(duration_ms=200, level=0.75))
    assert quiet.text == loud.text
