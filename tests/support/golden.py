"""Loader for golden fixtures.

Golden files hold metadata and text expectations only: closed sets, JSON
payload shapes, and strings produced by deterministic non-model code. They
never contain audio, model weights, or claimed ASR output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden" / "data"


def golden_path(name: str) -> Path:
    path = GOLDEN_DIR / name
    if not path.is_file():
        raise AssertionError(f"missing golden fixture: {path}")
    return path


def load_golden(name: str) -> dict[str, Any]:
    """Read a golden JSON object, rejecting duplicate keys."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"duplicate key in {name}: {key}")
            result[key] = value
        return result

    text = golden_path(name).read_text(encoding="utf-8")
    value = json.loads(text, object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise AssertionError(f"golden fixture {name} must contain a JSON object")
    return cast("dict[str, Any]", value)


__all__ = ["GOLDEN_DIR", "golden_path", "load_golden"]
