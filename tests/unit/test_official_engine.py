"""Local-only loading contracts for the official custom model wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from app.engine.official_engine import OfficialIndicConformerEngine


def test_loader_constructs_custom_model_from_verified_local_config(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    config = SimpleNamespace()

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> object:
            calls["config"] = (path, kwargs)
            return config

    class FakeModel:
        def to(self, device: str) -> None:
            calls["device"] = device

        def eval(self) -> None:
            calls["eval"] = True

    class FakeAutoModel:
        @staticmethod
        def from_config(value: object, **kwargs: object) -> FakeModel:
            calls["model"] = (value, kwargs)
            return FakeModel()

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    transformers = ModuleType("transformers")
    transformers.AutoConfig = FakeAutoConfig  # type: ignore[attr-defined]
    transformers.AutoModel = FakeAutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(OfficialIndicConformerEngine, "_verify_snapshot", lambda self: None)

    engine = OfficialIndicConformerEngine(
        tmp_path,
        tmp_path / "model-manifest.json",
        "owner/model",
        "a" * 40,
        require_cuda=False,
    )
    engine._load_sync()

    assert calls["config"] == (
        str(tmp_path),
        {"local_files_only": True, "trust_remote_code": True},
    )
    assert config.ts_folder == str(tmp_path)
    assert calls["model"] == (config, {"trust_remote_code": True})
    assert calls["device"] == "cpu"
    assert calls["eval"] is True
