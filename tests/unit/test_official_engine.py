"""Local-only loading contracts for the official custom model wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from app.core.config import Settings
from app.core.lifespan import _build_engine
from app.core.types import SUPPORTED_LANGUAGES, EngineKind
from app.engine.base import EngineState, TranscriptionRequest, normalize_pcm16_audio
from app.engine.mock import MockEngine
from app.engine.official_engine import OfficialIndicConformerEngine


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda_available: bool,
    model: object,
    calls: dict[str, object],
) -> SimpleNamespace:
    config = SimpleNamespace()

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> object:
            calls["config"] = (path, kwargs)
            return config

    class FakeAutoModel:
        @staticmethod
        def from_config(value: object, **kwargs: object) -> object:
            calls["model"] = (value, kwargs)
            return model

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: cuda_available)  # type: ignore[attr-defined]
    transformers = ModuleType("transformers")
    transformers.AutoConfig = FakeAutoConfig  # type: ignore[attr-defined]
    transformers.AutoModel = FakeAutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(OfficialIndicConformerEngine, "_verify_snapshot", lambda self: None)
    return config


def _engine(tmp_path: Path, *, require_cuda: bool) -> OfficialIndicConformerEngine:
    return OfficialIndicConformerEngine(
        tmp_path,
        tmp_path / "model-manifest.json",
        "owner/model",
        "a" * 40,
        require_cuda=require_cuda,
    )


def test_loader_constructs_custom_model_from_verified_local_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    class FakeModel:
        def to(self, device: str) -> None:
            calls["device"] = device

        def eval(self) -> None:
            calls["eval"] = True

    model = FakeModel()
    config = _install_runtime(
        monkeypatch,
        cuda_available=False,
        model=model,
        calls=calls,
    )

    engine = _engine(tmp_path, require_cuda=False)
    engine._load_sync()

    assert calls["config"] == (
        str(tmp_path),
        {"local_files_only": True, "trust_remote_code": True},
    )
    assert config.ts_folder == str(tmp_path)
    assert calls["model"] == (config, {"trust_remote_code": True})
    assert calls["device"] == "cpu"
    assert calls["eval"] is True


def test_engine_builder_selects_only_mock_or_official(tmp_path: Path) -> None:
    assert isinstance(_build_engine(Settings(environment="test")), MockEngine)

    settings = Settings(
        environment="test",
        engine=EngineKind.OFFICIAL,
        model_dir=tmp_path,
        model_manifest=tmp_path / "model-manifest.json",
        model_repo_id="owner/model",
        model_revision="a" * 40,
        require_cuda=False,
    )
    assert isinstance(_build_engine(settings), OfficialIndicConformerEngine)


def test_pcm16_normalization_clips_to_the_representable_float_range() -> None:
    source = np.array([-4.0, -1.0, -0.25, 0.0, 1.0, 3.0], dtype=np.float64)

    normalized = normalize_pcm16_audio(source)

    assert normalized.dtype == np.float32
    assert normalized.flags.c_contiguous
    np.testing.assert_array_equal(
        normalized,
        np.array([-1.0, -1.0, -0.25, 0.0, 32_767 / 32_768, 32_767 / 32_768], np.float32),
    )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_pcm16_normalization_rejects_nonfinite_audio(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        normalize_pcm16_audio(np.array([0.0, invalid], dtype=np.float32))


class _Session:
    def __init__(self, providers: list[str]) -> None:
        self._providers = providers
        self.fallback_disabled = False

    def get_providers(self) -> list[str]:
        return list(self._providers)

    def disable_fallback(self) -> None:
        self.fallback_disabled = True


def _complete_sessions(providers: list[str]) -> dict[str, object]:
    names = {
        "encoder",
        "ctc_decoder",
        "rnnt_decoder",
        "joint_enc",
        "joint_pred",
        "joint_pre_net",
        *(f"joint_post_net_{language}" for language in SUPPORTED_LANGUAGES),
    }
    return {name: _Session(providers) for name in names}


def test_strict_cuda_fails_when_custom_model_sessions_fall_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    class FakeModel:
        def __init__(self) -> None:
            self.models = _complete_sessions(["CPUExecutionProvider"])

        def to(self, device: str) -> None:
            calls["device"] = device

        def eval(self) -> None:
            calls["eval"] = True

    _install_runtime(
        monkeypatch,
        cuda_available=True,
        model=FakeModel(),
        calls=calls,
    )
    engine = _engine(tmp_path, require_cuda=True)

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider first"):
        engine._load_sync()

    assert "device" not in calls
    assert engine._model is None


def test_strict_cuda_disables_fallback_on_every_custom_onnx_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    class FakeModel:
        def __init__(self) -> None:
            self.models = _complete_sessions(["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.models["preprocessor"] = object()

        def to(self, device: str) -> None:
            calls["device"] = device

        def eval(self) -> None:
            calls["eval"] = True

    model = FakeModel()
    _install_runtime(
        monkeypatch,
        cuda_available=True,
        model=model,
        calls=calls,
    )

    _engine(tmp_path, require_cuda=True)._load_sync()

    sessions = [session for session in model.models.values() if isinstance(session, _Session)]
    assert sessions
    assert all(session.fallback_disabled for session in sessions)
    assert calls["device"] == "cuda"


def test_transcription_normalizes_audio_before_torch_conversion(tmp_path: Path) -> None:
    captured: dict[str, np.ndarray] = {}

    class FakeTensor:
        def unsqueeze(self, dimension: int) -> FakeTensor:
            assert dimension == 0
            return self

        def to(self, device: str) -> FakeTensor:
            assert device == "cpu"
            return self

    class InferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    class FakeTorch:
        cuda = SimpleNamespace(synchronize=lambda: None)

        @staticmethod
        def from_numpy(audio: np.ndarray) -> FakeTensor:
            captured["audio"] = audio
            return FakeTensor()

        @staticmethod
        def inference_mode() -> InferenceMode:
            return InferenceMode()

    class FakeModel:
        def __call__(self, waveform: object, language: str, decoder: str) -> str:
            return "text"

    engine = _engine(tmp_path, require_cuda=False)
    engine._torch = FakeTorch()
    engine._model = FakeModel()
    engine._set_readiness(EngineState.READY, "ready")

    engine.transcribe(
        TranscriptionRequest(
            audio=np.array([-2.0, 0.0, 2.0], dtype=np.float64),
            sample_rate=16_000,
            language="hi",
            decoder="ctc",
        )
    )

    np.testing.assert_array_equal(
        captured["audio"],
        np.array([-1.0, 0.0, 32_767 / 32_768], dtype=np.float32),
    )
