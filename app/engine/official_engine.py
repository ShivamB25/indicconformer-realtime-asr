"""Pinned, local-only wrapper around AI4Bharat's official Transformers model.

The gated model and its trusted custom Python code must be acquired out of band.
This module never downloads model assets and does not import Torch or Transformers
until application startup.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import anyio

from app.core.types import SUPPORTED_LANGUAGES
from app.engine.base import (
    BaseEngine,
    EngineState,
    ProgressCallback,
    TranscriptionRequest,
    TranscriptionResult,
    normalize_pcm16_audio,
)

_REQUIRED_ONNX_SESSIONS = frozenset(
    {
        "encoder",
        "ctc_decoder",
        "joint_enc",
        "joint_pred",
        "joint_pre_net",
        *(f"joint_post_net_{language}" for language in SUPPORTED_LANGUAGES),
    }
)
_RNNT_SESSION_LAYOUTS = (
    frozenset({"rnnt_decoder"}),
    frozenset({"rnnt_decoder_embed", "rnnt_decoder_rnn"}),
)


def _require_strict_cuda_sessions(model: Any) -> None:
    """Disable runtime fallback and prove every custom-model session is CUDA-primary."""

    custom_models = getattr(model, "models", None)
    if not isinstance(custom_models, Mapping):
        raise RuntimeError("official model does not expose its ONNX session registry")

    session_names: set[str] = set()
    for raw_name, session in custom_models.items():
        get_providers = getattr(session, "get_providers", None)
        if not callable(get_providers):
            continue
        name = str(raw_name)
        session_names.add(name)
        try:
            providers = tuple(get_providers())
        except Exception as exc:
            raise RuntimeError(f"cannot inspect ONNX providers for {name}") from exc
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                f"strict CUDA requires {name} to use CUDAExecutionProvider first; "
                f"configured providers are {providers!r}"
            )
        disable_fallback = getattr(session, "disable_fallback", None)
        if not callable(disable_fallback):
            raise RuntimeError(f"strict CUDA cannot disable provider fallback for {name}")
        try:
            disable_fallback()
        except Exception as exc:
            raise RuntimeError(f"cannot disable provider fallback for {name}") from exc

    missing = _REQUIRED_ONNX_SESSIONS.difference(session_names)
    if missing:
        raise RuntimeError(
            "official model is missing required ONNX sessions: " + ", ".join(sorted(missing))
        )
    if not any(layout <= session_names for layout in _RNNT_SESSION_LAYOUTS):
        raise RuntimeError("official model is missing its RNNT decoder ONNX sessions")


class OfficialIndicConformerEngine(BaseEngine):
    """Official custom-code model, loaded from a verified local snapshot."""

    def __init__(
        self,
        model_dir: Path,
        manifest_path: Path,
        repo_id: str,
        revision: str,
        *,
        require_cuda: bool = True,
    ) -> None:
        super().__init__()
        self._model_dir = model_dir
        self._manifest_path = manifest_path
        self._repo_id = repo_id
        self._revision = revision
        self._require_cuda = require_cuda
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device = "cpu"

    @property
    def name(self) -> str:
        return "official"

    async def startup(self, progress: ProgressCallback | None = None) -> None:
        self._set_readiness(EngineState.STARTING, "verifying_artifacts")
        if progress is not None:
            progress("verifying_artifacts")
        try:
            await anyio.to_thread.run_sync(self._load_sync)
        except BaseException as exc:
            self._set_readiness(EngineState.FAILED, "startup_failed", type(exc).__name__)
            raise
        self._set_readiness(EngineState.READY, "ready")
        if progress is not None:
            progress("official_ready")

    def _load_sync(self) -> None:
        self._verify_snapshot()

        # Heavy optional dependencies are deliberately startup-only.
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoConfig, AutoModel  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "the official engine requires the production torch/transformers extras"
            ) from exc

        if self._require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but is not available")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        config = AutoConfig.from_pretrained(
            str(self._model_dir),
            local_files_only=True,
            trust_remote_code=True,
        )
        config.ts_folder = str(self._model_dir)
        model = AutoModel.from_config(config, trust_remote_code=True)
        if self._require_cuda:
            _require_strict_cuda_sessions(model)
        model.to(self._device)
        model.eval()
        self._torch = torch
        self._model = model

    def _verify_snapshot(self) -> None:
        model_dir = self._model_dir.resolve(strict=True)
        configured_manifest = self._manifest_path.resolve(strict=True)
        expected_manifest = (model_dir / "model-manifest.json").resolve(strict=True)
        if configured_manifest != expected_manifest:
            raise RuntimeError("model_manifest must name model_dir/model-manifest.json")

        # Reuse the downloader's canonical streaming verifier, including the
        # completion-marker hash and exact repository/revision checks.
        from scripts.download_model import verify_model

        verify_model(model_dir, self._repo_id, self._revision)

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        if not self.readiness.ready or self._model is None or self._torch is None:
            raise RuntimeError("official engine is not ready")

        audio = normalize_pcm16_audio(request.audio)
        waveform = self._torch.from_numpy(audio).unsqueeze(0).to(self._device)
        if self._device == "cuda":
            self._torch.cuda.synchronize()
        started = perf_counter()
        with self._torch.inference_mode():
            output = self._model(waveform, request.language, request.decoder)
        if self._device == "cuda":
            self._torch.cuda.synchronize()
        inference_ms = (perf_counter() - started) * 1000

        if isinstance(output, str):
            text = output
        elif isinstance(output, (list, tuple)) and len(output) == 1 and isinstance(output[0], str):
            text = output[0]
        else:
            raise RuntimeError("official model returned an unsupported transcription value")

        return TranscriptionResult(
            text=text,
            language=request.language,
            decoder=request.decoder,
            audio_duration_ms=round(audio.size * 1000 / request.sample_rate),
            inference_ms=inference_ms,
        )

    async def shutdown(self) -> None:
        model, torch = self._model, self._torch
        self._model = None
        self._torch = None
        if model is not None:
            del model
        if torch is not None and self._device == "cuda":
            torch.cuda.empty_cache()
        self._set_readiness(EngineState.STOPPED, "stopped")
