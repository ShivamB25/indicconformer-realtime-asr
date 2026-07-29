"""Pinned, local-only wrapper around AI4Bharat's official Transformers model.

The gated model and its trusted custom Python code must be acquired out of band.
This module never downloads model assets and does not import Torch or Transformers
until application startup.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import anyio
import numpy as np

from app.engine.base import (
    BaseEngine,
    EngineState,
    ProgressCallback,
    TranscriptionRequest,
    TranscriptionResult,
)


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
            from transformers import AutoModel  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "the official engine requires the production torch/transformers extras"
            ) from exc

        if self._require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but is not available")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModel.from_pretrained(
            str(self._model_dir),
            local_files_only=True,
            trust_remote_code=True,
        )
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

        audio = np.ascontiguousarray(request.audio, dtype=np.float32)
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
