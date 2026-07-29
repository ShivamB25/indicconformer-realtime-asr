from __future__ import annotations

import asyncio
import importlib
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt

from app.core.types import LanguageCode
from app.engine.base import (
    BaseEngine,
    EngineState,
    ProgressCallback,
    TranscriptionRequest,
    TranscriptionResult,
)
from app.engine.ctc_decoder import CTCGreedyDecoder, LanguageVocabulary, load_language_vocabularies
from app.engine.errors import (
    EngineNotReadyError,
    ModelContractError,
    OrtEngineError,
    ProviderUnavailableError,
)
from app.engine.manifest import discover_assets, verify_manifest
from app.engine.rnnt_decoder import BoundedRNNTGreedyDecoder, RNNTDecodeLimits

SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(language.value for language in LanguageCode)


@dataclass(frozen=True, slots=True)
class GraphContract:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    input_types: tuple[str, ...]
    output_types: tuple[str, ...]


@dataclass(slots=True)
class _Buffer:
    audio: npt.NDArray[np.float32]
    lengths: npt.NDArray[np.int64]
    device_audio: Any | None = None
    device_lengths: Any | None = None


@dataclass(slots=True)
class _Encoded:
    value: Any
    lengths: npt.NDArray[np.int64]


class _BufferPool:
    """Reusable power-of-two host/device buffers; caller serializes access."""

    def __init__(self) -> None:
        self._buffers: dict[tuple[int, int, bool], _Buffer] = {}

    def acquire(
        self, items: Sequence[npt.NDArray[np.float32]], ort: ModuleType, cuda: bool, device: int
    ) -> _Buffer:
        longest = max(item.size for item in items)
        bucket = max(4096, 1 << max(0, longest - 1).bit_length())
        key = (len(items), bucket, cuda)
        result = self._buffers.get(key)
        if result is None:
            audio = np.zeros((len(items), bucket), np.float32)
            lengths = np.empty((len(items),), np.int64)
            result = _Buffer(audio, lengths)
            if cuda:
                result.device_audio = ort.OrtValue.ortvalue_from_shape_and_type(
                    audio.shape, np.float32, "cuda", device
                )
                result.device_lengths = ort.OrtValue.ortvalue_from_shape_and_type(
                    lengths.shape, np.int64, "cuda", device
                )
            self._buffers[key] = result
        result.audio.fill(0)
        for index, item in enumerate(items):
            result.audio[index, : item.size] = item
            result.lengths[index] = item.size
        if cuda:
            assert result.device_audio is not None
            assert result.device_lengths is not None
            result.device_audio.update_inplace(result.audio)
            result.device_lengths.update_inplace(result.lengths)
        return result

    def clear(self) -> None:
        self._buffers.clear()


def _ort_module() -> ModuleType:
    try:
        return importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise ProviderUnavailableError(
            "ONNX Runtime is not installed; install the pinned runtime or select MockEngine"
        ) from exc


def require_cuda_provider() -> None:
    """Exit unsuccessfully unless the CUDA provider is genuinely available."""
    ort = _ort_module()
    providers = tuple(ort.get_available_providers())
    if "CUDAExecutionProvider" not in providers or str(ort.get_device()).upper() != "GPU":
        raise ProviderUnavailableError(
            f"CUDAExecutionProvider required, but ORT reports device={ort.get_device()!r}, "
            f"providers={providers!r}; refusing CPU fallback"
        )


def _preload_cuda(ort: ModuleType) -> None:
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        preload()
        return
    try:  # ORT 1.20 has no preload_dlls; pinned torch loads CUDA/cuDNN first.
        importlib.import_module("torch")
    except ImportError as exc:
        raise ProviderUnavailableError(
            "ORT 1.20 GPU startup requires the pinned CUDA torch package for DLL preloading"
        ) from exc


class OrtIndicConformerEngine(BaseEngine):
    """Pinned local IndicConformer ONNX execution with fail-closed providers."""

    def __init__(
        self,
        model_dir: Path,
        manifest_path: Path | None = None,
        *,
        repo_id: str,
        revision: str,
        require_cuda: bool = True,
        allow_cpu: bool = False,
        require_complete: bool = True,
        device_id: int = 0,
        warmup_samples: int = 1600,
        rnnt_max_symbols_per_frame: int = 5,
        rnnt_max_total_symbols: int = 4096,
    ) -> None:
        super().__init__()
        if require_cuda == allow_cpu:
            raise ValueError("choose require_cuda=True or explicitly allow_cpu=True, never both")
        if device_id < 0 or warmup_samples <= 0:
            raise ValueError("invalid device_id or warmup_samples")
        self._root = Path(model_dir)
        self._manifest = (
            Path(manifest_path) if manifest_path else self._root / "model-manifest.json"
        )
        self._repo_id = repo_id
        self._revision = revision
        self._cuda = require_cuda
        self._complete = require_complete
        self._device = device_id
        self._warmup_samples = warmup_samples
        self._limits = RNNTDecodeLimits(rnnt_max_symbols_per_frame, rnnt_max_total_symbols)
        self._ort: ModuleType | None = None
        self._sessions: dict[str, Any] = {}
        self._graphs: dict[str, GraphContract] = {}
        self._vocabs: dict[str, LanguageVocabulary] = {}
        self._ctc: CTCGreedyDecoder | None = None
        self._rnnt: BoundedRNNTGreedyDecoder | None = None
        self._pool = _BufferPool()
        self._lock = threading.Lock()
        self._warmed: list[str] = []

    @property
    def name(self) -> str:
        return "ort-indicconformer"

    @property
    def warmup_progress(self) -> tuple[str, ...]:
        return tuple(self._warmed)

    async def startup(self, progress: ProgressCallback | None = None) -> None:
        if self.readiness.state is EngineState.READY:
            return
        if self.readiness.state is EngineState.STARTING:
            raise OrtEngineError("startup already in progress")
        self._set_readiness(EngineState.STARTING, "manifest")
        try:
            await asyncio.to_thread(self._startup, progress)
        except Exception as exc:
            self._set_readiness(EngineState.FAILED, "startup", str(exc))
            self._release()
            raise
        self._set_readiness(EngineState.READY, "ready")
        if progress:
            progress("ready")

    async def shutdown(self) -> None:
        await asyncio.to_thread(self._release)
        self._set_readiness(EngineState.STOPPED, "stopped")

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        return self.transcribe_batch((request,))[0]

    def transcribe_batch(
        self, requests: Sequence[TranscriptionRequest]
    ) -> list[TranscriptionResult]:
        if self.readiness.state is not EngineState.READY:
            raise EngineNotReadyError(
                f"engine state={self.readiness.state}, stage={self.readiness.stage}"
            )
        if not requests:
            return []
        with self._lock:
            return self._transcribe_batch(requests)

    def _startup(self, progress: ProgressCallback | None) -> None:
        manifest = verify_manifest(self._root, self._manifest, require_complete=self._complete)
        if manifest.repository != self._repo_id or manifest.revision != self._revision:
            raise ModelContractError(
                "verified manifest identity does not match the configured repository and revision"
            )
        assets = discover_assets(manifest, SUPPORTED_LANGUAGES)
        self._notify(progress, "manifest-verified")
        try:
            vocabs = load_language_vocabularies(assets.language_masks, SUPPORTED_LANGUAGES)
        except ValueError as exc:
            raise ModelContractError(f"language_masks.json mismatch: {exc}") from exc
        ort = _ort_module()
        if self._cuda:
            require_cuda_provider()
            _preload_cuda(ort)
            providers: list[Any] = [("CUDAExecutionProvider", {"device_id": self._device})]
        else:
            available = tuple(ort.get_available_providers())
            if "CPUExecutionProvider" not in available:
                raise ProviderUnavailableError(f"CPU provider unavailable: {available!r}")
            providers = ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = True
        options.enable_mem_pattern = True
        options.log_severity_level = 3
        paths: list[tuple[str, Path]] = [
            ("encoder", assets.encoder),
            ("ctc", assets.ctc_decoder),
            ("joint_enc", assets.joint_encoder),
            ("joint_pred", assets.joint_predictor),
            ("joint_pre", assets.joint_pre_net),
        ] + [(f"post:{lang}", assets.joint_post_nets[lang]) for lang in SUPPORTED_LANGUAGES]
        sessions: dict[str, Any] = {}
        graphs: dict[str, GraphContract] = {}
        for role, path in paths:
            try:
                session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            except Exception as exc:
                sessions.clear()
                raise ModelContractError(
                    f"cannot create {role} session from {path}: {exc}"
                ) from exc
            _check_provider(session, role, self._cuda)
            sessions[role] = session
            graphs[role] = _graph_contract(role, session)
            if role.startswith("post:"):
                self._notify(progress, f"session:{role[5:]}")
        self._ort, self._sessions, self._graphs, self._vocabs = ort, sessions, graphs, vocabs
        self._ctc = CTCGreedyDecoder(vocabs)
        self._rnnt = BoundedRNNTGreedyDecoder(vocabs, self._limits)
        self._notify(progress, "sessions-ready")
        self._warmup(progress)

    def _warmup(self, progress: ProgressCallback | None) -> None:
        self._warmed.clear()
        silence = np.zeros((self._warmup_samples,), np.float32)
        self._ctc_infer((silence,), (SUPPORTED_LANGUAGES[0],))
        for language in SUPPORTED_LANGUAGES:
            self._notify(progress, f"warmup:{language}:starting")
            self._rnnt_infer((silence,), language)
            self._warmed.append(language)
            self._notify(progress, f"warmup:{language}:ready")

    def _transcribe_batch(
        self, requests: Sequence[TranscriptionRequest]
    ) -> list[TranscriptionResult]:
        for request in requests:
            if request.language not in SUPPORTED_LANGUAGES:
                raise ValueError(f"unsupported language {request.language!r}")
        audio = [_audio(item.audio) for item in requests]
        results: list[TranscriptionResult | None] = [None] * len(requests)
        ctc_indices = [i for i, item in enumerate(requests) if item.decoder == "ctc"]
        if ctc_indices:
            started = time.perf_counter()
            texts = self._ctc_infer(
                tuple(audio[i] for i in ctc_indices),
                tuple(requests[i].language for i in ctc_indices),
            )
            elapsed = (time.perf_counter() - started) * 1000
            for index, text in zip(ctc_indices, texts, strict=True):
                results[index] = _result(requests[index], text, elapsed)
        for language in SUPPORTED_LANGUAGES:
            indices = [
                i
                for i, item in enumerate(requests)
                if item.decoder == "rnnt" and item.language == language
            ]
            if not indices:
                continue
            started = time.perf_counter()
            texts = self._rnnt_infer(tuple(audio[i] for i in indices), language)
            elapsed = (time.perf_counter() - started) * 1000
            for index, text in zip(indices, texts, strict=True):
                results[index] = _result(requests[index], text, elapsed)
        if any(item is None for item in results):
            raise AssertionError("batch result missing")
        return [item for item in results if item is not None]

    def _ctc_infer(
        self, audio: Sequence[npt.NDArray[np.float32]], languages: Sequence[str]
    ) -> list[str]:
        encoded = self._encode(audio)
        outputs = self._terminal("ctc", self._encoded_feeds("ctc", encoded))
        scores = _terminal_tensor("ctc", outputs)
        lengths = _length_output(outputs, len(audio))
        if self._ctc is None:
            raise EngineNotReadyError("CTC decoder unavailable")
        try:
            return self._ctc.decode_batch(
                scores, languages, lengths if lengths is not None else encoded.lengths
            )
        except ValueError as exc:
            raise ModelContractError(f"CTC output mismatch: {exc}") from exc

    def _rnnt_infer(self, audio: Sequence[npt.NDArray[np.float32]], language: str) -> list[str]:
        encoded = self._encode(audio)
        joint_enc = self._intermediate("joint_enc", self._encoded_feeds("joint_enc", encoded))[0]
        vocab = self._vocabs[language]
        scorers = [
            self._scorer(language, vocab, joint_enc, len(audio), i) for i in range(len(audio))
        ]
        if self._rnnt is None:
            raise EngineNotReadyError("RNNT decoder unavailable")
        try:
            return self._rnnt.decode_many(
                language=language,
                frame_counts=[max(0, int(value)) for value in encoded.lengths],
                scores=scorers,
            )
        except ValueError as exc:
            raise ModelContractError(f"RNNT output mismatch for {language}: {exc}") from exc

    def _scorer(
        self, language: str, vocab: LanguageVocabulary, joint_enc: Any, batch: int, item: int
    ) -> Callable[[int, tuple[int, ...]], npt.NDArray[np.float32]]:
        def score(frame: int, emitted: tuple[int, ...]) -> npt.NDArray[np.float32]:
            labels = np.full((batch, max(1, len(emitted) + 1)), vocab.blank_id, np.int64)
            if emitted:
                labels[item, 1:] = emitted
            pred_graph = self._graphs["joint_pred"]
            dtype = np.int32 if "int32" in pred_graph.input_types[0] else np.int64
            pred = self._intermediate(
                "joint_pred", {pred_graph.inputs[0]: labels.astype(dtype, copy=False)}
            )[0]
            pre_graph = self._graphs["joint_pre"]
            pre = self._intermediate(
                "joint_pre", {pre_graph.inputs[0]: joint_enc, pre_graph.inputs[1]: pred}
            )[0]
            role = f"post:{language}"
            logits = _terminal_tensor(
                role, self._terminal(role, {self._graphs[role].inputs[0]: pre})
            )
            return _rnnt_position(logits, item, frame)

        return score

    def _encode(self, audio: Sequence[npt.NDArray[np.float32]]) -> _Encoded:
        if self._ort is None:
            raise EngineNotReadyError("ORT unavailable")
        buffer = self._pool.acquire(audio, self._ort, self._cuda, self._device)
        graph = self._graphs["encoder"]
        audio_i = _one(
            "floating encoder input",
            [i for i, kind in enumerate(graph.input_types) if "float" in kind],
        )
        length_i = _one(
            "integer encoder input",
            [i for i, kind in enumerate(graph.input_types) if "int" in kind],
        )
        return self._run_encoder(
            {
                graph.inputs[audio_i]: buffer.device_audio if self._cuda else buffer.audio,
                graph.inputs[length_i]: buffer.device_lengths if self._cuda else buffer.lengths,
            }
        )

    def _encoded_feeds(self, role: str, encoded: _Encoded) -> dict[str, Any]:
        graph = self._graphs[role]
        feeds: dict[str, Any] = {}
        for name, kind in zip(graph.inputs, graph.input_types, strict=True):
            if "float" in kind:
                feeds[name] = encoded.value
            elif "int" in kind:
                feeds[name] = encoded.lengths
            else:
                raise ModelContractError(f"unsupported {role} input type {kind}")
        return feeds

    def _run_encoder(self, feeds: Mapping[str, Any]) -> _Encoded:
        session, graph = self._sessions["encoder"], self._graphs["encoder"]
        encoded_i = _one(
            "floating encoder output",
            [i for i, kind in enumerate(graph.output_types) if "float" in kind],
        )
        length_indices = [i for i, kind in enumerate(graph.output_types) if "int" in kind]
        if self._cuda:
            if not length_indices:
                raise ModelContractError(
                    "CUDA encoder must export lengths; host inference is forbidden"
                )
            binding = session.io_binding()
            keepalive = self._bind(binding, feeds)
            for index, name in enumerate(graph.outputs):
                if index == encoded_i:
                    binding.bind_output(name, "cuda", self._device)
                else:
                    binding.bind_output(name, "cpu")
            session.run_with_iobinding(binding)
            values = binding.get_outputs()
            result = _Encoded(
                values[encoded_i],
                np.asarray(values[length_indices[0]].numpy(), dtype=np.int64).reshape(-1),
            )
            del keepalive
            return result
        values = session.run(list(graph.outputs), dict(feeds))
        encoded = np.asarray(values[encoded_i])
        lengths = (
            np.asarray(values[length_indices[0]], dtype=np.int64).reshape(-1)
            if length_indices
            else np.full(encoded.shape[0], encoded.shape[1], np.int64)
        )
        return _Encoded(encoded, lengths)

    def _intermediate(self, role: str, feeds: Mapping[str, Any]) -> list[Any]:
        session, graph = self._sessions[role], self._graphs[role]
        if not self._cuda:
            return list(session.run(list(graph.outputs), dict(feeds)))
        binding = session.io_binding()
        keepalive = self._bind(binding, feeds)
        for name in graph.outputs:
            binding.bind_output(name, "cuda", self._device)
        session.run_with_iobinding(binding)
        result = list(binding.get_outputs())
        del keepalive
        return result

    def _terminal(self, role: str, feeds: Mapping[str, Any]) -> dict[str, npt.NDArray[np.generic]]:
        session, graph = self._sessions[role], self._graphs[role]
        if self._cuda:
            binding = session.io_binding()
            keepalive = self._bind(binding, feeds)
            for name in graph.outputs:
                binding.bind_output(name, "cpu")
            session.run_with_iobinding(binding)
            values = binding.copy_outputs_to_cpu()
            del keepalive
        else:
            values = session.run(list(graph.outputs), dict(feeds))
        return {name: np.asarray(value) for name, value in zip(graph.outputs, values, strict=True)}

    def _bind(self, binding: Any, feeds: Mapping[str, Any]) -> list[Any]:
        if self._ort is None:
            raise EngineNotReadyError("ORT unavailable")
        keepalive: list[Any] = []
        for name, value in feeds.items():
            if callable(getattr(value, "data_ptr", None)):
                binding.bind_ortvalue_input(name, value)
            else:
                array = np.ascontiguousarray(value)
                device_value = self._ort.OrtValue.ortvalue_from_numpy(array, "cuda", self._device)
                keepalive.extend((array, device_value))
                binding.bind_ortvalue_input(name, device_value)
        return keepalive

    def _notify(self, callback: ProgressCallback | None, stage: str) -> None:
        self._set_readiness(EngineState.STARTING, stage)
        if callback:
            callback(stage)

    def _release(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._graphs.clear()
            self._pool.clear()
            self._vocabs.clear()
            self._ctc = self._rnnt = None
            self._ort = None
            self._warmed.clear()


def _check_provider(session: Any, role: str, cuda: bool) -> None:
    providers = tuple(session.get_providers())
    if cuda:
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise ProviderUnavailableError(f"{role} providers={providers!r}; refusing CPU fallback")
        disable = getattr(session, "disable_fallback", None)
        if not callable(disable):
            raise ProviderUnavailableError(f"{role} cannot disable provider fallback")
        disable()
    elif providers != ("CPUExecutionProvider",):
        raise ProviderUnavailableError(f"CPU development session {role} providers={providers!r}")


def _graph_contract(role: str, session: Any) -> GraphContract:
    inputs, outputs = tuple(session.get_inputs()), tuple(session.get_outputs())
    allowed = {1, 2} if role == "ctc" else ({2} if role in {"encoder", "joint_pre"} else {1})
    if len(inputs) not in allowed or not outputs:
        raise ModelContractError(
            f"{role} graph has inputs={[item.name for item in inputs]!r}, "
            f"outputs={[item.name for item in outputs]!r}; expected {sorted(allowed)} "
            "inputs and at least one output. Cache-aware exports are unsupported"
        )
    contract = GraphContract(
        tuple(item.name for item in inputs),
        tuple(item.name for item in outputs),
        tuple(str(item.type).lower() for item in inputs),
        tuple(str(item.type).lower() for item in outputs),
    )
    if role == "encoder" and (
        sum("float" in kind for kind in contract.input_types) != 1
        or sum("int" in kind for kind in contract.input_types) != 1
        or sum("float" in kind for kind in contract.output_types) != 1
    ):
        raise ModelContractError(f"encoder tensor types mismatch: {contract!r}")
    if role == "joint_pred" and "int" not in contract.input_types[0]:
        raise ModelContractError(
            f"joint_pred token input is not integer: {contract.input_types[0]}"
        )
    return contract


def _one(description: str, values: Sequence[int]) -> int:
    if len(values) != 1:
        raise ModelContractError(f"expected one {description}, got indices {tuple(values)!r}")
    return values[0]


def _terminal_tensor(
    role: str, outputs: Mapping[str, npt.NDArray[np.generic]]
) -> npt.NDArray[np.generic]:
    values = [
        value
        for value in outputs.values()
        if value.ndim >= 2 and np.issubdtype(value.dtype, np.floating)
    ]
    if len(values) != 1:
        raise ModelContractError(
            f"{role} terminal outputs ambiguous: {[(v.shape, v.dtype) for v in outputs.values()]!r}"
        )
    return values[0]


def _length_output(
    outputs: Mapping[str, npt.NDArray[np.generic]], batch: int
) -> npt.NDArray[np.int64] | None:
    values = [
        np.asarray(value, dtype=np.int64).reshape(-1)
        for value in outputs.values()
        if np.issubdtype(value.dtype, np.integer) and value.size == batch
    ]
    if len(values) > 1:
        raise ModelContractError("multiple terminal length outputs")
    return values[0] if values else None


def _rnnt_position(
    logits: npt.NDArray[np.generic], item: int, frame: int
) -> npt.NDArray[np.float32]:
    if logits.ndim == 2:
        value = logits[item : item + 1]
    elif logits.ndim == 3:
        value = logits[item : item + 1, min(frame, logits.shape[1] - 1)]
    elif logits.ndim >= 4:
        value = logits[item : item + 1, min(frame, logits.shape[1] - 1), -1]
    else:
        raise ModelContractError(f"RNNT logits shape unsupported: {logits.shape}")
    return np.asarray(value, dtype=np.float32)


def _audio(value: npt.NDArray[np.generic]) -> npt.NDArray[np.float32]:
    if value.dtype == np.int16:
        return np.asarray(value, np.float32) / np.float32(32768)
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"audio dtype must be int16 or float, got {value.dtype}")
    result = np.asarray(value, np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("audio contains non-finite samples")
    return np.ascontiguousarray(np.clip(result, -1, 1))


def _result(request: TranscriptionRequest, text: str, elapsed: float) -> TranscriptionResult:
    return TranscriptionResult(
        text=text,
        language=request.language,
        decoder=request.decoder,
        audio_duration_ms=int(request.audio.size * 1000 / request.sample_rate),
        inference_ms=elapsed,
    )
