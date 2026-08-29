# Research ledger

Durable source and decision record: `docs/ARCHITECTURE.md`. Research was used to separate safe wrapper changes from model/export/retraining work.

## Source-backed findings

- Python `asyncio.to_thread` runs synchronous work in another thread; `wait_for`/Future cancellation does not safely preempt a callable already running. Source: https://docs.python.org/3/library/asyncio-task.html and CPython concurrent.futures Future source.
- ONNX Runtime `InferenceSession::Run` is documented thread-safe, but caller-owned input feeds and output-name storage must remain unchanged during execution. The repository keeps its required list-form output names private and never mutates them; `Final` is a static annotation, not runtime immutability. Source: https://github.com/microsoft/onnxruntime/blob/main/core/session/inference_session.h.
- Silero VAD supports ONNX Runtime and portable streaming wrappers, but the wrapper owns I/O, post-processing, and state. Source: https://github.com/snakers4/silero-vad.
- Label-Looping reports up to 2x faster batched transducer decoding at batch size 32. Source: https://arxiv.org/abs/2406.06220.
- Stateful Conformer uses bounded past/look-ahead context and activation caching to remove repeated streaming encoder work. Source: https://arxiv.org/abs/2312.17279.
- Practical Conformer reports large paper-specific latency reductions from architecture changes with quality tradeoffs. Source: https://arxiv.org/abs/2304.00171.
- A 2026 VAD window study reports Silero outperforming WebRTC/RMS and shows window/hysteresis choices affect accuracy. Source: https://arxiv.org/abs/2601.17270.

## Decisions

1. Implement only wrapper-safe, checkpoint-preserving changes in the current service.
2. Keep queued-job cancellation narrow: skip only work not yet started; never pretend to interrupt running ONNX inference.
3. Keep per-job buffer ownership after start and retain runtime output validation.
4. Defer RNNT batching, cache-aware Conformer inference, quantization, CUDA graph/I/O binding, and architecture resizing until the active pinned export supports them and quality gates exist.
5. Require the protected NVIDIA workflow for CUDA/latency claims and the pinned multilingual/noisy corpus for VAD quality claims.

Model-level changes require a compatible active checkpoint/export or retraining; changing inactive `.models` code/config is prohibited.


## Documentation synchronization

Research decisions are recorded in `docs/ARCHITECTURE.md`; `README.md` links the record and exposes only the stable runtime contract. `AGENTS.md` is the binding policy and must stay aligned without duplicating volatile benchmark details.
