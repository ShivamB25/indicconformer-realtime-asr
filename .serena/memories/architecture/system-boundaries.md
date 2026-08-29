# ASR architecture and ownership

Durable reference: `docs/ARCHITECTURE.md`; binding policy: `AGENTS.md`; API guide: `README.md`.

## Dependency direction

`app/main.py` composes the app and maps top-level exceptions. Routers register routes only. Connection modules own WebSocket lifecycle and dispatch. State modules own mutable connection invariants. Schema modules own serialized protocol models. Lower layers implement use cases, scheduler, engines, audio, VAD, and observability; they never import routers or application composition. Internal modules import concrete defining modules rather than package facades.

## Protocols

OpenAI-compatible surfaces: `POST /v1/audio/transcriptions`, `GET /v1/models`, and JSON `WS /v1/realtime`. Native backward-compatible surfaces: `POST /v1/transcribe` and binary `WS /v1/realtime/native`; they are separate protocols. Production inference routes share one bearer key; health/metrics are public.

## Inference invariant

Every REST/native/OpenAI inference request enters the shared bounded scheduler. No model inference runs on the asyncio event loop. Finals have priority; one partial per session is allowed; stale partials may be superseded; compatible batching is allowed only through the active engine's supported contract. Decoder policy is server-owned: CTC for latency/partials and RNNT for hybrid/accuracy finals.

## Audio/VAD ownership

Each connection owns one VAD stream. The process owns the provider and bounded CPU dispatch. Each score call receives exactly one 20 ms PCM16LE frame. Provider state includes recurrent/context/resampler state and never crosses connections. Providers return finite scores in `[0, 1]`; transport code owns thresholds, endpoint state, commits, and protocol events. Stateful framing/resampling preserves every sample without padding, dropping, reordering, or synthesizing frames.

## Runtime cancellation

`BoundedVADRuntime` uses a lock-protected per-job start gate. Deadline/caller cancellation can suppress a job that is still queued in the executor. It cannot safely preempt a callable after the gate opens and `asyncio.to_thread`/ONNX Runtime has started. Running jobs retain ownership of their input/state/context snapshots until return; stale results are ignored by Future/epoch checks, while queue accounting and cleanup still execute.

The regression test `test_runtime_skips_executor_queued_work_after_abandonment` saturates a one-thread executor and covers both deadline and caller-cancellation abandonment.

## Model constraints

The pinned IndicConformer checkpoint has no LID head. Language is explicit and selects CTC vocabulary/mask or language-specific RNNT behavior. `auto` requires a separately trained/calibrated 22-class spoken-LID model. Do not mutate `.models`, force inactive model artifacts, or add CPU fallback when CUDA is required.

ONNX Runtime documents `InferenceSession::Run` as thread-safe while requiring caller-owned feeds/output-name storage to remain unchanged during execution. `_OUTPUT_NAMES` is deliberately a private list because the existing session contract expects a list; `Final` is static-only, not runtime immutability. Application code must never mutate it during concurrent calls. Never reuse a timed-out model-input buffer or mutate recurrent/context snapshots in place.


## Documentation synchronization

`README.md` now summarizes the two-phase cancellation contract and links to `docs/ARCHITECTURE.md`. `AGENTS.md` requires README and architecture-record alignment.
