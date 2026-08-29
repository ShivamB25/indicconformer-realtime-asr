# Repository maintenance guidance

## Runtime contracts

- Treat `POST /v1/audio/transcriptions`, `GET /v1/models`, and
  `WS /v1/realtime` as the OpenAI-compatible public API.
- Keep `POST /v1/transcribe` and binary `WS /v1/realtime/native` backward compatible;
  they are separate native protocols, not aliases for the OpenAI adapters.
- Route every inference surface through the shared scheduler. Never run model
  inference on the asyncio event loop or bypass bounded admission control.
- Require an explicit supported language. The checkpoint uses it to select a
  CTC language mask/vocabulary or language-specific RNNT joint network and has no
  LID head. `auto` requires a separately trained and calibrated 22-class spoken-LID
  model that resolves to a supported code before inference; do not simulate it.
- Keep decoder policy server-owned: CTC for partial/latency work and RNNT for
  hybrid/accuracy finals.
- Keep VAD provider-neutral: process-owned `VADProvider`, one connection-owned
  `VADStream`, and exactly one 20 ms PCM16LE frame per `score` call. Stream
  state must never cross connections.
- Keep thresholds and endpoint state in the transport. Providers return finite
  scores in `[0, 1]`; they do not commit audio or emit protocol events.
- Preserve every sample through stateful framing/resampling. Do not zero-pad,
  drop, reorder, or synthesize transport frames to satisfy a model window.
- A session may never switch VAD algorithms after a provider failure. Map
  capacity to existing retryable overload behavior and inference/runtime
  faults to the protocol's server-error close path.

## Module boundaries

- Keep `app/main.py` limited to application composition, router inclusion, and
  top-level exception mapping.
- Router modules own route registration only. WebSocket lifecycle and dispatch
  live in `connection.py`; mutable connection data and invariants live in
  `state.py`; serialized protocol models live in `schemas.py`.
- Preserve `app.api.websocket`, `app.api.openai_realtime`, and
  `app.openai_compat.realtime` as stable public facades. Add public exports
  deliberately; never use wildcard imports.
- Maintain one-way dependencies:
  `main -> router -> connection -> schema/state/use-case`. Lower layers must
  never import routers or application composition.
- Internal modules import the concrete defining submodule, not its package
  facade. This keeps object ownership and CodeGraph edges explicit and prevents
  initialization cycles.
- Split a module only around a real responsibility boundary. Do not create
  one-symbol modules or duplicate schema classes merely to shorten files.

## Architecture and research record

- Keep the durable implementation map, state-ownership table, protocol flows,
  deployment topology, research ledger, and optimization history in
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Update it when a boundary,
  runtime invariant, deployment procedure, or accepted performance result
  changes; never record credentials, model weights, audio, transcripts, or
  host-only paths.
- Keep the architecture/API/deployment summaries in `README.md` aligned with
  `docs/ARCHITECTURE.md`; link to the durable record instead of duplicating
  changing race or benchmark details in multiple places.
- `BoundedVADRuntime` may prevent a deadline/cancelled job from starting while
  it is still queued in the executor, but it cannot safely preempt a running
  `asyncio.to_thread`/ONNX call. After the start gate opens, retain job-owned
  input/state/context snapshots until the worker returns; late results are
  ignored by Future/epoch checks and cleanup still runs.
- ONNX Runtime `InferenceSession.Run` is thread-safe, but caller-owned input
  feeds and output-name storage must not change during execution. Never reuse a
  timed-out model-input buffer or mutate recurrent/context snapshots in place.
- `_OUTPUT_NAMES` remains a private list for the existing session contract;
  `Final` is static-only, not runtime immutability. Treat the list as
  application-owned read-only metadata and never mutate it during inference.
- Any synthetic VAD microbenchmark is a deterministic speed/numerical-regression
  harness only. It cannot support VAD-quality, WER/CER, or production-latency
  claims. Quality requires the pinned labeled corpus and manifest hash; CUDA and
  latency claims require the protected NVIDIA workflow.
- Model-level RNNT batching, cache-aware Conformer streaming, quantization, and
  architecture changes require a supported active checkpoint/export contract or
  retraining. Do not force inactive model artifacts, mutate `.models`, or turn a
  wrapper experiment into a model-quality claim.

## Security and deployment

- Production requires a regular absolute `ASR_API_KEY_FILE`; the same bearer key
  protects native and OpenAI inference routes. Health and metrics remain public.
- Never put model weights, Hugging Face credentials, API keys, audio, or
  transcripts in source, images, logs, metrics, fixtures, or CI artifacts.
- Model loading is local-only and revision-pinned. Do not add request-time Hub
  downloads or CPU fallback when CUDA is required by the official engine.
- The only runtime extras are `official-cpu` and `official-gpu`; they conflict.
  Each contains the matching Torch/torchaudio stack and ONNX Runtime package used
  by Silero. Select `official-cpu` for local/CI and real CPU serving, or
  `official-gpu` for CUDA serving; never synchronize all extras.
- Image builds must receive a dated immutable Ubuntu snapshot URL and exact
  package versions. Do not add mutable APT sources or guessed version defaults.
- GitHub Actions currently pins `APT_SNAPSHOT_URL` to
  `https://snapshot.ubuntu.com/ubuntu/20260701T000000Z/` with
  `ca-certificates=20260601~22.04.1`, `libgomp1=12.3.0-1ubuntu1~22.04.3`,
  `libsndfile1=1.0.31-2ubuntu0.2`, and `tini=0.19.0-1` in repository variables.
  Move the snapshot and all four versions together after a reviewed rebuild.
- GPU smoke repository variables point to
  `ghcr.io/shivamb25/indicconformer-realtime-asr` and model revision
  `e9b71b369c048e2c6b634d4c131061c34e441179`. Set the image digest only from
  the published immutable SHA image; never guess or reuse a prior digest.
- Publish only the SHA staging image before GPU smoke. Promote semantic/latest
  tags from that exact digest only after every protected GPU gate succeeds.
- Serving containers use an internal network with published ingress and no
  external route. Provisioning retains egress; do not attach serving to it.
- Prepare Compose/runtime secrets as UID/GID 10001, directory `0700`, file `0400`,
  and mount them read-only. Compose bind-secret mode/uid fields are not sufficient.
- Production VAD must be `disabled`, `silero`, or `webrtc`; `disabled` requires
  client commits because it performs no automatic endpointing. `energy` is an
  explicit development/rollback implementation and must continue to fail
  validation in production.
- Keep the Silero revision, URL, model SHA-256, and downloader identity in
  `app/vad/artifact.py`. Runtime code and provisioning must import that single
  definition. Never bake VAD weights into the image or fetch them at startup.
- Silero must use CPU ONNX Runtime with explicit thread counts and per-stream
  recurrent/context/resampler state. WebRTC must keep one classifier and
  resampler per stream. Both share only bounded process-owned dispatch.
- VAD metrics must use the closed provider/protocol/result/error label sets.
  Never add session IDs, audio, transcript content, exception text, paths, or
  user-controlled strings as metric labels.

## Live CPU deployment memory

- The verified real-model CPU image is `indic-asr-local:cpu-official`, built
  with the mutually exclusive `official-cpu` uv extra. It loads the pinned
  checkpoint from a read-only local mount and must remain offline at runtime.
- `indic-asr-cpu-vad` serves `http://127.0.0.1:18011` through
  `https://audio.aniex.site` with Silero VAD.
- `indic-asr-cpu-no-vad` serves `http://127.0.0.1:18010` through
  `https://audio-manual.aniex.site` with VAD disabled; realtime clients must
  send `input.commit`.
- Both containers use `restart=unless-stopped`. The existing
  `cloudflared.service` is a remotely managed token tunnel; configure durable
  public hostnames in Cloudflare, never with ephemeral quick tunnels.
- Both deployments share a bearer secret from one untracked host-restricted
  file. Store its client-side value as `ASR_API_KEY`; never copy the value into
  this repository, prompts, frontend code, logs, or test fixtures.
- Public `/health/ready`, real Hindi REST transcription, and permanent WSS
  `session.ready` were verified for both hostnames. Re-verify all three after
  changing containers, origins, tunnel routes, authentication, or model assets.
- Native browser `WebSocket` cannot add the required bearer header. Browser
  microphone integrations need a trusted backend proxy or short-lived ticket
  authentication; never expose the shared service key to a browser.

## Change verification

- Use the deterministic MockEngine for CPU protocol, scheduling, health, and
  client-compatibility checks. Never present mock transcripts as model-quality
  evidence.
- API changes require focused wire-contract coverage for malformed input,
  authentication, bounds, scheduler submission, and error shape.
- Verify OpenAI REST changes with the unmodified `openai-python` client. Verify
  realtime changes at the JSON event boundary, including client `event_id`,
  item correlation, and per-item delta-before-completed ordering.
- Run `uv run ruff format --check app scripts tests`, `uv run ruff check app scripts tests`,
  `uv run mypy app tests`, and `uv run pytest -q` before release.
- Official CUDA availability, no-CPU-fallback, warmup, model quality, and
  latency claims require the protected GPU workflow on an actual NVIDIA runner.
- VAD changes require contract coverage at 16 kHz and 24 kHz, exact-frame
  fragmentation, stream isolation/release, reset/final/disconnect paths,
  threshold boundaries, capacity/deadline/inference failures, and native plus
  OpenAI event ordering.
- Run `uv run python scripts/benchmark_vad.py --self-check` for pipeline smoke.
  Provider-quality claims require the same pinned labeled multilingual/noisy
  corpus for Silero, WebRTC modes 0–3, and Energy; report the manifest hash and
  all metrics emitted by the benchmark. Never treat the synthetic self-check as
  accuracy evidence.
