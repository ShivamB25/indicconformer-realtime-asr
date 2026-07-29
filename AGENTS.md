# Repository maintenance guidance

## Runtime contracts

- Treat `POST /v1/audio/transcriptions`, `GET /v1/models`, and
  `WS /v1/realtime/transcription_sessions` as the OpenAI-compatible public API.
- Keep `POST /v1/transcribe` and binary `WS /v1/realtime` backward compatible;
  they are separate native protocols, not aliases for the OpenAI adapters.
- Route every inference surface through the shared scheduler. Never run model
  inference on the asyncio event loop or bypass bounded admission control.
- Require an explicit supported language. The checkpoint does not provide
  trustworthy automatic language detection.
- Keep decoder policy server-owned: CTC for partial/latency work and RNNT for
  hybrid/accuracy finals.

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

## Security and deployment

- Production requires a regular absolute `ASR_API_KEY_FILE`; the same bearer key
  protects native and OpenAI inference routes. Health and metrics remain public.
- Never put model weights, Hugging Face credentials, API keys, audio, or
  transcripts in source, images, logs, metrics, fixtures, or CI artifacts.
- Model loading is local-only and revision-pinned. Do not add request-time Hub
  downloads or CPU fallback to the production ORT engine.
- CPU and GPU ONNX Runtime extras conflict. Use `--extra cpu --group dev` for
  local/CI work and `--extra gpu --no-group dev` for the serving image; never
  synchronize all extras.
- The optional `official-gpu` Transformers wrapper is not part of the lean ORT
  production image.

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
- CUDA provider selection, no-CPU-fallback, warmup, model quality, and latency
  claims require the protected GPU workflow on an actual NVIDIA runner.
