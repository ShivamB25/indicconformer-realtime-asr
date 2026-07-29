# IndicConformer Realtime ASR

GPU-first, real-time ASR service for `ai4bharat/indic-conformer-600m-multilingual`.
It serves 22 language codes through REST and WebSocket APIs, keeps inference off
the asyncio event loop, prioritizes final jobs over partials, and fails closed in
production if the pinned model or CUDA provider contract is not satisfied.

## Architecture

```text
PCM16 mono 16 kHz -> FastAPI -> VAD/endpointing -> bounded final-first scheduler
                  -> ONNX Runtime CUDA -> CTC partial / RNNT final
```

One process owns one GPU. WebSocket sessions must remain sticky to that process.
The scheduler permits one outstanding partial per session, supersedes stale
partials, length-buckets compatible work, bounds batch size and total audio, and
drains active inference before engine shutdown.

## Dependency environments

Dependencies are locked in `uv.lock`. CPU and GPU wheels intentionally conflict
and must not be synchronized together.

```bash
uv sync --frozen --extra cpu --group dev
uv run ruff format --check app scripts tests
uv run ruff check app scripts tests
uv run mypy app
uv run pytest -q
```

The production ORT image synchronizes `--extra gpu --no-group dev`, so it does not
carry the separate multi-gigabyte PyTorch stack. Install `--extra official-gpu`
only when running the optional Hugging Face wrapper engine. The CUDA base and
external uv bootstrap image are digest-pinned. Never use `uv sync --all-extras`.

## Local CPU-safe service

The default development engine is deterministic and does not claim ASR quality:

```bash
uv sync --frozen --extra cpu --group dev
ASR_ENGINE=mock uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Check `GET /health/live`, `GET /health/ready`, and `GET /metrics`. The mock engine
exists for protocol, scheduling, health, and CI validation only.

## Immutable model publication

Use a full 40-character Hugging Face commit revision, never a branch or tag. The
downloader uses a same-filesystem staging directory, validates required assets,
hashes every file, fsyncs payloads and metadata, writes `.complete`, and publishes
with an atomic rename.

```bash
uv run python scripts/download_model.py \
  --repository ai4bharat/indic-conformer-600m-multilingual \
  --revision <40-hex-commit> \
  --token-file /run/secrets/huggingface_token \
  --output-dir /models/indicconformer

uv run python scripts/verify_model.py \
  --model-dir /models/indicconformer \
  --repository ai4bharat/indic-conformer-600m-multilingual \
  --revision <40-hex-commit>
```

Production loads only that local verified snapshot with Hub and Transformers
offline modes enabled. Readiness remains false until manifest identity, assets,
CUDA provider selection, disabled provider fallback, session initialization,
warmup, and scheduler startup succeed.

## APIs

### REST

`POST /v1/transcribe` accepts multipart fields:

- `audio`: mono 16 kHz signed PCM16 WAV or headerless `pcm_s16le`
- `language`: one supported language code
- `mode`: `latency`, `hybrid`, or `accuracy`

Decoder selection is server-owned: latency uses CTC; hybrid and accuracy use RNNT.
A client-supplied `decoder` field is rejected rather than silently ignored.

### WebSocket

Connect to `WS /v1/realtime`. Send one strict `session.start` JSON event, then raw
binary PCM16 frames. The server emits `session.ready`, `speech.started`, unstable
`transcript.partial` revisions, and one stable `transcript.final` per commit.
Malformed framing, text after startup, oversized sessions, and invalid state
transitions close with explicit protocol errors.

In production every connection requires
`Authorization: Bearer <token-from-ASR_WEBSOCKET_BEARER_TOKEN_FILE>`. A present
browser `Origin` must exactly match `ASR_WEBSOCKET_ALLOWED_ORIGINS`; an empty
allowlist rejects all Origin-bearing clients while still permitting authenticated
native clients without an Origin header. Uvicorn protocol ping/pong and graceful
shutdown timeouts are explicit in the image command.

## GPU deployment

Prepare two local secret files and an environment file:

```dotenv
ASR_IMAGE=ghcr.io/<owner>/<repository>@sha256:<64-hex-image-digest>
ASR_MODEL_REVISION=<40-hex-model-commit>
HF_TOKEN_FILE=/absolute/path/to/huggingface_token
ASR_WEBSOCKET_TOKEN_FILE=/absolute/path/to/websocket_token
ASR_WEBSOCKET_ALLOWED_ORIGINS=["https://speech.example.com"]
```

The WebSocket token must be one non-whitespace value of 32-4096 characters.
Then run:

```bash
docker compose -f deploy/compose.yaml config --quiet
docker compose -f deploy/compose.yaml up --build
```

`model-init` is the only network-enabled model step. The ASR service mounts the
completed volume read-only, runs as a non-root user with a read-only root
filesystem and dropped capabilities, reserves one NVIDIA GPU, and exposes port
8000 on loopback by default.

## Benchmark and release gates

```bash
uv run python scripts/benchmark.py --self-check
uv run python scripts/benchmark.py \
  --base-url http://127.0.0.1:8000 \
  --manifest /protected/golden/benchmark-manifest.json \
  --concurrency 1,2,4 \
  --duration-seconds 30,120 \
  --mode latency,hybrid,accuracy \
  --decoder ctc,rnnt \
  --output benchmark.json
```

The benchmark filters mode/decoder selections to valid server mappings and gates
p95 latency, real-time factor, and throughput without logging audio or transcript
content. CPU CI is offline and uses the mock engine. The GPU workflow runs only
on protected self-hosted runners against an exact image digest, exact model
revision, protected golden manifest/hook, CUDA provider check, all-language
warmup, and benchmark thresholds. Tagged image builds pass their produced digest
directly into the reusable GPU release gate.

## Optimization boundary

The ORT engine already uses CUDA I/O binding, reusable device buffers,
length-bucketed CTC batching, language-bucketed RNNT execution, and fail-closed
provider checks. TensorRT, FP16/TF32 policy changes, CUDA Graph capture, and a
compiled RNNT decoder are intentionally disabled until an actual target GPU and
the protected multilingual golden set prove latency gains without WER/CER
regression. This CPU workstation cannot provide that evidence; no GPU execution
or model-quality claim is made from the local test suite.
