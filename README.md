# IndicConformer Realtime ASR

GPU-first, real-time ASR for
[`ai4bharat/indic-conformer-600m-multilingual`](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual).

The service accepts Indian-language audio over REST or a streaming WebSocket. It
keeps model execution off the asyncio event loop, gives final results priority
over partials, and fails closed in production unless the exact local model
snapshot and CUDA execution provider are healthy.

> **Status:** the transport, scheduling, deployment, health, and observability
> contracts are implemented and CPU-tested. The real model is gated and this
> workstation has no NVIDIA GPU, so no local model-quality, latency, WER/CER,
> or CUDA-execution claim is made. See [Verification and limitations](#verification-and-limitations).

## Architecture

```text
client microphone / WAV
        |
        | OpenAI/native REST or WebSocket
        v
FastAPI gateway
  +-- protocol validation and admission control
  +-- process-owned bounded VAD provider and per-session stream state
  +-- bounded per-session audio buffers
  +-- final-first, bounded, dynamic-batching scheduler
        |
        v
ONNX Runtime CUDA engine
  +-- CTC partials (latency mode)
  +-- RNNT finals (hybrid and accuracy modes)
        |
        v
transcript events / HTTP response / Prometheus metrics
```

One ASR process owns one GPU. WebSocket connections must remain sticky to that
process for their entire lifetime. The scheduler allows only one outstanding
partial per session, supersedes stale partial work, length-buckets compatible
requests, and drains active inference before shutdown.

### Code organization

The package layout follows one-way dependencies so a symbol search lands in the
module that owns the behavior:

```text
app/main.py                         application composition and exception mapping
app/api/                            HTTP and WebSocket transport adapters
  websocket/router.py               native WebSocket route assembly
  websocket/connection.py           native connection lifecycle and dispatch
  websocket/state.py                native configuration and live session state
  openai_realtime/router.py         OpenAI realtime route assembly
  openai_realtime/connection.py     OpenAI event dispatch and inference handoff
app/openai_compat/                  OpenAI-specific wire contracts and mapping
  realtime/schemas.py               client/server Pydantic event models
  realtime/state.py                 buffers, VAD state, and committed turns
app/transcription.py                transport-independent transcription use case
app/audio/                          PCM, resampling, endpointing, stable prefix
app/vad/                            Energy, WebRTC, and direct-ONNX Silero providers
app/engine/                         engine contracts, scheduler, ORT and decoders
app/core/                           settings, lifespan, readiness, logging, shared types
app/observability/                  metrics and tracing
```

Dependencies point from application composition to routers, from routers to
connection handlers, and from handlers to schemas/state/use cases. Schema and
state modules never import routers. Package `__init__.py` files are compatibility
facades only: internal modules import the concrete defining module to avoid
cycles and make CodeGraph call paths unambiguous.

## Supported languages and modes

Language selection is required. The supported codes are:

```text
as  bn  brx doi gu  hi  kn  kok ks  mai ml
mni mr  ne  or  pa  sa  sat sd  ta  te  ur
```

| Client mode | Server decoder | Intended result |
| --- | --- | --- |
| `latency` | CTC | Lowest-latency partial/final behavior |
| `hybrid` | RNNT | Default: CTC partials plus RNNT final |
| `accuracy` | RNNT | Final-oriented RNNT behavior |

The decoder is server-owned. A REST `decoder` field is rejected rather than
silently overriding the mode.

## Quick start: CPU-safe local service

The default local engine is `mock`. It is deterministic and useful for client,
protocol, scheduler, health, and deployment integration. It does **not** perform
speech recognition and must never be used to evaluate transcription quality.

```bash
uv sync --frozen --extra cpu --group dev
ASR_ENGINE=mock uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl -fsS http://127.0.0.1:8000/health/live
curl -fsS http://127.0.0.1:8000/health/ready
curl -fsS http://127.0.0.1:8000/metrics
```

Dependencies are pinned in `uv.lock`. CPU and GPU ORT packages intentionally
conflict; do not synchronize them together.

```bash
# CPU development, lint, typing, and test environment
uv sync --frozen --extra cpu --group dev
uv run ruff format --check app scripts tests
uv run ruff check app scripts tests
uv run mypy app
uv run pytest -q

# Production ORT/CUDA environment; excludes the development group
uv sync --frozen --extra gpu --no-group dev

# Official Transformers wrapper with local CPU or CUDA inference
uv sync --frozen --extra official-cpu
uv sync --frozen --extra official-gpu
```

Never use `uv sync --all-extras`: each lean ORT or official-wrapper CPU/GPU
runtime selects one mutually exclusive wheel set. The production ORT image
intentionally excludes the separate multi-gigabyte PyTorch stack.

## Voice activity detection

VAD is a process-owned provider with one isolated stream per VAD-enabled
connection. Both native 16 kHz PCM and OpenAI-compatible 24 kHz PCM enter as
exact 20 ms frames. The selected provider owns model/resampler state, bounded
CPU workers, pending admission, deadlines, and live-stream leases; transport
handlers own protocol thresholds and endpoint state. A provider error fails the
affected session instead of switching algorithms mid-utterance.

| `ASR_VAD_PROVIDER` | Intended use | Production |
| --- | --- | --- |
| `disabled` | No classifier; every valid frame is retained until client commit | Explicit opt-out |
| `silero` | Pinned Silero VAD 6.2.1 through direct CPU ONNX Runtime | Default Compose choice |
| `webrtc` | Lightweight binary baseline; modes `0` through `3` | Explicit alternative |
| `energy` | Deterministic normalized-RMS development/rollback provider | Rejected at startup |

The local default is `energy`; production must explicitly select `disabled`,
`silero`, or `webrtc`. With `disabled`, automatic endpointing is unavailable
and clients must send `input.commit`.

```dotenv
ASR_VAD_PROVIDER=silero
ASR_VAD_MODEL_PATH=/models/vad/silero-v6.2.1/silero_vad.onnx
ASR_VAD_MODEL_SHA256=1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3
ASR_VAD_MAX_STREAMS=128
ASR_VAD_CPU_WORKERS=2
ASR_VAD_PENDING_CAPACITY=128
ASR_VAD_CLASSIFICATION_DEADLINE_SECONDS=0.1
ASR_VAD_SPEECH_THRESHOLD=0.5
ASR_WEBRTC_VAD_MODE=1
```

`ASR_VAD_SPEECH_THRESHOLD` is the native protocol override. OpenAI realtime
uses the `server_vad.threshold` supplied in `session.update`. Silero consumes
contiguous 512-sample windows while preserving its recurrent/context state;
the 20 ms transport frame is never zero-padded or dropped. WebRTC resamples
24 kHz input statefully to its exact 16 kHz frame contract.

Provision or verify the immutable Silero ONNX model and license outside the
serving process:

```bash
uv run python scripts/download_vad_model.py --output-dir /models/vad/silero-v6.2.1
uv run python scripts/download_vad_model.py \
  --output-dir /models/vad/silero-v6.2.1 --offline
```

The downloader pins the upstream revision and SHA-256 digests, publishes
atomically, rejects symlinks/non-regular files, and is idempotent. Compose runs
it in `model-init`; the serving container receives the completed volume
read-only and remains offline.

## Interactive API documentation

FastAPI serves interactive Swagger UI at `/docs`, ReDoc at `/redoc`, and the
OpenAPI 3.1 document at `/openapi.json`. Swagger provides upload controls and
defaults for both `POST /v1/transcribe` and
`POST /v1/audio/transcriptions`. Use **Authorize** to set the shared bearer API
key before executing an inference request; keyless local development can leave
it empty. Check `/health/ready` first.

OpenAPI does not execute WebSocket protocols. The Swagger introduction lists
both realtime URLs and their audio framing requirements; use the native or
OpenAI realtime client examples below to exercise them.

## OpenAI-compatible API

The primary client surface is compatible with an unmodified current
`openai-python` client. Set its `base_url` to this service's `/v1` prefix and
use the same bearer API key configured for the server:

```python
from openai import OpenAI

client = OpenAI(
    api_key="read-from-your-auth-flow",
    base_url="https://asr.example.com/v1",
)

with open("speech.wav", "rb") as audio:
    transcript = client.audio.transcriptions.create(
        model="indicconformer-600m",
        file=audio,
        language="hi",
        response_format="json",
    )

print(transcript.text)
```

`POST /v1/audio/transcriptions` accepts normal multipart audio containers,
decodes them to mono 16 kHz audio, and submits an RNNT final. `model` and
`language` are required. The canonical model ID is
`ai4bharat/indic-conformer-600m-multilingual`; `indicconformer-600m` is an
accepted alias. Supported response formats are `json` (default) and `text`.
`stream=true`, timestamps, prompts, non-zero temperature, unknown fields, and
unsupported languages are rejected with an OpenAI-shaped error rather than
silently ignored.

Model discovery is available at `GET /v1/models` and `GET /v1/models/{model}`.
Every OpenAI REST response includes `x-request-id`.

### OpenAI realtime transcription

Connect to `wss://HOST/v1/realtime/transcription_sessions`. This is the current
GA transcription-session event contract, separate from the native binary
protocol:

1. Receive `session.created`.
2. Send `session.update` with `session.type: "transcription"`, PCM input at
   24 kHz, one supported language, and either `server_vad` or `null`.
3. Send `input_audio_buffer.append` events containing base64 PCM16LE mono
   24 kHz audio.
4. With manual endpointing, send `input_audio_buffer.commit`; the server first
   emits `input_audio_buffer.committed`, then item-correlated transcription
   `delta` and `completed` events. With `server_vad`, speech boundaries commit
   automatically.

```json
{
  "type": "session.update",
  "event_id": "update-1",
  "session": {
    "type": "transcription",
    "audio": {
      "input": {
        "format": {"type": "audio/pcm", "rate": 24000},
        "transcription": {
          "model": "indicconformer-600m",
          "languages": ["hi"]
        },
        "turn_detection": {
          "type": "server_vad",
          "threshold": 0.5,
          "prefix_padding_ms": 300,
          "silence_duration_ms": 500
        }
      }
    }
  }
}
```

The adapter validates base64 and buffer bounds, resamples 24 kHz input to the
engine's 16 kHz contract, preserves per-item `delta`-before-`completed`
ordering, and permits later turns to finish before earlier turns. Use client
`event_id` values to correlate recoverable errors.

## Native REST API

The native API remains available for low-overhead internal clients.

`POST /v1/transcribe` accepts multipart form data:

| Field | Required | Value |
| --- | --- | --- |
| `audio` | yes | mono, 16 kHz signed PCM16 WAV, or headerless `pcm_s16le` |
| `language` | yes | one supported language code |
| `mode` | yes | `latency`, `hybrid`, or `accuracy` |

Example using a PCM16 WAV:

```bash
curl --fail-with-body \
  --request POST http://127.0.0.1:8000/v1/transcribe \
  --form audio=@sample-pcm16.wav \
  --form language=hi \
  --form mode=hybrid
```

A successful response has this shape:

```json
{
  "text": "…",
  "language": "hi",
  "mode": "hybrid",
  "decoder": "rnnt",
  "audio_duration_ms": 5160,
  "inference_ms": 42.7,
  "request_id": "…"
}
```

The server validates input size, mono channel count, 16 kHz sample rate, PCM16
encoding, and configured duration limits before scheduling inference. A `4xx`
response contains a machine-readable `error` and `request_id`; server failures
are not reported as successful transcripts.

## Native WebSocket API

Connect to `ws://HOST:PORT/v1/realtime` locally or `wss://…/v1/realtime` behind
TLS. The wire contract is deliberately strict:

1. Send exactly one JSON `session.start` event.
2. Send raw binary PCM16LE audio frames only: **mono, 16 kHz, exactly 20 ms** per
   WebSocket message. Each frame is **320 samples / 640 bytes**, little-endian.
3. Send JSON `input.commit` to finalize buffered audio immediately when the
   client controls segment boundaries.
4. Receive JSON server events until `transcript.final` or an `error` event.

Client start event:

```json
{
  "type": "session.start",
  "language": "hi",
  "format": "pcm_s16le",
  "sample_rate": 16000,
  "channels": 1,
  "mode": "hybrid",
  "vad": true
}
```

`mode` defaults to `hybrid`; `vad` defaults to `true`.

Server events:

```jsonc
{"type":"session.ready", "session_id":"…"}
{"type":"speech.started"}
{"type":"transcript.partial", "text":"…", "revision":3, "is_stable":false}
{"type":"transcript.final", "text":"…", "language":"hi", "decoder":"rnnt",
 "audio_duration_ms":5160, "endpoint_to_final_ms":61.2}
{"type":"error", "code":"invalid_frame_size", "message":"…", "retryable":false}
```

### Browser client sketch

The browser must capture or resample microphone audio to 16 kHz mono PCM16 and
split it into exact 640-byte buffers. Do not send WAV headers or arbitrary sized
`Float32Array` chunks over the socket.

```js
const token = "read-from-your-auth-flow";
const ws = new WebSocket("wss://asr.example.com/v1/realtime");
ws.binaryType = "arraybuffer";

ws.addEventListener("open", () => {
  ws.send(JSON.stringify({
    type: "session.start",
    language: "hi",
    format: "pcm_s16le",
    sample_rate: 16000,
    channels: 1,
    mode: "hybrid",
    vad: true,
  }));
});

ws.addEventListener("message", ({ data }) => {
  const event = JSON.parse(data);
  if (event.type === "transcript.partial") renderPartial(event.text, event.is_stable);
  if (event.type === "transcript.final") renderFinal(event.text);
  if (event.type === "error") reportAsrError(event);
});

// `pcmFrame` must be an ArrayBuffer containing exactly 640 bytes of PCM16LE.
function sendPcmFrame(pcmFrame) {
  if (ws.readyState !== WebSocket.OPEN || pcmFrame.byteLength !== 640) {
    throw new Error("expected one 20 ms / 640-byte PCM16LE frame");
  }
  ws.send(pcmFrame);
}

function commit() {
  ws.send(JSON.stringify({ type: "input.commit" }));
}
```

With VAD enabled, speech starts after roughly 60 ms of speech and ends after
roughly 600 ms of silence; a 30-second utterance limit forces a final event.
`input.commit` is the preferred mechanism when the client knows the speaker has
finished. With `vad: false`, the client must commit; the server cannot infer an
endpoint from silence.

Malformed JSON, a second start event, text frames after startup, non-binary
audio, frames other than 640 bytes, or invalid state transitions produce an
explicit protocol error and close the session. Use a new connection to retry a
non-retryable protocol failure.

### Production inference admission

Every production REST or WebSocket inference request needs:

```text
Authorization: Bearer <token-read-from-ASR_API_KEY_FILE>
```

The key file must be an absolute-path regular file containing one 32–4096
character ASCII value without whitespace. A browser `Origin`, when present,
must exactly match one entry in `ASR_WEBSOCKET_ALLOWED_ORIGINS`. An empty
allowlist rejects all Origin-bearing WebSocket clients but permits authenticated
native clients that omit an Origin header. Authentication is disabled only when
no API key is configured outside production. Health and metrics stay
unauthenticated. Terminate TLS at the ingress/load balancer and use sticky
sessions for both WebSocket routes.

## Model provisioning and GPU deployment

The AI4Bharat checkpoint is gated. A Hugging Face account must be granted access
and the provisioning token must have permission to read it. Pin a full
40-character model commit—not `main`, a branch, or a mutable tag.

### Provision a verified snapshot

The standalone downloader is idempotent. It stages on the target filesystem,
validates expected assets, hashes every payload, writes a manifest and `.complete`
marker, then atomically publishes the snapshot.

```bash
uv run python scripts/download_model.py \
  --repository ai4bharat/indic-conformer-600m-multilingual \
  --revision <40-hex-model-commit> \
  --token-file /run/secrets/huggingface_token \
  --output-dir /models/indicconformer

uv run python scripts/verify_model.py \
  --model-dir /models/indicconformer \
  --repository ai4bharat/indic-conformer-600m-multilingual \
  --revision <40-hex-model-commit>
```

The inference service never downloads a model on the request path. It loads only
the completed local snapshot with Hub and Transformers offline mode enabled.

### Published Docker image matrix

The public Docker Hub repository exposes four explicit Linux/amd64 variants:

| Tag | Source | Runtime/VAD | OCI manifest digest |
| --- | --- | --- | --- |
| `cpu-no-vad` | `main` | CPU; no VAD package | `sha256:a233f24cc31fd94d080d99f3919ee18753d1db0b469946637538bf0cf6574918` |
| `gpu-no-vad` | `main` | CUDA/TensorRT; no VAD package | `sha256:05470881ea523bc8d07f73b48eb560b33ee1ae31785b003f9707f39489db9093` |
| `cpu-vad` | VAD branch | CPU; Silero/WebRTC/Energy | `sha256:44a35fee708d11050f9d0e92bf10740e51ec3a8c56c1690d770e11b3fa58552f` |
| `gpu-vad` | VAD branch | CUDA/TensorRT; CPU Silero/WebRTC/Energy | `sha256:8ed92881a719c0e5e1aa0f3ea681a94eb6aab73cea04bfc79c17c761af7b3620` |

All four tags are public and contain application dependencies only; ASR and
Silero model weights remain external mounts. Pin deployments by the published
digest rather than relying on a mutable tag. CPU variants default to
`ASR_REQUIRE_CUDA=false` and are intended for CPU execution and validation.

The VAD branch Dockerfile builds either accelerator dependency set:

```bash
# GPU + VAD (the default)
docker build -f deploy/Dockerfile \
  --build-arg UV_EXTRA=gpu \
  -t shivam250/indicconformer-realtime-asr:gpu-vad .

# CPU + VAD
docker build -f deploy/Dockerfile \
  --build-arg RUNTIME_IMAGE=ubuntu:22.04@sha256:0e0a0fc6d18feda9db1590da249ac93e8d5abfea8f4c3c0c849ce512b5ef8982 \
  --build-arg UV_EXTRA=cpu \
  --build-arg ASR_REQUIRE_CUDA=false \
  -t shivam250/indicconformer-realtime-asr:cpu-vad .

# CPU official-wrapper engine used for real local CPU transcription
docker build -f deploy/Dockerfile \
  --build-arg RUNTIME_IMAGE=ubuntu:22.04@sha256:0e0a0fc6d18feda9db1590da249ac93e8d5abfea8f4c3c0c849ce512b5ef8982 \
  --build-arg UV_EXTRA=official-cpu \
  --build-arg ASR_REQUIRE_CUDA=false \
  -t indic-asr-local:cpu-official .
```

### Compose deployment

Prerequisites:

- NVIDIA driver compatible with the pinned CUDA base image and NVIDIA Container
  Toolkit installed on the host.
- One GPU per ASR process.
- A valid Hugging Face token file for the gated checkpoint.
- An independent 32+ character API key file.
- No registry credential is required for the public Docker Hub images.

Create untracked local files:

```bash
mkdir -p .secrets
printf '%s' '<hf-read-token>' > .secrets/huggingface_token
printf '%s' '<long-random-api-key>' > .secrets/api_key
chmod 600 .secrets/huggingface_token .secrets/api_key
```

Create `.env`:

```dotenv
ASR_IMAGE=shivam250/indicconformer-realtime-asr:gpu-vad@sha256:8ed92881a719c0e5e1aa0f3ea681a94eb6aab73cea04bfc79c17c761af7b3620
ASR_MODEL_REVISION=<40-hex-model-commit>
HF_TOKEN_FILE=/absolute/path/to/.secrets/huggingface_token
ASR_API_KEY_TOKEN_FILE=/absolute/path/to/.secrets/api_key
ASR_WEBSOCKET_ALLOWED_ORIGINS=["https://speech.example.com"]
ASR_LISTEN_ADDRESS=127.0.0.1
ASR_HOST_PORT=8000
```

Validate the rendered deployment, pull the public image, then start it:

```bash
docker compose -f deploy/compose.yaml config --quiet
docker compose -f deploy/compose.yaml pull
docker compose -f deploy/compose.yaml up --no-build
```

`model-init` is the sole network-enabled model step. The ASR process mounts the
completed model volume read-only, runs non-root, uses a read-only root filesystem,
drops Linux capabilities, requires CUDA, and exposes port 8000 on loopback by
default. `/health/ready` stays unavailable until snapshot verification, CUDA EP
selection without CPU fallback, ONNX session initialization, warmup, and
scheduler startup all succeed.

For multiple GPUs, run one replica per GPU and keep every WebSocket session
pinned to its selected replica.

## Operations

| Endpoint | Meaning |
| --- | --- |
| `GET /health/live` | Process/event-loop liveness only; never invokes inference |
| `GET /health/ready` | Exact model, CUDA, warmup, and scheduler readiness |
| `GET /metrics` | Prometheus counters, gauges, and latency histograms |
| `POST /v1/audio/transcriptions` | OpenAI-compatible bounded audio transcription |
| `GET /v1/models` | OpenAI-compatible model discovery |
| `WS /v1/realtime/transcription_sessions` | OpenAI GA realtime transcription events |
| `POST /v1/transcribe` | Native bounded PCM transcription |
| `WS /v1/realtime` | Native low-overhead PCM16 transcription |

Use `/health/live` for restart decisions and `/health/ready` for load-balancer
admission. Do not route production traffic until readiness is successful.

Structured logs deliberately redact audio and transcript content. Prometheus
metrics expose scheduler/VAD queue depth, selected VAD provider, live VAD
streams, decisions, endpoint events, bounded runtime errors, queue/inference
latency, admission/protocol errors, session lifecycle, and audio seconds—not
audio payloads or text. All label domains are closed.

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

Compare Energy, WebRTC modes 0–3, and the pinned Silero model on the same
labeled corpus:

```bash
uv run python scripts/benchmark_vad.py --self-check
uv run python scripts/benchmark_vad.py \
  --manifest /protected/vad/benchmark-manifest.json \
  --silero-model /models/vad/silero-v6.2.1/silero_vad.onnx \
  --max-concurrency 32 \
  --output vad-benchmark.json
```

The VAD manifest is validated against
`scripts/vad_benchmark_manifest.schema.json` and pins every audio/noise file by
SHA-256. The report includes overall, per-language, per-condition, and
per-variant frame F1, miss rate, false-positive time, false activations/hour,
onset/endpoint p50/p95, CPU real-time factor, classification p50/p95,
RSS/live-stream, and sustainable bounded concurrency. Run all providers on the
same pre-generated noisy multilingual corpus; the synthetic self-check proves
the pipeline only, not model quality.

The benchmark filters mode/decoder selections to valid server mappings and
gates p95 latency, real-time factor, and throughput without logging audio or
transcript content. CPU CI remains offline with MockEngine. The GPU workflow is
restricted to protected self-hosted NVIDIA runners and requires an immutable
image digest, exact model revision, protected multilingual golden manifest,
CUDA provider validation, all-language warmup, and benchmark thresholds.

## Verification and limitations

### Observed on this CPU workstation

A local Docker image was built from the locked production ORT dependency set.
Inside that image, `onnxruntime` was present and `torch` was absent. The container
then ran with the deterministic MockEngine and successfully handled:

```text
GET /health/live                         -> {"status":"live"}
GET /health/ready                        -> engine=ready, scheduler=ready
POST /v1/transcribe (hi, hybrid, 5.16 s) -> decoder=rnnt, HTTP 200
GET /metrics                             -> Hindi hybrid transcription counter incremented
```

The current VAD implementation was also exercised locally with the MockEngine:

```text
WS /v1/realtime (WebRTC VAD)             -> ready, speech.started, transcript.final
WS /v1/realtime/transcription_sessions  -> one OpenAI speech/commit/delta/completed chain
GET /metrics                             -> WebRTC selected and OpenAI endpoint incremented
```

The exercised audio was a Google FLEURS `hi_in` dev recording from its
CC-BY-4.0 dataset, pinned at revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`. It was converted from IEEE-float
WAV to the API-required mono 16 kHz PCM16 WAV before submission.

This proves the container, REST API, PCM validation, Hindi request routing,
mode-to-decoder mapping, health, scheduler, and metrics contracts. It does not
prove recognition accuracy because the test engine is intentionally synthetic.

### Not yet verified

- CUDA execution, CUDA I/O binding, warmup, and no-CPU-fallback behavior.
- Real IndicConformer transcription quality, WER/CER, throughput, or latency.
- Model download on this host: unauthenticated access to the gated model returns
  `401` by design.
- Anonymous Docker Registry requests returned `200` for all four public tags;
  their published OCI manifest digests are listed in the image matrix above.

## Next steps

1. **Authorize model access.** Accept the gated model terms and create a
   least-privilege Hugging Face read token; provision and verify an exact model
   commit using the downloader.
2. **Use a real NVIDIA host.** Confirm driver/CUDA compatibility, mount the
   verified snapshot read-only, and start Compose with `ASR_ENGINE=ort`.
3. **Run the GPU release gate.** Exercise every supported language with a
   protected multilingual golden corpus; record WER/CER, p50/p95 latency,
   real-time factor, throughput, GPU memory, and queue behavior.
4. **Validate streaming clients.** Test real microphone resampling, exact
   640-byte framing, endpoint behavior under speech/silence, reconnects,
   backpressure, and sticky-session routing through the production ingress.
5. **Only then tune.** TensorRT, FP16/TF32 policy changes, CUDA Graph capture,
   and a compiled RNNT decoder remain intentionally disabled until target-GPU
   profiling shows gains without multilingual WER/CER regression.
