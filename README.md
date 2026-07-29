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
        | REST multipart or WSS binary PCM16
        v
FastAPI gateway
  +-- protocol validation and admission control
  +-- energy VAD and endpoint detector
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

# Only for the optional official Transformers wrapper engine, not the ORT service
uv sync --frozen --extra official-gpu
```

Never use `uv sync --all-extras`: it would attempt to resolve mutually exclusive
CPU and GPU runtime wheels. The production ORT image intentionally excludes the
separate multi-gigabyte PyTorch stack.

## REST API

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

## WebSocket API

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

### Production WebSocket admission

Every production connection needs:

```text
Authorization: Bearer <token-read-from-ASR_WEBSOCKET_BEARER_TOKEN_FILE>
```

The token file must be an absolute-path regular file containing one 32–4096
character non-whitespace value. A browser `Origin`, when present, must exactly
match one entry in `ASR_WEBSOCKET_ALLOWED_ORIGINS`. An empty allowlist rejects
all Origin-bearing clients but permits authenticated native clients that omit an
Origin header. Terminate TLS at the ingress/load balancer; use a sticky-session
rule for `/v1/realtime`.

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

### Compose deployment

Prerequisites:

- NVIDIA driver compatible with the pinned CUDA base image and NVIDIA Container
  Toolkit installed on the host.
- One GPU per ASR process.
- A valid Hugging Face token file for the gated checkpoint.
- An independent 32+ character WebSocket bearer token file.
- A private GHCR `read:packages` credential if the serving image remains private.

Create untracked local files:

```bash
mkdir -p .secrets
printf '%s' '<hf-read-token>' > .secrets/huggingface_token
printf '%s' '<long-random-websocket-token>' > .secrets/websocket_token
chmod 600 .secrets/huggingface_token .secrets/websocket_token
```

Create `.env`:

```dotenv
ASR_IMAGE=ghcr.io/shivamb25/indicconformer-realtime-asr@sha256:<64-hex-image-digest>
ASR_MODEL_REVISION=<40-hex-model-commit>
HF_TOKEN_FILE=/absolute/path/to/.secrets/huggingface_token
ASR_WEBSOCKET_TOKEN_FILE=/absolute/path/to/.secrets/websocket_token
ASR_WEBSOCKET_ALLOWED_ORIGINS=["https://speech.example.com"]
ASR_LISTEN_ADDRESS=127.0.0.1
ASR_HOST_PORT=8000
```

Authenticate to a private package, validate the rendered deployment, pull, then
start it:

```bash
echo "$GHCR_READ_TOKEN" | docker login ghcr.io --username <github-user> --password-stdin
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
| `POST /v1/transcribe` | Bounded file/PCM transcription |
| `WS /v1/realtime` | Stateful low-latency PCM16 transcription |

Use `/health/live` for restart decisions and `/health/ready` for load-balancer
admission. Do not route production traffic until readiness is successful.

Structured logs deliberately redact audio and transcript content. Prometheus
metrics expose queue depth, admission/protocol errors, session lifecycle, audio
seconds, and latency—not audio payloads or text.

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
- Pulling the private GHCR image on this host: the current GitHub credential
  lacks `read:packages` and receives `403`. The successful local Docker image
  smoke test used the locally built image.

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
