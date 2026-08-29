# Dual CPU ASR deployment behind Cloudflared

Date of original deployment record: 2026-08-21. Durable deployment topology and safety record. See `docs/ARCHITECTURE.md` for the full architecture/research ledger.

## Intended deployment

- `indic-asr-cpu-vad`: Silero automatic endpointing, internal ASR port 8000, Cloudflared origin `http://127.0.0.1:18011`.
- `indic-asr-cpu-no-vad`: VAD disabled, internal ASR port 8000, Cloudflared origin `http://127.0.0.1:18010`. Realtime clients must send `input.commit`.
- Both model containers use `indic-asr-local:cpu-official`, the pinned local checkpoint at `.models` mounted read-only at `/models`, runtime offline, and the shared prepared API-key volume mounted read-only.
- Both containers listening on port 8000 is correct because each container has an independent network namespace. Host publications must be distinct.

## Failure observed

Direct Compose mappings `127.0.0.1:18011:8000` and `127.0.0.1:18010:8000` were accepted into `HostConfig.PortBindings`, but Docker reported only `8000/tcp`; `NetworkSettings.Ports` was null and `docker port` reported no publication. A full Compose teardown, internal-network recreation, and `up --force-recreate` did not change it. Explicit bridge `gateway_mode_ipv4=nat` also did not establish the mappings and was reverted. This host runs Docker Compose v5.4.0 on Docker Engine 29-era internal-bridge behavior documented by Moby discussion 53256.

## Working ingress topology

Keep the two model containers attached only to the Compose `serving` network, which remains `internal: true`; remove their ineffective direct `ports` entries. Add a hardened `indic-asr-ingress` relay container that:

- mounts neither models nor secrets;
- joins `serving` for access to `asr-cpu:8000` and `asr-cpu-no-vad:8000`;
- joins a normal `ingress` bridge solely for Docker host publication;
- publishes `127.0.0.1:18011` and `127.0.0.1:18010`;
- forwards raw TCP, preserving HTTP streaming and WebSockets;
- runs read-only as the image's unprivileged user, drops all capabilities, uses `no-new-privileges`, and has small CPU/memory/PID limits;
- propagates EOF with `write_eof()`, allows the peer direction up to 30 seconds to drain, then closes both writers. The first implementation cancelled the peer pump on the first EOF and was corrected to avoid truncating half-closed request/response traffic.

Verified result:

```text
18010/tcp -> 127.0.0.1:18010
18011/tcp -> 127.0.0.1:18011
```

The two model containers remain attached only to `indicconformer-realtime-asr_serving`.

## Authentication

Inference routes require the host-only bearer key from the prepared API-key volume. The Hugging Face token is provisioning-only and is not needed by clients. Never place either token, the client API value, audio, transcripts, or exception text in source, docs, memories, logs, metrics, fixtures, or commits. Browser WebSocket clients cannot attach the bearer header; use a trusted backend proxy or short-lived ticket authentication.

## Deployment artifacts

The reproducible fix is tracked in `deploy/compose.cloudflare.yaml`, with the relay implementation in `scripts/tcp_ingress.py` and half-close regression coverage in `tests/unit/test_tcp_ingress.py`. The tracked Compose file contains no tokens or absolute host paths.

Host-specific values remain in ignored `.env`; it must define `ASR_LOCAL_MODEL_DIR` as the absolute path to the local `.models` directory plus the existing image, revision, and secret-file variables. Never commit `.env`, `.env.compose.yaml`, `.secrets`, tokens, or model assets.

Use the tracked override for every Compose operation:

```bash
docker compose --env-file .env -f deploy/compose.yaml -f deploy/compose.cloudflare.yaml --profile cpu up -d --no-build asr-cpu asr-cpu-no-vad ingress
```

For source changes, build `indic-asr-local:cpu-official` first with the exact pinned Ubuntu snapshot/package arguments and verify the resulting image contains the pulled `main` source before using `--no-build`. Never mutate the checkpoint/configuration to force unsupported batching.

## Recent verified state

After PR #5 (Silero optimization/runtime hardening) and PR #6 (executor test lifecycle cleanup) were merged into `main`, the local CPU image was rebuilt from verified `main` with `official-cpu` and the pinned model revision. Both CPU services were force-recreated and readiness returned on both public hostnames. Authenticated real-model Hindi REST transcription succeeded on both. Native `WSS /v1/realtime/native` returned `session.ready` with a session ID on both. OpenAI-compatible `WSS /v1/realtime` returned `session.created` and accepted `session.update` on both.

This verification is deployment evidence only; repeat it after any image, model asset, tunnel, origin, authentication, or container change.


## Documentation synchronization

`README.md` contains the stable deployment/API summary and links to `docs/ARCHITECTURE.md`; `AGENTS.md` requires those summaries to remain aligned. Keep host-only values and live verification details out of tracked documentation.
