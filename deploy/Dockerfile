# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:2fcc4280646484290cc50dce5e65f388dd04352b07cbe89a635703bd1f9aedb6
FROM ${CUDA_IMAGE}

ARG CUDA_IMAGE
COPY --from=ghcr.io/astral-sh/uv:0.11.33@sha256:77280f2f771df71f90786c314fe1bbc1e023feac652969bbf139c280babf2eb7 /uv /uvx /usr/local/bin/

RUN case "${CUDA_IMAGE}" in \
      *@sha256:*) ;; \
      *) echo >&2 "CUDA_IMAGE must be pinned by digest"; exit 64 ;; \
    esac \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends ca-certificates libgomp1 libsndfile1 tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 asr \
    && useradd --uid 10001 --gid asr --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin asr \
    && install --directory --owner=asr --group=asr --mode=0755 /app /models /models/vad

ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    ASR_HOST=0.0.0.0 \
    ASR_PORT=8000 \
    ASR_OFFLINE=true \
    ASR_REQUIRE_CUDA=true \
    ASR_ENGINE=ort \
    ASR_VAD_PROVIDER=silero \
    ASR_VAD_MODEL_PATH=/models/vad/silero-v6.2.1/silero_vad.onnx \
    ASR_VAD_MODEL_SHA256=1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3

WORKDIR /app
RUN uv python install 3.11

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --extra gpu --no-install-project

COPY --chown=asr:asr app/ ./app/
COPY --chown=asr:asr scripts/ ./scripts/

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3).read()"]

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--ws-ping-interval", "20", "--ws-ping-timeout", "20", "--timeout-graceful-shutdown", "30", "--no-access-log"]
