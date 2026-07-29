# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:2fcc4280646484290cc50dce5e65f388dd04352b07cbe89a635703bd1f9aedb6
ARG CPU_IMAGE=ubuntu:22.04@sha256:0e0a0fc6d18feda9db1590da249ac93e8d5abfea8f4c3c0c849ce512b5ef8982
ARG RUNTIME_IMAGE=${CUDA_IMAGE}
ARG UV_EXTRA=official-gpu
ARG ASR_ENGINE=official
ARG ASR_REQUIRE_CUDA=true
ARG ASR_VAD_PROVIDER=silero
ARG ASR_VAD_MODEL_PATH=/models/vad/silero-v6.2.1/silero_vad.onnx
ARG ASR_VAD_MODEL_SHA256=1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3
FROM ${RUNTIME_IMAGE}

ARG RUNTIME_IMAGE
ARG UV_EXTRA
ARG ASR_ENGINE
ARG ASR_REQUIRE_CUDA
ARG ASR_VAD_PROVIDER
ARG ASR_VAD_MODEL_PATH
ARG ASR_VAD_MODEL_SHA256
# No snapshot/package defaults are committed: maintainers must source these exact
# values from a trusted Ubuntu snapshot before a release build.
ARG APT_SNAPSHOT_URL
ARG APT_CA_CERTIFICATES_VERSION
ARG APT_LIBGOMP1_VERSION
ARG APT_LIBSNDFILE1_VERSION
ARG APT_TINI_VERSION
COPY --from=ghcr.io/astral-sh/uv:0.11.33@sha256:77280f2f771df71f90786c314fe1bbc1e023feac652969bbf139c280babf2eb7 /uv /uvx /usr/local/bin/

RUN printf '%s' "${RUNTIME_IMAGE}" \
      | grep -Eq '@sha256:[0-9a-f]{64}$' \
      || { echo >&2 "RUNTIME_IMAGE must end in an exact lowercase sha256 digest"; exit 64; } \
    && case "${UV_EXTRA}:${ASR_REQUIRE_CUDA}" in \
      official-cpu:false|official-gpu:true) ;; \
      *) echo >&2 "UV_EXTRA/ASR_REQUIRE_CUDA must be official-cpu/false or official-gpu/true"; exit 64 ;; \
    esac \
    && test "${ASR_ENGINE}" = official \
      || { echo >&2 "ASR_ENGINE must be official"; exit 64; } \
    && case "${ASR_VAD_PROVIDER}" in \
      disabled|silero|webrtc) ;; \
      *) echo >&2 "ASR_VAD_PROVIDER must be disabled, silero, or webrtc"; exit 64 ;; \
    esac \
    && case "${ASR_VAD_MODEL_PATH}" in \
      /*) ;; \
      *) echo >&2 "ASR_VAD_MODEL_PATH must be absolute"; exit 64 ;; \
    esac \
    && printf '%s' "${ASR_VAD_MODEL_SHA256}" \
      | grep -Eq '^[0-9a-f]{64}$' \
      || { echo >&2 "ASR_VAD_MODEL_SHA256 must be 64 lowercase hex characters"; exit 64; } \
    && printf '%s' "${APT_SNAPSHOT_URL}" \
      | grep -Eq '^https://snapshot\.ubuntu\.com/ubuntu/[0-9]{8}T[0-9]{6}Z/$' \
      || { echo >&2 "APT_SNAPSHOT_URL must be an immutable Ubuntu snapshot URL"; exit 64; }

RUN for value in \
      "${APT_CA_CERTIFICATES_VERSION}" \
      "${APT_LIBGOMP1_VERSION}" \
      "${APT_LIBSNDFILE1_VERSION}" \
      "${APT_TINI_VERSION}"; do \
      test -n "${value}" && ! printf '%s' "${value}" | grep -q '[[:space:]]' \
        || { echo >&2 "every APT package version build argument is required and whitespace-free"; exit 64; }; \
    done \
    && rm -f /etc/apt/sources.list.d/* \
    && printf '%s\n' \
      "deb [check-valid-until=no signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] ${APT_SNAPSHOT_URL} jammy main restricted universe multiverse" \
      "deb [check-valid-until=no signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] ${APT_SNAPSHOT_URL} jammy-updates main restricted universe multiverse" \
      "deb [check-valid-until=no signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] ${APT_SNAPSHOT_URL} jammy-security main restricted universe multiverse" \
      > /etc/apt/sources.list \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      "ca-certificates=${APT_CA_CERTIFICATES_VERSION}" \
      "libgomp1=${APT_LIBGOMP1_VERSION}" \
      "libsndfile1=${APT_LIBSNDFILE1_VERSION}" \
      "tini=${APT_TINI_VERSION}" \
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
    ASR_REQUIRE_CUDA=${ASR_REQUIRE_CUDA} \
    ASR_ENGINE=${ASR_ENGINE} \
    ASR_VAD_PROVIDER=${ASR_VAD_PROVIDER} \
    ASR_VAD_MODEL_PATH=${ASR_VAD_MODEL_PATH} \
    ASR_VAD_MODEL_SHA256=${ASR_VAD_MODEL_SHA256}

LABEL org.opencontainers.image.title="indicconformer-realtime-asr" \
      org.opencontainers.image.description="AI4Bharat IndicConformer ASR service; model weights mounted at runtime" \
      io.github.shivamb25.indicconformer.accelerator="${UV_EXTRA}" \
      io.github.shivamb25.indicconformer.engine="${ASR_ENGINE}" \
      io.github.shivamb25.indicconformer.vad="${ASR_VAD_PROVIDER}"

WORKDIR /app
RUN uv python install 3.11

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --extra "${UV_EXTRA}" --no-install-project

COPY --chown=asr:asr app/ ./app/
COPY --chown=asr:asr scripts/ ./scripts/

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3).read()"]

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--ws-ping-interval", "20", "--ws-ping-timeout", "20", "--timeout-graceful-shutdown", "150", "--no-access-log"]
