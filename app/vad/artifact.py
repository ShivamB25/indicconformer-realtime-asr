"""Immutable metadata for the provisioned Silero VAD artifact."""

SILERO_VAD_VERSION = "6.2.1"
SILERO_VAD_REVISION = "7e30209a3e901f9842f81b225f3e93d8199902b1"
SILERO_VAD_MODEL_FILENAME = "silero_vad.onnx"
SILERO_VAD_MODEL_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/"
    f"{SILERO_VAD_REVISION}/src/silero_vad/data/{SILERO_VAD_MODEL_FILENAME}"
)
SILERO_VAD_MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
