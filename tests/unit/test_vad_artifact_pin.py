"""Immutable public provenance for the provisioned Silero VAD artifact."""

from app.vad.artifact import (
    SILERO_VAD_MODEL_FILENAME,
    SILERO_VAD_MODEL_SHA256,
    SILERO_VAD_MODEL_URL,
    SILERO_VAD_REVISION,
    SILERO_VAD_VERSION,
)
from scripts.download_vad_model import (
    SILERO_VAD_LICENSE_SHA256,
    SILERO_VAD_LICENSE_URL,
)


def test_silero_v621_artifacts_are_pinned_to_the_reviewed_commit() -> None:
    assert SILERO_VAD_VERSION == "6.2.1"
    assert SILERO_VAD_REVISION == "7e30209a3e901f9842f81b225f3e93d8199902b1"
    assert SILERO_VAD_MODEL_FILENAME == "silero_vad.onnx"
    assert SILERO_VAD_MODEL_URL == (
        "https://raw.githubusercontent.com/snakers4/silero-vad/"
        "7e30209a3e901f9842f81b225f3e93d8199902b1/"
        "src/silero_vad/data/silero_vad.onnx"
    )
    assert SILERO_VAD_MODEL_SHA256 == (
        "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
    )
    assert SILERO_VAD_LICENSE_URL == (
        "https://raw.githubusercontent.com/snakers4/silero-vad/"
        "7e30209a3e901f9842f81b225f3e93d8199902b1/LICENSE"
    )
    assert SILERO_VAD_LICENSE_SHA256 == (
        "2e63e9a38b6e8fc0c7bc37ce174caca1862870856c6daf5697cfb785e925520b"
    )
