#!/usr/bin/env python3
"""Provision the pinned Silero VAD model and its license without baking weights into images."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, NoReturn
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

# Running this file by absolute path makes ``scripts/`` sys.path[0]. Add the
# repository/application root so the one canonical artifact definition is used.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.vad.artifact import (  # noqa: E402
    SILERO_VAD_MODEL_FILENAME,
    SILERO_VAD_MODEL_SHA256,
    SILERO_VAD_MODEL_URL,
    SILERO_VAD_REVISION,
)

MODEL_MAX_BYTES = 4 * 1024 * 1024
LICENSE_MAX_BYTES = 16 * 1024
SILERO_VAD_LICENSE_FILENAME = "LICENSE"
SILERO_VAD_LICENSE_URL = (
    f"https://raw.githubusercontent.com/snakers4/silero-vad/{SILERO_VAD_REVISION}/LICENSE"
)
SILERO_VAD_LICENSE_SHA256 = "2e63e9a38b6e8fc0c7bc37ce174caca1862870856c6daf5697cfb785e925520b"
_DOWNLOAD_CHUNK_BYTES = 64 * 1024


class VADModelValidationError(RuntimeError):
    """A local or downloaded VAD artifact is unsafe or does not match its pin."""


class VADModelDownloadError(RuntimeError):
    """A pinned VAD artifact could not be retrieved over HTTPS."""


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    filename: str
    url: str
    sha256: str
    max_bytes: int


MODEL_ARTIFACT = ArtifactSpec(
    SILERO_VAD_MODEL_FILENAME,
    SILERO_VAD_MODEL_URL,
    SILERO_VAD_MODEL_SHA256,
    MODEL_MAX_BYTES,
)
LICENSE_ARTIFACT = ArtifactSpec(
    SILERO_VAD_LICENSE_FILENAME,
    SILERO_VAD_LICENSE_URL,
    SILERO_VAD_LICENSE_SHA256,
    LICENSE_MAX_BYTES,
)


def _validate_spec(spec: ArtifactSpec) -> None:
    if (
        not spec.filename
        or Path(spec.filename).name != spec.filename
        or spec.filename in (".", "..")
    ):
        raise VADModelValidationError("artifact filename must be one safe path component")
    parsed = urlsplit(spec.url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "raw.githubusercontent.com"
        or parsed.query
        or parsed.fragment
        or SILERO_VAD_REVISION not in parsed.path.split("/")
    ):
        raise VADModelValidationError("artifact URL must be HTTPS and pinned to the exact revision")
    if len(spec.sha256) != 64 or any(
        character not in "0123456789abcdef" for character in spec.sha256
    ):
        raise VADModelValidationError("artifact SHA-256 must be 64 lowercase hex characters")
    if isinstance(spec.max_bytes, bool) or spec.max_bytes <= 0:
        raise VADModelValidationError("artifact size bound must be positive")


def _artifact_specs() -> tuple[ArtifactSpec, ArtifactSpec]:
    specs = (MODEL_ARTIFACT, LICENSE_ARTIFACT)
    for spec in specs:
        _validate_spec(spec)
    if specs[0].filename == specs[1].filename:
        raise VADModelValidationError("artifact filenames must be unique")
    return specs


def _sha256_bounded(path: Path, max_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as source:
            while chunk := source.read(_DOWNLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    raise VADModelValidationError(
                        f"{path.name} exceeds the {max_bytes}-byte size limit"
                    )
                digest.update(chunk)
    except VADModelValidationError:
        raise
    except OSError as exc:
        raise VADModelValidationError(f"cannot read {path.name}: {type(exc).__name__}") from None
    return digest.hexdigest(), size


def _regular_file(directory: Path, spec: ArtifactSpec) -> Path:
    path = directory / spec.filename
    try:
        mode = path.lstat().st_mode
    except OSError:
        raise VADModelValidationError(f"missing VAD artifact: {spec.filename}") from None
    if not stat.S_ISREG(mode):
        raise VADModelValidationError(f"VAD artifact must be a regular file: {spec.filename}")
    return path


def verify_vad_model(destination: Path) -> None:
    """Verify the complete pinned model directory without any network access."""

    specs = _artifact_specs()
    try:
        mode = destination.lstat().st_mode
    except OSError:
        raise VADModelValidationError("VAD model directory does not exist") from None
    if not stat.S_ISDIR(mode):
        raise VADModelValidationError("VAD model path must be a real directory")

    try:
        names = {entry.name for entry in os.scandir(destination)}
    except OSError as exc:
        raise VADModelValidationError(
            f"cannot inspect VAD model directory: {type(exc).__name__}"
        ) from None
    expected_names = {spec.filename for spec in specs}
    if names != expected_names:
        raise VADModelValidationError("VAD model directory is incomplete or contains extra files")

    for spec in specs:
        path = _regular_file(destination, spec)
        digest, _ = _sha256_bounded(path, spec.max_bytes)
        if digest != spec.sha256:
            raise VADModelValidationError(f"SHA-256 mismatch for {spec.filename}")


def _response_stream(response: object) -> BinaryIO:
    if not hasattr(response, "read"):
        raise VADModelDownloadError("artifact response is not a byte stream")
    return response  # type: ignore[return-value]


def _download_artifact(spec: ArtifactSpec, destination: Path) -> None:
    request = Request(
        spec.url,
        headers={
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
            "User-Agent": "indicconformer-vad-model-provisioner/1",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - exact validated HTTPS URL
            final_url = response.geturl()
            if final_url != spec.url or urlsplit(final_url).scheme != "https":
                raise VADModelDownloadError("artifact download redirected away from the pinned URL")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except (TypeError, ValueError):
                    raise VADModelDownloadError(
                        "artifact response has an invalid Content-Length"
                    ) from None
                if declared_size < 0 or declared_size > spec.max_bytes:
                    raise VADModelDownloadError(
                        f"{spec.filename} exceeds the {spec.max_bytes}-byte size limit"
                    )

            digest = hashlib.sha256()
            size = 0
            with destination.open("xb") as target:
                stream = _response_stream(response)
                while chunk := stream.read(_DOWNLOAD_CHUNK_BYTES):
                    if not isinstance(chunk, bytes):
                        raise VADModelDownloadError("artifact response yielded non-byte content")
                    size += len(chunk)
                    if size > spec.max_bytes:
                        raise VADModelDownloadError(
                            f"{spec.filename} exceeds the {spec.max_bytes}-byte size limit"
                        )
                    target.write(chunk)
                    digest.update(chunk)
                target.flush()
                os.fsync(target.fileno())
    except (VADModelDownloadError, VADModelValidationError):
        raise
    except Exception as exc:
        raise VADModelDownloadError(
            f"download failed for {spec.filename}: {type(exc).__name__}"
        ) from None

    if digest.hexdigest() != spec.sha256:
        raise VADModelValidationError(f"SHA-256 mismatch for {spec.filename}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def provision_vad_model(destination: Path, *, offline: bool = False) -> None:
    """Verify an existing pin or download and atomically publish it."""

    specs = _artifact_specs()
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        verify_vad_model(destination)
        return
    if offline:
        raise VADModelValidationError("VAD model is unavailable in offline verification mode")

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent_mode = destination.parent.lstat().st_mode
    except OSError:
        raise VADModelValidationError("VAD model parent directory cannot be inspected") from None
    if not stat.S_ISDIR(parent_mode):
        raise VADModelValidationError("VAD model parent path must be a real directory")

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    published = False
    try:
        for spec in specs:
            _download_artifact(spec, staging / spec.filename)
        verify_vad_model(staging)
        _fsync_directory(staging)
        try:
            os.rename(staging, destination)
            published = True
            _fsync_directory(destination.parent)
        except OSError:
            if destination.exists() and not destination.is_symlink():
                verify_vad_model(destination)
            else:
                raise
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="final VAD artifact directory",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="verify an existing directory and never attempt network access",
    )
    return parser


def _fail(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(message)


def main() -> None:
    parser = _parser()
    arguments = parser.parse_args()
    try:
        provision_vad_model(arguments.output_dir, offline=arguments.offline)
    except (VADModelValidationError, VADModelDownloadError, OSError) as exc:
        _fail(parser, str(exc))
    print(f"Silero VAD model ready at {arguments.output_dir}")


if __name__ == "__main__":
    main()
