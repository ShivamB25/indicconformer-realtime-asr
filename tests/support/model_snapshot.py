"""Helpers for building and corrupting local model snapshots.

The download/verify tooling lives in ``scripts/`` and is executed as a script,
so it is loaded here by path instead of by package import. Nothing in this
module touches the network: snapshots are tiny deterministic byte strings whose
only purpose is to exercise manifest, hash, and path verification.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

REPOSITORY = "ai4bharat/indic-conformer-600m-multilingual"
REVISION = "0123456789abcdef0123456789abcdef01234567"
OTHER_REVISION = "89abcdef0123456789abcdef0123456789abcdef"

_MODULE_NAME = "download_model"
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_download_model() -> ModuleType:
    cached = sys.modules.get(_MODULE_NAME)
    if cached is not None:
        return cached
    script = _SCRIPTS_DIR / f"{_MODULE_NAME}.py"
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, script)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    # Registered under its script name so scripts/verify_model.py can import it.
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


downloader = _load_download_model()

MANIFEST_NAME = cast(str, downloader.MANIFEST_NAME)
COMPLETE_NAME = cast(str, downloader.COMPLETE_NAME)
MANIFEST_SCHEMA_VERSION = cast(int, downloader.MANIFEST_SCHEMA_VERSION)
REQUIRED_ASSETS = cast("tuple[str, ...]", downloader.REQUIRED_ASSETS)
LANGUAGES = cast("tuple[str, ...]", downloader.LANGUAGES)
ModelValidationError = cast("type[Exception]", downloader.ModelValidationError)
canonical_revision = downloader.canonical_revision
validate_repository = downloader.validate_repository
ModelDownloadError = cast("type[Exception]", downloader.ModelDownloadError)

SCRIPTS_DIR = _SCRIPTS_DIR
DOWNLOAD_CLI = _SCRIPTS_DIR / "download_model.py"
VERIFY_CLI = _SCRIPTS_DIR / "verify_model.py"


def asset_bytes(relative_path: str) -> bytes:
    """Deterministic placeholder content; not a model and not model output."""

    return f"placeholder-asset:{relative_path}\n".encode("ascii")


def write_assets(root: Path, *, omit: tuple[str, ...] = ()) -> None:
    """Create every required asset path with deterministic placeholder bytes."""

    for relative_path in REQUIRED_ASSETS:
        if relative_path in omit:
            continue
        target = root.joinpath(*relative_path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(asset_bytes(relative_path))


def build_manifest(
    root: Path, repository: str = REPOSITORY, revision: str = REVISION
) -> dict[str, Any]:
    return cast("dict[str, Any]", downloader.build_manifest(root, repository, revision))


def write_completion_metadata(root: Path, manifest: dict[str, Any]) -> None:
    downloader.write_completion_metadata(root, manifest)


def verify_model(
    root: Path, repository: str = REPOSITORY, revision: str = REVISION
) -> dict[str, Any]:
    return cast("dict[str, Any]", downloader.verify_model(root, repository, revision))


def download_model(destination: Path, repository: str, revision: str, token_file: Path) -> None:
    downloader.download_model(destination, repository, revision, token_file)


def sha256_file(path: Path) -> str:
    return cast(str, downloader.sha256_file(path))


def publish_snapshot(
    root: Path,
    *,
    repository: str = REPOSITORY,
    revision: str = REVISION,
    omit: tuple[str, ...] = (),
    extra_files: dict[str, bytes] | None = None,
) -> dict[str, Any]:
    """Create a fully published, verifiable snapshot directory."""

    root.mkdir(parents=True, exist_ok=True)
    write_assets(root, omit=omit)
    for relative_path, content in (extra_files or {}).items():
        target = root.joinpath(*relative_path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    manifest = build_manifest(root, repository, revision)
    write_completion_metadata(root, manifest)
    return manifest


def canonical_json(value: dict[str, Any]) -> bytes:
    """Byte-for-byte the encoding the downloader uses for its metadata."""

    text = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return (text + "\n").encode("utf-8")


def read_manifest(root: Path) -> dict[str, Any]:
    raw = (root / MANIFEST_NAME).read_text(encoding="utf-8")
    return cast("dict[str, Any]", json.loads(raw))


def read_marker(root: Path) -> dict[str, Any]:
    raw = (root / COMPLETE_NAME).read_text(encoding="utf-8")
    return cast("dict[str, Any]", json.loads(raw))


def replace_manifest(root: Path, manifest: dict[str, Any], *, resync_marker: bool = True) -> None:
    """Overwrite the manifest, optionally leaving the marker hash stale."""

    marker_path = root / COMPLETE_NAME
    marker_backup = marker_path.read_bytes() if marker_path.exists() else None
    (root / MANIFEST_NAME).unlink(missing_ok=True)
    marker_path.unlink(missing_ok=True)
    if resync_marker:
        write_completion_metadata(root, manifest)
        return
    (root / MANIFEST_NAME).write_bytes(canonical_json(manifest))
    if marker_backup is not None:
        marker_path.write_bytes(marker_backup)


def replace_marker(root: Path, marker: dict[str, Any]) -> None:
    (root / COMPLETE_NAME).write_bytes(canonical_json(marker))


def manifest_entry(manifest: dict[str, Any], relative_path: str) -> dict[str, Any]:
    for entry in cast("list[dict[str, Any]]", manifest["files"]):
        if entry["path"] == relative_path:
            return entry
    raise AssertionError(f"{relative_path} is not in the manifest")


def write_token_file(path: Path, token: str = "hf_deterministic_test_token") -> Path:
    path.write_text(token, encoding="utf-8")
    return path


__all__ = [
    "COMPLETE_NAME",
    "DOWNLOAD_CLI",
    "LANGUAGES",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA_VERSION",
    "OTHER_REVISION",
    "REPOSITORY",
    "REQUIRED_ASSETS",
    "REVISION",
    "SCRIPTS_DIR",
    "VERIFY_CLI",
    "ModelDownloadError",
    "ModelValidationError",
    "asset_bytes",
    "build_manifest",
    "canonical_json",
    "download_model",
    "downloader",
    "manifest_entry",
    "publish_snapshot",
    "read_manifest",
    "read_marker",
    "replace_manifest",
    "replace_marker",
    "sha256_file",
    "verify_model",
    "write_assets",
    "write_completion_metadata",
    "write_token_file",
]
