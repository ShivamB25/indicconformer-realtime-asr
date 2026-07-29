#!/usr/bin/env python3
"""Download an immutable gated model snapshot into an atomically published directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

LANGUAGES: tuple[str, ...] = (
    "as",
    "bn",
    "brx",
    "doi",
    "gu",
    "hi",
    "kn",
    "kok",
    "ks",
    "mai",
    "ml",
    "mni",
    "mr",
    "ne",
    "or",
    "pa",
    "sa",
    "sat",
    "sd",
    "ta",
    "te",
    "ur",
)
REQUIRED_ASSETS: tuple[str, ...] = (
    "config.json",
    "model_onnx.py",
    "assets/encoder.onnx",
    "assets/ctc_decoder.onnx",
    "assets/joint_enc.onnx",
    "assets/joint_pre_net.onnx",
    "assets/joint_pred.onnx",
    "assets/language_masks.json",
    *(f"assets/joint_post_net_{language}.onnx" for language in LANGUAGES),
)
MANIFEST_NAME = "model-manifest.json"
COMPLETE_NAME = ".complete"
MANIFEST_SCHEMA_VERSION = 1
_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_REPOSITORY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class ModelValidationError(RuntimeError):
    """The local model snapshot is incomplete, unsafe, or inconsistent."""


class ModelDownloadError(RuntimeError):
    """The remote model snapshot could not be downloaded safely."""


def canonical_revision(revision: str) -> str:
    """Validate and normalize an immutable Git commit revision."""
    if _REVISION_RE.fullmatch(revision) is None:
        raise ModelValidationError("revision must be a full 40-hex commit SHA")
    return revision.lower()


def validate_repository(repository: str) -> str:
    """Validate the Hugging Face owner/name repository identifier."""
    if _REPOSITORY_RE.fullmatch(repository) is None:
        raise ModelValidationError("repository must be an owner/name identifier")
    return repository


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as source:
            value = json.load(source, object_pairs_hook=_reject_duplicate_keys)
    except ModelValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelValidationError(f"cannot read {path.name}: {type(exc).__name__}") from None
    if not isinstance(value, dict):
        raise ModelValidationError(f"{path.name} must contain a JSON object")
    return value


def _safe_relative_path(raw_path: Any) -> PurePosixPath:
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ModelValidationError("manifest contains an invalid file path")
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or raw_path != relative.as_posix():
        raise ModelValidationError(f"manifest path is not normalized: {raw_path}")
    if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        raise ModelValidationError(f"manifest path escapes model directory: {raw_path}")
    if relative.parts[0] in (MANIFEST_NAME, COMPLETE_NAME):
        raise ModelValidationError(f"manifest cannot hash reserved path: {raw_path}")
    return relative


def _regular_file_inside(root: Path, relative: PurePosixPath) -> Path:
    candidate = root.joinpath(*relative.parts)
    current = root
    try:
        for part in relative.parts:
            current = current / part
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ModelValidationError(f"symbolic links are forbidden: {relative.as_posix()}")
        if not stat.S_ISREG(candidate.lstat().st_mode):
            raise ModelValidationError(
                f"manifest path is not a regular file: {relative.as_posix()}"
            )
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ModelValidationError:
        raise
    except (FileNotFoundError, OSError, ValueError):
        raise ModelValidationError(
            f"missing or escaped model file: {relative.as_posix()}"
        ) from None
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        for name in directory_names:
            path = base / name
            if path.is_symlink():
                relative = path.relative_to(root).as_posix()
                raise ModelValidationError(f"symbolic links are forbidden: {relative}")
        for name in file_names:
            path = base / name
            relative = path.relative_to(root).as_posix()
            if relative in (MANIFEST_NAME, COMPLETE_NAME):
                continue
            try:
                mode = path.lstat().st_mode
            except OSError:
                raise ModelValidationError(f"cannot inspect model file: {relative}") from None
            if not stat.S_ISREG(mode):
                raise ModelValidationError(f"non-regular model file is forbidden: {relative}")
            files[relative] = path
    return files


def build_manifest(root: Path, repository: str, revision: str) -> dict[str, Any]:
    """Hash every downloaded file and create the canonical model manifest."""
    repository = validate_repository(repository)
    revision = canonical_revision(revision)
    files = _snapshot_files(root)
    missing = sorted(set(REQUIRED_ASSETS).difference(files))
    if missing:
        raise ModelValidationError(f"required model assets are missing: {', '.join(missing)}")
    entries = [
        {
            "path": relative,
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        for relative, path in sorted(files.items())
    ]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "repository": repository,
        "revision": revision,
        "files": entries,
    }


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _write_new_file(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_completion_metadata(root: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest and, last, the completion marker into a staging tree."""
    manifest_bytes = _canonical_json(manifest)
    _write_new_file(root / MANIFEST_NAME, manifest_bytes)
    marker = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    _write_new_file(root / COMPLETE_NAME, _canonical_json(marker))


def verify_model(root: Path, repository: str, revision: str) -> dict[str, Any]:
    """Fully verify a published model snapshot without trusting its manifest paths."""
    repository = validate_repository(repository)
    revision = canonical_revision(revision)
    try:
        root_mode = root.lstat().st_mode
    except OSError:
        raise ModelValidationError(f"model directory does not exist: {root}") from None
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise ModelValidationError("model directory must be a real directory")

    manifest_path = root / MANIFEST_NAME
    marker_path = root / COMPLETE_NAME
    for metadata_path in (manifest_path, marker_path):
        try:
            mode = metadata_path.lstat().st_mode
        except OSError:
            raise ModelValidationError(
                f"missing completion metadata: {metadata_path.name}"
            ) from None
        if not stat.S_ISREG(mode):
            raise ModelValidationError(
                f"completion metadata is not a regular file: {metadata_path.name}"
            )

    manifest = _load_json(manifest_path)
    marker = _load_json(marker_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ModelValidationError("unsupported model manifest schema")
    if manifest.get("repository") != repository or marker.get("repository") != repository:
        raise ModelValidationError("model repository does not match the expected repository")
    if manifest.get("revision") != revision or marker.get("revision") != revision:
        raise ModelValidationError("model revision does not match the expected immutable revision")
    if marker.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ModelValidationError("unsupported completion marker schema")

    manifest_hash = sha256_file(manifest_path)
    marker_hash = marker.get("manifest_sha256")
    if not isinstance(marker_hash, str) or not _SHA256_RE.fullmatch(marker_hash):
        raise ModelValidationError("completion marker contains an invalid manifest hash")
    if marker_hash != manifest_hash:
        raise ModelValidationError("completion marker does not match the model manifest")

    raw_entries = manifest.get("files")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ModelValidationError("model manifest must contain a non-empty files list")
    listed_paths: set[str] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict) or set(raw_entry) != {"path", "sha256", "size"}:
            raise ModelValidationError("model manifest contains an invalid file entry")
        relative = _safe_relative_path(raw_entry["path"])
        relative_text = relative.as_posix()
        if relative_text in listed_paths:
            raise ModelValidationError(f"duplicate manifest path: {relative_text}")
        listed_paths.add(relative_text)
        expected_hash = raw_entry["sha256"]
        expected_size = raw_entry["size"]
        if not isinstance(expected_hash, str) or _SHA256_RE.fullmatch(expected_hash) is None:
            raise ModelValidationError(f"invalid SHA-256 for model file: {relative_text}")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise ModelValidationError(f"invalid size for model file: {relative_text}")
        path = _regular_file_inside(root, relative)
        try:
            actual_size = path.stat().st_size
            actual_hash = sha256_file(path)
        except OSError:
            raise ModelValidationError(f"cannot read model file: {relative_text}") from None
        if actual_size != expected_size:
            raise ModelValidationError(f"size mismatch for model file: {relative_text}")
        if actual_hash != expected_hash:
            raise ModelValidationError(f"SHA-256 mismatch for model file: {relative_text}")

    missing_required = sorted(set(REQUIRED_ASSETS).difference(listed_paths))
    if missing_required:
        raise ModelValidationError(
            f"required model assets are missing: {', '.join(missing_required)}"
        )
    actual_paths = set(_snapshot_files(root))
    if actual_paths != listed_paths:
        missing = sorted(listed_paths.difference(actual_paths))
        unexpected = sorted(actual_paths.difference(listed_paths))
        detail = "; ".join(
            part
            for part in (
                f"missing: {', '.join(missing)}" if missing else "",
                f"unmanifested: {', '.join(unexpected)}" if unexpected else "",
            )
            if part
        )
        raise ModelValidationError(f"model directory does not match manifest ({detail})")
    return manifest


def _read_token(token_file: Path) -> str:
    try:
        mode = token_file.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ModelValidationError("token file must be a regular file, not a symbolic link")
        token = token_file.read_text(encoding="utf-8").strip()
    except ModelValidationError:
        raise
    except (OSError, UnicodeError):
        raise ModelValidationError("token file cannot be read") from None
    if not token or len(token) > 4096 or any(character.isspace() for character in token):
        raise ModelValidationError("token file must contain one non-empty token")
    return token


def _remove_huggingface_metadata(staging: Path) -> None:
    metadata = staging / ".cache"
    if metadata.exists() or metadata.is_symlink():
        if metadata.is_symlink() or metadata.is_file():
            metadata.unlink()
        else:
            shutil.rmtree(metadata)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_payload_tree(root: Path) -> None:
    """Persist every downloaded payload and directory before publication."""
    for path in _snapshot_files(root).values():
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directories = [Path(directory) for directory, _, _ in os.walk(root)]
    for directory in reversed(directories):
        _fsync_directory(directory)


def download_model(destination: Path, repository: str, revision: str, token_file: Path) -> None:
    """Download, verify, and atomically publish an immutable model snapshot."""
    repository = validate_repository(repository)
    revision = canonical_revision(revision)
    token = _read_token(token_file)
    destination = destination.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() or destination.is_symlink():
        verify_model(destination, repository, revision)
        return

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    published = False
    try:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repository,
                revision=revision,
                token=token,
                local_dir=staging,
                allow_patterns=[
                    "assets/**",
                    "config.json",
                    "model_onnx.py",
                    "model_onnx_1b_batched_rnnt.py",
                    "model_ts.py",
                ],
            )
        except Exception as exc:
            raise ModelDownloadError(f"model download failed: {type(exc).__name__}") from None
        finally:
            token = ""

        _remove_huggingface_metadata(staging)
        _fsync_payload_tree(staging)
        manifest = build_manifest(staging, repository, revision)
        write_completion_metadata(staging, manifest)
        verify_model(staging, repository, revision)
        _fsync_directory(staging)
        try:
            os.rename(staging, destination)
            published = True
            _fsync_directory(destination.parent)
        except OSError:
            if destination.exists() and not destination.is_symlink():
                verify_model(destination, repository, revision)
            else:
                raise
    finally:
        token = ""
        if not published and staging.exists():
            shutil.rmtree(staging)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, help="Hugging Face owner/name repository")
    parser.add_argument("--revision", required=True, help="full immutable 40-hex commit SHA")
    parser.add_argument(
        "--token-file", required=True, type=Path, help="path to the gated-repository token"
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="final model directory")
    return parser


def _fail(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(message)


def main() -> None:
    parser = _parser()
    arguments = parser.parse_args()
    try:
        download_model(
            destination=arguments.output_dir,
            repository=arguments.repository,
            revision=arguments.revision,
            token_file=arguments.token_file,
        )
    except (ModelValidationError, ModelDownloadError, OSError) as exc:
        _fail(parser, str(exc))
    print(f"model snapshot ready at {arguments.output_dir}")


if __name__ == "__main__":
    main()
