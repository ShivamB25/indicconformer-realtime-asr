from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from app.engine.errors import AssetDiscoveryError, ManifestVerificationError

_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ManifestFile:
    relative_path: str
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedManifest:
    repository: str
    revision: str
    digest: str
    files: Mapping[str, ManifestFile]

    def one_asset(self, filename: str) -> Path:
        matches = [
            item.path
            for item in self.files.values()
            if PurePosixPath(item.relative_path).name == filename
        ]
        if len(matches) != 1:
            locations = ", ".join(str(path) for path in matches) or "none"
            raise AssetDiscoveryError(
                f"verified manifest must contain exactly one {filename!r}; found {locations}"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class ModelAssets:
    encoder: Path
    ctc_decoder: Path
    joint_encoder: Path
    joint_predictor: Path
    joint_pre_net: Path
    joint_post_nets: Mapping[str, Path]
    language_masks: Path


def verify_manifest(
    model_dir: Path, manifest_path: Path, *, require_complete: bool
) -> VerifiedManifest:
    try:
        root = model_dir.resolve(strict=True)
    except OSError as exc:
        raise ManifestVerificationError(
            f"model directory is unavailable: {model_dir}: {exc}"
        ) from exc
    if not root.is_dir():
        raise ManifestVerificationError(f"model directory is not a directory: {root}")
    if manifest_path.is_symlink():
        raise ManifestVerificationError(f"model manifest cannot be a symlink: {manifest_path}")
    try:
        raw = manifest_path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestVerificationError(
            f"cannot read model manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise ManifestVerificationError("model manifest must be a JSON object")
    version = document.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != 1:
        raise ManifestVerificationError("model manifest schema_version must be exactly 1")
    repository = document.get("repository")
    revision = document.get("revision")
    entries = document.get("files")
    if not isinstance(repository, str) or not repository.strip():
        raise ManifestVerificationError("model manifest repository must be a non-empty string")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ManifestVerificationError(
            "model manifest revision must be a pinned 40-character lowercase commit hash"
        )
    if not isinstance(entries, list) or not entries:
        raise ManifestVerificationError("model manifest files must be a non-empty list")
    digest = hashlib.sha256(raw).hexdigest()

    files: dict[str, ManifestFile] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestVerificationError(f"manifest files[{index}] must be an object")
        relative = entry.get("path")
        expected_hash = entry.get("sha256")
        expected_size = entry.get("size")
        if not isinstance(relative, str) or not relative:
            raise ManifestVerificationError(f"manifest files[{index}].path must be non-empty")
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or not pure.parts
            or relative != pure.as_posix()
            or ".." in pure.parts
        ):
            raise ManifestVerificationError(f"manifest files[{index}].path is unsafe: {relative!r}")
        if relative in files:
            raise ManifestVerificationError(f"duplicate manifest path: {relative!r}")
        if not isinstance(expected_hash, str) or _SHA256.fullmatch(expected_hash) is None:
            raise ManifestVerificationError(
                f"manifest files[{index}].sha256 must be 64 lowercase hex characters"
            )
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            raise ManifestVerificationError(
                f"manifest files[{index}].size must be a nonnegative integer"
            )
        path = root.joinpath(*pure.parts)
        if path.is_symlink():
            raise ManifestVerificationError(f"manifest asset cannot be a symlink: {relative}")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
            actual_size = resolved.stat().st_size
        except (OSError, ValueError) as exc:
            raise ManifestVerificationError(
                f"manifest asset is missing or escapes model directory: {relative}: {exc}"
            ) from exc
        if not resolved.is_file():
            raise ManifestVerificationError(f"manifest asset is not a regular file: {relative}")
        if actual_size != expected_size:
            raise ManifestVerificationError(
                f"size mismatch for {relative}: expected {expected_size}, got {actual_size}; "
                "discard the snapshot and download the pinned revision again"
            )
        actual_hash = _sha256_file(resolved)
        if actual_hash != expected_hash:
            raise ManifestVerificationError(
                f"SHA-256 mismatch for {relative}: expected {expected_hash}, got {actual_hash}; "
                "discard the snapshot and download the pinned revision again"
            )
        files[relative] = ManifestFile(relative, resolved, expected_size, expected_hash)

    if require_complete:
        _verify_complete(
            manifest_path.parent / ".complete",
            repository=repository,
            revision=revision,
            manifest_digest=digest,
        )
    return VerifiedManifest(repository, revision, digest, files)


def discover_assets(manifest: VerifiedManifest, languages: tuple[str, ...]) -> ModelAssets:
    return ModelAssets(
        encoder=manifest.one_asset("encoder.onnx"),
        ctc_decoder=manifest.one_asset("ctc_decoder.onnx"),
        joint_encoder=manifest.one_asset("joint_enc.onnx"),
        joint_predictor=manifest.one_asset("joint_pred.onnx"),
        joint_pre_net=manifest.one_asset("joint_pre_net.onnx"),
        joint_post_nets={
            language: manifest.one_asset(f"joint_post_net_{language}.onnx")
            for language in languages
        },
        language_masks=manifest.one_asset("language_masks.json"),
    )


def _verify_complete(path: Path, *, repository: str, revision: str, manifest_digest: str) -> None:
    if path.is_symlink():
        raise ManifestVerificationError(f"completion marker cannot be a symlink: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestVerificationError(
            f"verified model completion marker is required at {path}: {exc}"
        ) from exc
    expected = {
        "schema_version": 1,
        "repository": repository,
        "revision": revision,
        "manifest_sha256": manifest_digest,
    }
    if not isinstance(document, dict) or any(
        document.get(key) != value for key, value in expected.items()
    ):
        raise ManifestVerificationError(
            f"completion marker {path} does not match the verified manifest; expected {expected!r}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
