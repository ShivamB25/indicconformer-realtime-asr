"""Model manifest verification: the trust boundary in front of local weights.

Every rejection path is exercised with real files on disk, so a manifest that
points outside the model directory, disagrees on a hash or size, or is not
pinned to an immutable revision can never load. No network, no ONNX Runtime, no
real weights: the "assets" are tiny deterministic byte strings.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from app.engine.errors import AssetDiscoveryError, ManifestVerificationError, OrtEngineError
from app.engine.manifest import VerifiedManifest, discover_assets, verify_manifest
from tests.support.model_snapshot import (
    COMPLETE_NAME,
    LANGUAGES,
    MANIFEST_NAME,
    REPOSITORY,
    REQUIRED_ASSETS,
    REVISION,
    asset_bytes,
    canonical_json,
    manifest_entry,
    publish_snapshot,
    read_manifest,
    read_marker,
    replace_manifest,
    replace_marker,
)


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "snapshot"
    publish_snapshot(root)
    return root


def manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def verify(root: Path, *, require_complete: bool = True) -> VerifiedManifest:
    return verify_manifest(root, manifest_path(root), require_complete=require_complete)


def rewrite(root: Path, mutate: Any, *, resync_marker: bool = True) -> None:
    document = read_manifest(root)
    mutate(document)
    replace_manifest(root, document, resync_marker=resync_marker)


class TestAcceptance:
    def test_a_published_snapshot_verifies(self, snapshot: Path) -> None:
        verified = verify(snapshot)
        assert verified.repository == REPOSITORY
        assert verified.revision == REVISION
        assert len(verified.files) == len(REQUIRED_ASSETS)
        assert set(verified.files) == set(REQUIRED_ASSETS)

    def test_the_reported_digest_is_the_manifest_file_hash(self, snapshot: Path) -> None:
        expected = hashlib.sha256(manifest_path(snapshot).read_bytes()).hexdigest()
        assert verify(snapshot).digest == expected
        assert read_marker(snapshot)["manifest_sha256"] == expected

    def test_every_entry_records_its_verified_size_and_hash(self, snapshot: Path) -> None:
        verified = verify(snapshot)
        entry = verified.files["assets/encoder.onnx"]
        payload = asset_bytes("assets/encoder.onnx")
        assert entry.size == len(payload)
        assert entry.sha256 == hashlib.sha256(payload).hexdigest()
        assert entry.path == (snapshot / "assets" / "encoder.onnx").resolve()

    def test_completion_metadata_can_be_waived_explicitly(self, snapshot: Path) -> None:
        (snapshot / COMPLETE_NAME).unlink()
        assert verify(snapshot, require_complete=False).revision == REVISION

    def test_extra_unmanifested_files_do_not_block_startup(self, snapshot: Path) -> None:
        """The app verifies what it loads; the downloader owns exhaustiveness."""

        (snapshot / "README.md").write_bytes(b"not a model asset\n")
        assert verify(snapshot).repository == REPOSITORY

    def test_verification_is_repeatable(self, snapshot: Path) -> None:
        first = verify(snapshot)
        second = verify(snapshot)
        assert first.digest == second.digest
        assert set(first.files) == set(second.files)


class TestModelDirectoryRejection:
    def test_a_missing_model_directory_is_reported(self, tmp_path: Path) -> None:
        missing = tmp_path / "absent"
        with pytest.raises(ManifestVerificationError, match="model directory is unavailable"):
            verify_manifest(missing, missing / MANIFEST_NAME, require_complete=False)

    def test_a_file_cannot_stand_in_for_the_model_directory(self, tmp_path: Path) -> None:
        not_a_directory = tmp_path / "model"
        not_a_directory.write_bytes(b"")
        with pytest.raises(ManifestVerificationError, match="not a directory"):
            verify_manifest(not_a_directory, tmp_path / MANIFEST_NAME, require_complete=False)

    def test_a_symlinked_manifest_is_refused(self, snapshot: Path, tmp_path: Path) -> None:
        link = tmp_path / "linked-manifest.json"
        link.symlink_to(manifest_path(snapshot))
        with pytest.raises(ManifestVerificationError, match="cannot be a symlink"):
            verify_manifest(snapshot, link, require_complete=False)

    def test_a_missing_manifest_is_reported(self, snapshot: Path) -> None:
        manifest_path(snapshot).unlink()
        with pytest.raises(ManifestVerificationError, match="cannot read model manifest"):
            verify(snapshot, require_complete=False)

    @pytest.mark.parametrize("text", ["", "{", "[]", "null", '"manifest"', "17"])
    def test_a_manifest_that_is_not_a_json_object_is_refused(
        self, snapshot: Path, text: str
    ) -> None:
        manifest_path(snapshot).write_text(text, encoding="utf-8")
        with pytest.raises(ManifestVerificationError):
            verify(snapshot, require_complete=False)


class TestManifestHeaderRejection:
    @pytest.mark.parametrize("schema_version", [0, 2, "1", None, [1]])
    def test_only_schema_version_one_is_supported(
        self, snapshot: Path, schema_version: object
    ) -> None:
        rewrite(snapshot, lambda document: document.__setitem__("schema_version", schema_version))
        with pytest.raises(ManifestVerificationError, match="schema_version must be exactly 1"):
            verify(snapshot, require_complete=False)

    @pytest.mark.parametrize("repository", ["", "   ", None, 7, ["owner/name"]])
    def test_the_repository_must_be_a_non_empty_string(
        self, snapshot: Path, repository: object
    ) -> None:
        rewrite(snapshot, lambda document: document.__setitem__("repository", repository))
        with pytest.raises(ManifestVerificationError, match="repository must be a non-empty"):
            verify(snapshot, require_complete=False)

    @pytest.mark.parametrize(
        "revision",
        [
            "main",
            "v1.0.0",
            "abc123",
            REVISION[:-1],
            REVISION + "0",
            REVISION.upper(),
            "g" * 40,
            None,
            1,
        ],
        ids=[
            "branch_name",
            "tag",
            "short_hash",
            "one_char_short",
            "one_char_long",
            "uppercase",
            "not_hex",
            "null",
            "integer",
        ],
    )
    def test_the_revision_must_be_a_pinned_lowercase_commit_hash(
        self, snapshot: Path, revision: object
    ) -> None:
        rewrite(snapshot, lambda document: document.__setitem__("revision", revision))
        with pytest.raises(ManifestVerificationError, match="pinned 40-character lowercase"):
            verify(snapshot, require_complete=False)

    @pytest.mark.parametrize("files", [[], {}, None, "assets/encoder.onnx"])
    def test_the_file_list_must_be_a_non_empty_list(self, snapshot: Path, files: object) -> None:
        rewrite(snapshot, lambda document: document.__setitem__("files", files))
        with pytest.raises(ManifestVerificationError, match="files must be a non-empty list"):
            verify(snapshot, require_complete=False)


class TestManifestEntryRejection:
    @pytest.mark.parametrize("entry", ["assets/encoder.onnx", 12, None, ["a"]])
    def test_every_entry_must_be_an_object(self, snapshot: Path, entry: object) -> None:
        rewrite(snapshot, lambda document: document["files"].insert(0, entry))
        with pytest.raises(ManifestVerificationError, match=r"files\[0\] must be an object"):
            verify(snapshot, require_complete=False)

    @pytest.mark.parametrize("path_value", ["", None, 5, []])
    def test_every_entry_needs_a_path(self, snapshot: Path, path_value: object) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["files"][0]["path"] = path_value

        rewrite(snapshot, mutate)
        with pytest.raises(ManifestVerificationError, match="path must be non-empty"):
            verify(snapshot, require_complete=False)

    @pytest.mark.parametrize(
        "unsafe_path",
        [
            "/etc/passwd",
            "/assets/encoder.onnx",
            "../outside.onnx",
            "assets/../../outside.onnx",
            "assets/../assets/encoder.onnx",
        ],
    )
    def test_paths_may_not_escape_the_model_directory(
        self, snapshot: Path, unsafe_path: str
    ) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["files"][0]["path"] = unsafe_path

        rewrite(snapshot, mutate)
        with pytest.raises(ManifestVerificationError, match="is unsafe"):
            verify(snapshot, require_complete=False)

    def test_duplicate_paths_are_refused(self, snapshot: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["files"].append(dict(document["files"][0]))

        rewrite(snapshot, mutate)
        with pytest.raises(ManifestVerificationError, match="duplicate manifest path"):
            verify(snapshot, require_complete=False)

    @pytest.mark.parametrize(
        "digest",
        ["", "abc", "0" * 63, "0" * 65, "A" * 64, "z" * 64, None, 0],
        ids=[
            "empty",
            "too_short",
            "one_short",
            "one_long",
            "uppercase",
            "not_hex",
            "null",
            "integer",
        ],
    )
    def test_hashes_must_be_lowercase_sha256(self, snapshot: Path, digest: object) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["files"][0]["sha256"] = digest

        rewrite(snapshot, mutate)
        with pytest.raises(ManifestVerificationError, match="64 lowercase hex"):
            verify(snapshot, require_complete=False)

    @pytest.mark.parametrize("size", [-1, "12", None, True, 1.0])
    def test_sizes_must_be_nonnegative_integers(self, snapshot: Path, size: object) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["files"][0]["size"] = size

        rewrite(snapshot, mutate)
        with pytest.raises(ManifestVerificationError, match="must be a nonnegative integer"):
            verify(snapshot, require_complete=False)


class TestAssetContentRejection:
    def test_a_missing_asset_is_reported_by_name(self, snapshot: Path) -> None:
        (snapshot / "assets" / "encoder.onnx").unlink()
        with pytest.raises(ManifestVerificationError, match="assets/encoder.onnx"):
            verify(snapshot, require_complete=False)

    def test_a_modified_asset_fails_its_hash(self, snapshot: Path) -> None:
        target = snapshot / "assets" / "ctc_decoder.onnx"
        original = target.read_bytes()
        target.write_bytes(bytes(reversed(original)))
        assert target.stat().st_size == len(original)
        with pytest.raises(ManifestVerificationError, match="SHA-256 mismatch"):
            verify(snapshot, require_complete=False)

    def test_a_hash_mismatch_tells_the_operator_what_to_do(self, snapshot: Path) -> None:
        target = snapshot / "assets" / "encoder.onnx"
        target.write_bytes(target.read_bytes() + b"tampered")
        with pytest.raises(ManifestVerificationError, match="download the pinned revision again"):
            verify(snapshot, require_complete=False)

    def test_a_truncated_asset_fails_its_size(self, snapshot: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            manifest_entry(document, "assets/encoder.onnx")["size"] += 10

        rewrite(snapshot, mutate)
        with pytest.raises(ManifestVerificationError, match="size mismatch"):
            verify(snapshot, require_complete=False)

    def test_a_symlinked_asset_is_refused(self, snapshot: Path) -> None:
        target = snapshot / "assets" / "encoder.onnx"
        payload = target.read_bytes()
        elsewhere = snapshot / "assets" / "real_encoder.bin"
        elsewhere.write_bytes(payload)
        target.unlink()
        target.symlink_to(elsewhere)
        with pytest.raises(ManifestVerificationError, match="cannot be a symlink"):
            verify(snapshot, require_complete=False)

    def test_an_asset_reached_through_a_symlinked_directory_is_refused(
        self, snapshot: Path, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "encoder.onnx").write_bytes(asset_bytes("assets/encoder.onnx"))
        assets = snapshot / "assets"
        for child in assets.iterdir():
            child.unlink()
        assets.rmdir()
        assets.symlink_to(outside, target_is_directory=True)
        with pytest.raises(ManifestVerificationError, match="missing or escapes"):
            verify(snapshot, require_complete=False)

    def test_a_directory_cannot_impersonate_an_asset(self, snapshot: Path) -> None:
        target = snapshot / "assets" / "encoder.onnx"
        target.unlink()
        target.mkdir()
        with pytest.raises(ManifestVerificationError):
            verify(snapshot, require_complete=False)


class TestCompletionMarker:
    def test_a_missing_marker_blocks_a_complete_verification(self, snapshot: Path) -> None:
        (snapshot / COMPLETE_NAME).unlink()
        with pytest.raises(ManifestVerificationError, match="completion marker is required"):
            verify(snapshot)

    def test_a_symlinked_marker_is_refused(self, snapshot: Path, tmp_path: Path) -> None:
        marker = snapshot / COMPLETE_NAME
        elsewhere = tmp_path / "complete.json"
        elsewhere.write_bytes(marker.read_bytes())
        marker.unlink()
        marker.symlink_to(elsewhere)
        with pytest.raises(ManifestVerificationError, match="marker cannot be a symlink"):
            verify(snapshot)

    def test_a_marker_from_a_different_manifest_is_refused(self, snapshot: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            manifest_entry(document, "assets/language_masks.json")["size"] += 0
            document["files"].sort(key=lambda entry: entry["path"], reverse=True)

        rewrite(snapshot, mutate, resync_marker=False)
        with pytest.raises(ManifestVerificationError, match="does not match the verified manifest"):
            verify(snapshot)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"schema_version": 2},
            {"repository": "someone/else"},
            {"revision": "f" * 40},
            {"manifest_sha256": "0" * 64},
        ],
    )
    def test_a_marker_that_disagrees_with_the_manifest_is_refused(
        self, snapshot: Path, overrides: dict[str, object]
    ) -> None:
        marker = read_marker(snapshot)
        marker.update(overrides)
        replace_marker(snapshot, marker)
        with pytest.raises(ManifestVerificationError, match="does not match the verified manifest"):
            verify(snapshot)

    def test_a_corrupt_marker_is_refused(self, snapshot: Path) -> None:
        (snapshot / COMPLETE_NAME).write_text("{not json", encoding="utf-8")
        with pytest.raises(ManifestVerificationError, match="completion marker is required"):
            verify(snapshot)

    def test_the_marker_is_read_from_beside_the_manifest(self, snapshot: Path) -> None:
        """The marker location follows the manifest, not the model directory."""

        nested = snapshot / "metadata"
        nested.mkdir()
        moved_manifest = nested / MANIFEST_NAME
        moved_manifest.write_bytes((snapshot / MANIFEST_NAME).read_bytes())
        (snapshot / MANIFEST_NAME).unlink()

        with pytest.raises(ManifestVerificationError, match="completion marker is required"):
            verify_manifest(snapshot, moved_manifest, require_complete=True)

        (nested / COMPLETE_NAME).write_bytes((snapshot / COMPLETE_NAME).read_bytes())
        assert verify_manifest(snapshot, moved_manifest, require_complete=True).revision == REVISION


class TestAssetDiscovery:
    def test_every_required_asset_is_discovered(self, snapshot: Path) -> None:
        assets = discover_assets(verify(snapshot), LANGUAGES)
        assert assets.encoder.name == "encoder.onnx"
        assert assets.ctc_decoder.name == "ctc_decoder.onnx"
        assert assets.joint_encoder.name == "joint_enc.onnx"
        assert assets.joint_predictor.name == "joint_pred.onnx"
        assert assets.joint_pre_net.name == "joint_pre_net.onnx"
        assert assets.language_masks.name == "language_masks.json"
        assert set(assets.joint_post_nets) == set(LANGUAGES)
        assert len(assets.joint_post_nets) == 22

    def test_every_discovered_asset_exists_on_disk(self, snapshot: Path) -> None:
        assets = discover_assets(verify(snapshot), LANGUAGES)
        for path in (assets.encoder, assets.ctc_decoder, assets.language_masks):
            assert path.is_file()
        for path in assets.joint_post_nets.values():
            assert path.is_file()

    def test_an_unknown_language_cannot_be_discovered(self, snapshot: Path) -> None:
        with pytest.raises(AssertionError):
            self._discover_unknown(snapshot)

    def _discover_unknown(self, snapshot: Path) -> None:
        try:
            discover_assets(verify(snapshot), ("en",))
        except AssetDiscoveryError as exc:
            raise AssertionError(str(exc)) from exc

    def test_a_missing_asset_name_is_reported_as_discovery_failure(self, snapshot: Path) -> None:
        def drop_language_masks(document: dict[str, Any]) -> None:
            document["files"] = [
                entry
                for entry in document["files"]
                if entry["path"] != "assets/language_masks.json"
            ]

        rewrite(snapshot, drop_language_masks)
        (snapshot / "assets" / "language_masks.json").unlink()

        verified = verify(snapshot)
        assert "assets/language_masks.json" not in verified.files
        with pytest.raises(AssetDiscoveryError, match="language_masks.json"):
            discover_assets(verified, LANGUAGES)

    def test_an_ambiguous_asset_name_is_reported(self, tmp_path: Path) -> None:
        root = tmp_path / "ambiguous"
        publish_snapshot(root, extra_files={"extra/encoder.onnx": b"duplicate name\n"})
        verified = verify(root, require_complete=False)
        with pytest.raises(AssetDiscoveryError, match="exactly one 'encoder.onnx'"):
            discover_assets(verified, LANGUAGES)


class TestErrorHierarchy:
    @pytest.mark.parametrize("error", [ManifestVerificationError, AssetDiscoveryError])
    def test_manifest_errors_are_actionable_runtime_errors(self, error: type[Exception]) -> None:
        assert issubclass(error, OrtEngineError)
        assert issubclass(error, RuntimeError)

    def test_a_verified_manifest_is_immutable(self, snapshot: Path) -> None:
        verified = verify(snapshot)
        with pytest.raises(AttributeError):
            verified.revision = "f" * 40  # type: ignore[misc]

    def test_the_manifest_is_never_rewritten_by_verification(self, snapshot: Path) -> None:
        before = manifest_path(snapshot).read_bytes()
        verify(snapshot)
        assert manifest_path(snapshot).read_bytes() == before
        assert json.loads(before.decode("utf-8"))["revision"] == REVISION
        assert canonical_json(read_manifest(snapshot)) == before
