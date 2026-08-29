"""Snapshot publishing and verification in scripts/download_model.py.

The downloader is the only component allowed to touch a remote repository, so
these tests pin the parts that must hold *without* one: revision/repository
validation, manifest construction, exhaustive verification of a published
directory, token-file hygiene, and refusing to publish anything it cannot
verify. The network is blocked by the suite-wide fixture.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import scripts._atomic_directory as atomic_directory
from tests.support.model_snapshot import (
    COMPLETE_NAME,
    LANGUAGES,
    MANIFEST_NAME,
    MANIFEST_SCHEMA_VERSION,
    OTHER_REVISION,
    REPOSITORY,
    REQUIRED_ASSETS,
    REVISION,
    ModelDownloadError,
    ModelValidationError,
    asset_bytes,
    build_manifest,
    canonical_json,
    download_model,
    downloader,
    manifest_entry,
    publish_snapshot,
    read_manifest,
    read_marker,
    replace_manifest,
    replace_marker,
    sha256_file,
    verify_model,
    write_assets,
    write_completion_metadata,
    write_token_file,
)


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "snapshot"
    publish_snapshot(root)
    return root


def rewrite(root: Path, mutate: Any, *, resync_marker: bool = True) -> None:
    document = read_manifest(root)
    mutate(document)
    replace_manifest(root, document, resync_marker=resync_marker)


class TestRevisionValidation:
    def test_a_full_commit_hash_is_accepted_and_lowercased(self) -> None:
        assert downloader.canonical_revision(REVISION.upper()) == REVISION

    @pytest.mark.parametrize(
        "revision",
        ["", "main", "v1", REVISION[:39], REVISION + "a", "g" * 40, " " + REVISION[1:]],
    )
    def test_anything_but_a_pinned_commit_hash_is_refused(self, revision: str) -> None:
        with pytest.raises(ModelValidationError, match="full 40-hex commit SHA"):
            downloader.canonical_revision(revision)


class TestRepositoryValidation:
    @pytest.mark.parametrize(
        "repository",
        [REPOSITORY, "owner/name", "Owner-1/model.v2", "a/b", "o_w/n-a.m_e"],
    )
    def test_owner_name_identifiers_are_accepted(self, repository: str) -> None:
        assert downloader.validate_repository(repository) == repository

    @pytest.mark.parametrize(
        "repository",
        ["", "owner", "owner/", "/name", "owner/name/extra", "-owner/name", "owner/.name", "o w/n"],
    )
    def test_anything_else_is_refused(self, repository: str) -> None:
        with pytest.raises(ModelValidationError, match="owner/name identifier"):
            downloader.validate_repository(repository)


class TestManifestConstruction:
    def test_every_downloaded_file_is_hashed(self, tmp_path: Path) -> None:
        root = tmp_path / "staging"
        root.mkdir()
        write_assets(root)
        manifest = build_manifest(root)

        assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert manifest["repository"] == REPOSITORY
        assert manifest["revision"] == REVISION
        assert [entry["path"] for entry in manifest["files"]] == sorted(REQUIRED_ASSETS)
        entry = manifest_entry(manifest, "assets/encoder.onnx")
        assert entry["size"] == len(asset_bytes("assets/encoder.onnx"))
        assert entry["sha256"] == sha256_file(root / "assets" / "encoder.onnx")

    def test_the_asset_list_covers_one_joint_post_net_per_language(self) -> None:
        post_nets = {
            name.removeprefix("assets/joint_post_net_").removesuffix(".onnx")
            for name in REQUIRED_ASSETS
            if "joint_post_net_" in name
        }
        assert post_nets == set(LANGUAGES)
        assert len(post_nets) == 22
        assert len(REQUIRED_ASSETS) == len(set(REQUIRED_ASSETS))
        assert all(not name.startswith("/") for name in REQUIRED_ASSETS)

    def test_a_missing_required_asset_stops_publication(self, tmp_path: Path) -> None:
        root = tmp_path / "staging"
        root.mkdir()
        write_assets(root, omit=("assets/joint_pred.onnx",))
        with pytest.raises(ModelValidationError, match="assets/joint_pred.onnx"):
            build_manifest(root)

    def test_symlinked_files_are_never_published(self, tmp_path: Path) -> None:
        root = tmp_path / "staging"
        root.mkdir()
        write_assets(root)
        target = root / "assets" / "encoder.onnx"
        payload = target.read_bytes()
        real = root / "assets" / "real.bin"
        real.write_bytes(payload)
        target.unlink()
        target.symlink_to(real)
        with pytest.raises(ModelValidationError, match="forbidden"):
            build_manifest(root)

    def test_symlinked_directories_are_never_published(self, tmp_path: Path) -> None:
        root = tmp_path / "staging"
        root.mkdir()
        write_assets(root)
        (root / "shortcut").symlink_to(root / "assets", target_is_directory=True)
        with pytest.raises(ModelValidationError, match="symbolic links are forbidden"):
            build_manifest(root)

    def test_extra_files_are_included_in_the_manifest(self, tmp_path: Path) -> None:
        root = tmp_path / "staging"
        root.mkdir()
        write_assets(root)
        (root / "config.json").write_bytes(b"{}\n")
        manifest = build_manifest(root)
        assert "config.json" in [entry["path"] for entry in manifest["files"]]

    def test_completion_metadata_is_canonical_and_write_once(self, tmp_path: Path) -> None:
        root = tmp_path / "staging"
        root.mkdir()
        write_assets(root)
        manifest = build_manifest(root)
        write_completion_metadata(root, manifest)

        assert (root / MANIFEST_NAME).read_bytes() == canonical_json(manifest)
        marker = read_marker(root)
        assert marker == {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "repository": REPOSITORY,
            "revision": REVISION,
            "manifest_sha256": sha256_file(root / MANIFEST_NAME),
        }
        with pytest.raises(FileExistsError):
            write_completion_metadata(root, manifest)

    def test_metadata_files_are_not_hashed_into_the_manifest(self, snapshot: Path) -> None:
        listed = {entry["path"] for entry in read_manifest(snapshot)["files"]}
        assert MANIFEST_NAME not in listed
        assert COMPLETE_NAME not in listed


class TestVerification:
    def test_a_published_snapshot_verifies(self, snapshot: Path) -> None:
        manifest = verify_model(snapshot)
        assert len(manifest["files"]) == len(REQUIRED_ASSETS)

    def test_an_uppercase_expected_revision_is_normalized(self, snapshot: Path) -> None:
        assert verify_model(snapshot, REPOSITORY, REVISION.upper())["revision"] == REVISION

    def test_the_expected_revision_must_match(self, snapshot: Path) -> None:
        with pytest.raises(ModelValidationError, match="revision does not match"):
            verify_model(snapshot, REPOSITORY, OTHER_REVISION)

    def test_the_expected_repository_must_match(self, snapshot: Path) -> None:
        with pytest.raises(ModelValidationError, match="repository does not match"):
            verify_model(snapshot, "someone/else", REVISION)

    def test_a_missing_directory_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(ModelValidationError, match="model directory does not exist"):
            verify_model(tmp_path / "absent")

    def test_a_symlinked_model_directory_is_refused(self, snapshot: Path, tmp_path: Path) -> None:
        link = tmp_path / "linked"
        link.symlink_to(snapshot, target_is_directory=True)
        with pytest.raises(ModelValidationError, match="must be a real directory"):
            verify_model(link)

    @pytest.mark.parametrize("name", [MANIFEST_NAME, COMPLETE_NAME])
    def test_missing_completion_metadata_is_reported(self, snapshot: Path, name: str) -> None:
        (snapshot / name).unlink()
        with pytest.raises(ModelValidationError, match="missing completion metadata"):
            verify_model(snapshot)

    @pytest.mark.parametrize("name", [MANIFEST_NAME, COMPLETE_NAME])
    def test_metadata_must_be_a_regular_file(self, snapshot: Path, name: str) -> None:
        target = snapshot / name
        payload = target.read_bytes()
        target.unlink()
        target.mkdir()
        assert stat.S_ISDIR(target.lstat().st_mode)
        assert payload
        with pytest.raises(ModelValidationError, match="not a regular file"):
            verify_model(snapshot)

    def test_a_manifest_with_duplicate_json_keys_is_refused(self, snapshot: Path) -> None:
        (snapshot / MANIFEST_NAME).write_text(
            '{"schema_version": 1, "schema_version": 1}', encoding="utf-8"
        )
        with pytest.raises(ModelValidationError, match="duplicate JSON key"):
            verify_model(snapshot)

    def test_an_unsupported_schema_is_refused(self, snapshot: Path) -> None:
        rewrite(snapshot, lambda document: document.__setitem__("schema_version", 2))
        with pytest.raises(ModelValidationError, match="unsupported model manifest schema"):
            verify_model(snapshot)

    def test_a_marker_that_does_not_match_the_manifest_is_refused(self, snapshot: Path) -> None:
        def reorder(document: dict[str, Any]) -> None:
            document["files"] = list(reversed(document["files"]))

        rewrite(snapshot, reorder, resync_marker=False)
        with pytest.raises(ModelValidationError, match="does not match the model manifest"):
            verify_model(snapshot)

    @pytest.mark.parametrize("digest", ["", "abc", "0" * 63, "A" * 64, None, 12])
    def test_a_marker_hash_of_the_wrong_shape_is_refused(
        self, snapshot: Path, digest: object
    ) -> None:
        marker = read_marker(snapshot)
        marker["manifest_sha256"] = digest
        replace_marker(snapshot, marker)
        with pytest.raises(ModelValidationError, match="invalid manifest hash"):
            verify_model(snapshot)

    def test_a_marker_schema_mismatch_is_refused(self, snapshot: Path) -> None:
        marker = read_marker(snapshot)
        marker["schema_version"] = 99
        replace_marker(snapshot, marker)
        with pytest.raises(ModelValidationError, match="unsupported completion marker schema"):
            verify_model(snapshot)

    @pytest.mark.parametrize("files", [[], {}, "assets/encoder.onnx", None])
    def test_the_file_list_must_be_a_non_empty_list(self, snapshot: Path, files: object) -> None:
        rewrite(snapshot, lambda document: document.__setitem__("files", files))
        with pytest.raises(ModelValidationError, match="non-empty files list"):
            verify_model(snapshot)

    @pytest.mark.parametrize(
        "entry",
        [
            {"path": "assets/encoder.onnx", "sha256": "0" * 64},
            {"path": "assets/encoder.onnx", "sha256": "0" * 64, "size": 1, "extra": True},
            {"pathname": "assets/encoder.onnx", "sha256": "0" * 64, "size": 1},
            "assets/encoder.onnx",
        ],
        ids=["missing_size", "extra_key", "wrong_key", "not_an_object"],
    )
    def test_every_entry_must_have_exactly_path_hash_and_size(
        self, snapshot: Path, entry: object
    ) -> None:
        rewrite(snapshot, lambda document: document["files"].append(entry))
        with pytest.raises(ModelValidationError, match="invalid file entry"):
            verify_model(snapshot)

    @pytest.mark.parametrize(
        "path_value",
        [
            "/etc/passwd",
            "../escape.onnx",
            "assets/../../escape.onnx",
            "./assets/encoder.onnx",
            "assets//encoder.onnx",
            "assets\\encoder.onnx",
            "",
            MANIFEST_NAME,
            COMPLETE_NAME,
        ],
    )
    def test_unsafe_or_denormalized_paths_are_refused(
        self, snapshot: Path, path_value: str
    ) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["files"][0]["path"] = path_value

        rewrite(snapshot, mutate)
        with pytest.raises(ModelValidationError):
            verify_model(snapshot)

    def test_duplicate_manifest_paths_are_refused(self, snapshot: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["files"].append(dict(document["files"][0]))

        rewrite(snapshot, mutate)
        with pytest.raises(ModelValidationError, match="duplicate manifest path"):
            verify_model(snapshot)

    @pytest.mark.parametrize("digest", ["", "abcdef", "0" * 63, "A" * 64, None, 5])
    def test_asset_hashes_must_be_lowercase_sha256(self, snapshot: Path, digest: object) -> None:
        def mutate(document: dict[str, Any]) -> None:
            manifest_entry(document, "assets/encoder.onnx")["sha256"] = digest

        rewrite(snapshot, mutate)
        with pytest.raises(ModelValidationError, match="invalid SHA-256"):
            verify_model(snapshot)

    @pytest.mark.parametrize("size", [-1, True, "42", None, 1.5])
    def test_asset_sizes_must_be_nonnegative_integers(self, snapshot: Path, size: object) -> None:
        def mutate(document: dict[str, Any]) -> None:
            manifest_entry(document, "assets/encoder.onnx")["size"] = size

        rewrite(snapshot, mutate)
        with pytest.raises(ModelValidationError, match="invalid size"):
            verify_model(snapshot)

    def test_a_size_mismatch_is_refused(self, snapshot: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            manifest_entry(document, "assets/encoder.onnx")["size"] = 1

        rewrite(snapshot, mutate)
        with pytest.raises(ModelValidationError, match="size mismatch"):
            verify_model(snapshot)

    def test_a_content_mismatch_is_refused(self, snapshot: Path) -> None:
        target = snapshot / "assets" / "encoder.onnx"
        target.write_bytes(bytes(reversed(target.read_bytes())))
        with pytest.raises(ModelValidationError, match="SHA-256 mismatch"):
            verify_model(snapshot)

    def test_a_missing_asset_is_refused(self, snapshot: Path) -> None:
        (snapshot / "assets" / "joint_enc.onnx").unlink()
        with pytest.raises(ModelValidationError, match="missing or escaped model file"):
            verify_model(snapshot)

    def test_a_symlinked_asset_is_refused(self, snapshot: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside.onnx"
        outside.write_bytes(asset_bytes("assets/encoder.onnx"))
        target = snapshot / "assets" / "encoder.onnx"
        target.unlink()
        target.symlink_to(outside)
        with pytest.raises(ModelValidationError, match="symbolic links are forbidden"):
            verify_model(snapshot)

    def test_a_required_asset_missing_from_the_manifest_is_refused(self, snapshot: Path) -> None:
        def drop(document: dict[str, Any]) -> None:
            document["files"] = [
                entry for entry in document["files"] if entry["path"] != "assets/encoder.onnx"
            ]

        rewrite(snapshot, drop)
        (snapshot / "assets" / "encoder.onnx").unlink()
        with pytest.raises(ModelValidationError, match="required model assets are missing"):
            verify_model(snapshot)

    def test_an_unmanifested_file_on_disk_is_refused(self, snapshot: Path) -> None:
        (snapshot / "assets" / "sneaky.onnx").write_bytes(b"unlisted\n")
        with pytest.raises(ModelValidationError, match="unmanifested: assets/sneaky.onnx"):
            verify_model(snapshot)

    def test_a_non_regular_file_on_disk_is_refused(self, snapshot: Path) -> None:
        fifo = snapshot / "assets" / "pipe"
        try:
            import os

            os.mkfifo(fifo)
        except (AttributeError, NotImplementedError, OSError):  # pragma: no cover
            pytest.skip("FIFOs are unavailable on this platform")
        with pytest.raises(ModelValidationError, match="non-regular model file"):
            verify_model(snapshot)

    def test_verification_does_not_modify_the_snapshot(self, snapshot: Path) -> None:
        before = {
            path.relative_to(snapshot).as_posix(): path.read_bytes()
            for path in sorted(snapshot.rglob("*"))
            if path.is_file()
        }
        verify_model(snapshot)
        after = {
            path.relative_to(snapshot).as_posix(): path.read_bytes()
            for path in sorted(snapshot.rglob("*"))
            if path.is_file()
        }
        assert before == after


class TestTokenHandling:
    def test_a_missing_token_file_is_reported(self, snapshot: Path, tmp_path: Path) -> None:
        with pytest.raises(ModelValidationError, match="token file"):
            download_model(snapshot, REPOSITORY, REVISION, tmp_path / "absent.token")

    @pytest.mark.parametrize(
        "content",
        ["", "   \n", "two tokens", "tok en", "x" * 4_097],
        ids=[
            "empty",
            "whitespace_only",
            "two_words",
            "embedded_space",
            "too_long",
        ],
    )
    def test_a_malformed_token_is_refused(
        self, snapshot: Path, tmp_path: Path, content: str
    ) -> None:
        token_file = tmp_path / "hf.token"
        token_file.write_text(content, encoding="utf-8")
        with pytest.raises(ModelValidationError, match="token file"):
            download_model(snapshot, REPOSITORY, REVISION, token_file)

    def test_a_symlinked_token_file_is_refused(self, snapshot: Path, tmp_path: Path) -> None:
        real = write_token_file(tmp_path / "real.token")
        link = tmp_path / "link.token"
        link.symlink_to(real)
        with pytest.raises(ModelValidationError, match="regular file, not a symbolic link"):
            download_model(snapshot, REPOSITORY, REVISION, link)


class TestPublishedDirectoryReuse:
    def test_an_existing_verified_snapshot_is_reused_without_downloading(
        self, snapshot: Path, tmp_path: Path
    ) -> None:
        """The network fixture would fail this test if a download were attempted."""

        token_file = write_token_file(tmp_path / "hf.token")
        download_model(snapshot, REPOSITORY, REVISION, token_file)
        assert verify_model(snapshot)["revision"] == REVISION

    def test_an_existing_snapshot_of_the_wrong_revision_is_refused(
        self, snapshot: Path, tmp_path: Path
    ) -> None:
        token_file = write_token_file(tmp_path / "hf.token")
        with pytest.raises(ModelValidationError, match="revision does not match"):
            download_model(snapshot, REPOSITORY, OTHER_REVISION, token_file)

    def test_a_corrupt_existing_snapshot_is_refused(self, snapshot: Path, tmp_path: Path) -> None:
        target = snapshot / "assets" / "encoder.onnx"
        target.write_bytes(bytes(reversed(target.read_bytes())))
        token_file = write_token_file(tmp_path / "hf.token")
        with pytest.raises(ModelValidationError, match="SHA-256 mismatch"):
            download_model(snapshot, REPOSITORY, REVISION, token_file)

    def test_an_unreachable_repository_leaves_no_partial_directory(self, tmp_path: Path) -> None:
        """Offline, the download fails and nothing half-written survives."""

        destination = tmp_path / "fresh-model"
        token_file = write_token_file(tmp_path / "hf.token")
        with pytest.raises(ModelDownloadError, match="model download failed"):
            download_model(destination, REPOSITORY, REVISION, token_file)

        assert not destination.exists()
        leftovers = [path.name for path in tmp_path.iterdir() if path.name.startswith(".")]
        assert leftovers == []


class TestAtomicPublication:
    def test_parent_fsync_failure_after_rename_is_propagated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        destination = tmp_path / "fresh-model"
        token_file = write_token_file(tmp_path / "hf.token")

        def snapshot_download(**arguments: Any) -> None:
            assert arguments["repo_id"] == REPOSITORY
            assert arguments["revision"] == REVISION
            write_assets(Path(arguments["local_dir"]))

        def fail_parent_fsync(path: Path) -> None:
            assert path == tmp_path
            raise OSError("parent fsync failed")

        monkeypatch.setitem(
            sys.modules,
            "huggingface_hub",
            SimpleNamespace(snapshot_download=snapshot_download),
        )
        monkeypatch.setattr(atomic_directory, "fsync_directory", fail_parent_fsync)

        with pytest.raises(OSError, match="parent fsync failed"):
            download_model(destination, REPOSITORY, REVISION, token_file)

        assert verify_model(destination)["revision"] == REVISION
        assert list(tmp_path.glob(".fresh-model.staging-*")) == []


class TestManifestJsonShape:
    def test_the_manifest_is_ascii_sorted_and_newline_terminated(self, snapshot: Path) -> None:
        raw = (snapshot / MANIFEST_NAME).read_bytes()
        assert raw.endswith(b"\n")
        assert raw.decode("ascii")
        document = json.loads(raw)
        assert list(document) == sorted(document)
        assert canonical_json(document) == raw

    def test_the_manifest_round_trips_through_verification(self, snapshot: Path) -> None:
        assert verify_model(snapshot) == read_manifest(snapshot)
