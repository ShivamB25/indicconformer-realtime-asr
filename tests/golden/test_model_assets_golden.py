"""Golden inventory of the pinned model snapshot.

Only names, counts, and formats are recorded: which files a snapshot must
contain, what a manifest entry looks like, and how a revision is spelled. No
weights, no real hashes, and no claim about what the model produces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.types import SUPPORTED_LANGUAGES
from tests.support.golden import load_golden
from tests.support.model_snapshot import (
    COMPLETE_NAME,
    LANGUAGES,
    MANIFEST_NAME,
    MANIFEST_SCHEMA_VERSION,
    REQUIRED_ASSETS,
    REVISION,
    ModelValidationError,
    build_manifest,
    canonical_revision,
    publish_snapshot,
    validate_repository,
)

GOLDEN = load_golden("model_assets.json")


class TestAssetInventory:
    def test_the_asset_count_is_pinned(self) -> None:
        assert GOLDEN["asset_count"] == len(REQUIRED_ASSETS)

    def test_the_shared_assets_are_exactly_the_language_independent_ones(self) -> None:
        shared = [name for name in REQUIRED_ASSETS if "joint_post_net_" not in name]
        assert shared == GOLDEN["shared_assets"]

    def test_one_language_asset_exists_per_supported_language(self) -> None:
        template = GOLDEN["per_language_asset_template"]
        expected = {template.format(language=language) for language in SUPPORTED_LANGUAGES}
        assert {name for name in REQUIRED_ASSETS if "joint_post_net_" in name} == expected

    def test_the_downloader_language_list_matches_the_service_contract(self) -> None:
        assert LANGUAGES == SUPPORTED_LANGUAGES

    def test_asset_paths_are_relative_posix_paths(self) -> None:
        for name in REQUIRED_ASSETS:
            assert not name.startswith("/")
            assert "\\" not in name
            assert ".." not in name.split("/")


class TestManifestShape:
    def test_the_reserved_names_are_pinned(self) -> None:
        assert GOLDEN["manifest_name"] == MANIFEST_NAME
        assert GOLDEN["completion_marker"] == COMPLETE_NAME
        assert GOLDEN["manifest_schema_version"] == MANIFEST_SCHEMA_VERSION

    def test_a_built_manifest_has_exactly_the_recorded_fields(self, tmp_path: Path) -> None:
        root = tmp_path / "model"
        publish_snapshot(root)
        manifest = build_manifest(root)
        assert sorted(manifest) == sorted(GOLDEN["manifest_fields"])
        assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert len(manifest["files"]) == GOLDEN["asset_count"]
        for entry in manifest["files"]:
            assert sorted(entry) == sorted(GOLDEN["manifest_file_fields"])
            assert len(entry["sha256"]) == 64
            assert entry["size"] > 0

    def test_the_manifest_never_hashes_its_own_reserved_files(self, tmp_path: Path) -> None:
        root = tmp_path / "model"
        publish_snapshot(root)
        paths = {entry["path"] for entry in build_manifest(root)["files"]}
        assert MANIFEST_NAME not in paths
        assert COMPLETE_NAME not in paths


class TestIdentifierFormats:
    def test_the_recorded_revision_format_is_enforced(self) -> None:
        assert GOLDEN["revision_format"] == "40 lowercase hex characters"
        assert canonical_revision(REVISION.upper()) == REVISION
        assert len(REVISION) == 40

    @pytest.mark.parametrize(
        "revision", ["main", "v1.0", "", "a" * 39, "a" * 41, "g" * 40, " " + "a" * 39]
    )
    def test_non_commit_revisions_are_refused(self, revision: str) -> None:
        with pytest.raises(ModelValidationError, match="40-hex commit SHA"):
            canonical_revision(revision)

    def test_the_recorded_repository_format_is_enforced(self) -> None:
        assert GOLDEN["repository_format"] == "owner/name"
        assert validate_repository("ai4bharat/indic-conformer-600m-multilingual")

    @pytest.mark.parametrize(
        "repository", ["", "owner", "owner/", "/name", "owner//name", "own er/name", "../name"]
    )
    def test_malformed_repositories_are_refused(self, repository: str) -> None:
        with pytest.raises(ModelValidationError, match="owner/name identifier"):
            validate_repository(repository)
