"""The model download/verify command-line entry points.

These run the scripts as scripts, in a subprocess, against a snapshot of
placeholder files. The environment is pinned offline, so a CLI that tried to
reach Hugging Face would fail rather than silently succeed on a dev machine.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.model_snapshot import (
    DOWNLOAD_CLI,
    OTHER_REVISION,
    REPOSITORY,
    REQUIRED_ASSETS,
    REVISION,
    publish_snapshot,
    write_token_file,
)

OFFLINE_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/nonexistent",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}
TOKEN_VALUE = "hf_secret_token_value"
VERIFY_CLI = Path(__file__).resolve().parents[2] / "scripts" / "verify_model.py"


def run_cli(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        capture_output=True,
        text=True,
        env=OFFLINE_ENV,
        timeout=120,
        check=False,
    )


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    publish_snapshot(root)
    return root


class TestVerifyCli:
    def test_a_valid_snapshot_is_reported_as_verified(self, snapshot: Path) -> None:
        result = run_cli(
            VERIFY_CLI,
            "--model-dir",
            str(snapshot),
            "--repository",
            REPOSITORY,
            "--revision",
            REVISION,
        )
        assert result.returncode == 0, result.stderr
        assert f"verified {len(REQUIRED_ASSETS)} files" in result.stdout
        assert f"{REPOSITORY}@{REVISION}" in result.stdout

    def test_a_revision_mismatch_fails_with_a_usage_error(self, snapshot: Path) -> None:
        result = run_cli(
            VERIFY_CLI,
            "--model-dir",
            str(snapshot),
            "--repository",
            REPOSITORY,
            "--revision",
            OTHER_REVISION,
        )
        assert result.returncode == 2
        assert "revision does not match" in result.stderr
        assert result.stdout == ""

    def test_a_corrupt_asset_fails(self, snapshot: Path) -> None:
        target = snapshot / "assets" / "encoder.onnx"
        target.write_bytes(bytes(reversed(target.read_bytes())))
        result = run_cli(
            VERIFY_CLI,
            "--model-dir",
            str(snapshot),
            "--repository",
            REPOSITORY,
            "--revision",
            REVISION,
        )
        assert result.returncode == 2
        assert "SHA-256 mismatch" in result.stderr

    def test_a_missing_directory_fails(self, tmp_path: Path) -> None:
        result = run_cli(
            VERIFY_CLI,
            "--model-dir",
            str(tmp_path / "absent"),
            "--repository",
            REPOSITORY,
            "--revision",
            REVISION,
        )
        assert result.returncode == 2
        assert "model directory does not exist" in result.stderr

    @pytest.mark.parametrize(
        "arguments",
        [
            (),
            ("--model-dir", "/tmp"),
            ("--repository", REPOSITORY, "--revision", REVISION),
        ],
        ids=["no_arguments", "missing_repository_and_revision", "missing_model_dir"],
    )
    def test_required_arguments_are_enforced(self, arguments: tuple[str, ...]) -> None:
        result = run_cli(VERIFY_CLI, *arguments)
        assert result.returncode == 2
        assert "the following arguments are required" in result.stderr

    def test_help_is_available_without_a_snapshot(self) -> None:
        result = run_cli(VERIFY_CLI, "--help")
        assert result.returncode == 0
        assert "--model-dir" in result.stdout
        assert "--revision" in result.stdout


class TestDownloadCli:
    def test_help_is_available(self) -> None:
        result = run_cli(DOWNLOAD_CLI, "--help")
        assert result.returncode == 0
        for option in ("--repository", "--revision", "--token-file", "--output-dir"):
            assert option in result.stdout

    def test_required_arguments_are_enforced(self) -> None:
        result = run_cli(DOWNLOAD_CLI)
        assert result.returncode == 2
        assert "the following arguments are required" in result.stderr

    def test_an_already_published_snapshot_is_accepted_offline(
        self, snapshot: Path, tmp_path: Path
    ) -> None:
        token_file = write_token_file(tmp_path / "hf.token", TOKEN_VALUE)
        result = run_cli(
            DOWNLOAD_CLI,
            "--repository",
            REPOSITORY,
            "--revision",
            REVISION,
            "--token-file",
            str(token_file),
            "--output-dir",
            str(snapshot),
        )
        assert result.returncode == 0, result.stderr
        assert f"model snapshot ready at {snapshot}" in result.stdout

    def test_the_token_is_never_echoed(self, snapshot: Path, tmp_path: Path) -> None:
        token_file = write_token_file(tmp_path / "hf.token", TOKEN_VALUE)
        result = run_cli(
            DOWNLOAD_CLI,
            "--repository",
            REPOSITORY,
            "--revision",
            OTHER_REVISION,
            "--token-file",
            str(token_file),
            "--output-dir",
            str(snapshot),
        )
        assert result.returncode == 2
        assert TOKEN_VALUE not in result.stdout
        assert TOKEN_VALUE not in result.stderr

    def test_an_invalid_revision_is_rejected_before_any_network_use(self, tmp_path: Path) -> None:
        token_file = write_token_file(tmp_path / "hf.token", TOKEN_VALUE)
        result = run_cli(
            DOWNLOAD_CLI,
            "--repository",
            REPOSITORY,
            "--revision",
            "main",
            "--token-file",
            str(token_file),
            "--output-dir",
            str(tmp_path / "out"),
        )
        assert result.returncode == 2
        assert "full 40-hex commit SHA" in result.stderr
        assert not (tmp_path / "out").exists()

    def test_an_unreadable_token_file_is_rejected(self, tmp_path: Path) -> None:
        result = run_cli(
            DOWNLOAD_CLI,
            "--repository",
            REPOSITORY,
            "--revision",
            REVISION,
            "--token-file",
            str(tmp_path / "absent.token"),
            "--output-dir",
            str(tmp_path / "out"),
        )
        assert result.returncode == 2
        assert "token file" in result.stderr
        assert not (tmp_path / "out").exists()
