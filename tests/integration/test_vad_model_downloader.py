"""Offline tests for atomic provisioning of the pinned Silero VAD artifacts."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest

import scripts._atomic_directory as atomic_directory
import scripts.download_vad_model as downloader
from app.vad.artifact import SILERO_VAD_REVISION
from scripts.download_vad_model import (
    ArtifactSpec,
    VADModelDownloadError,
    VADModelValidationError,
    provision_vad_model,
    verify_vad_model,
)


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        url: str,
        *,
        content_length: int | None = None,
    ) -> None:
        self._payload = payload
        self._offset = 0
        self._url = url
        self.headers: dict[str, str] = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *arguments: Any) -> None:
        return None

    def geturl(self) -> str:
        return self._url

    def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        end = len(self._payload) if size < 0 else self._offset + size
        chunk = self._payload[self._offset : end]
        self._offset += len(chunk)
        return chunk


@pytest.fixture
def artifacts(monkeypatch: pytest.MonkeyPatch) -> tuple[bytes, bytes]:
    model = b"test-owned fake ONNX bytes\x00\x01"
    license_text = b"MIT License\n\ntest-owned fixture, not an upstream artifact\n"
    revision = SILERO_VAD_REVISION
    monkeypatch.setattr(
        downloader,
        "MODEL_ARTIFACT",
        ArtifactSpec(
            "silero_vad.onnx",
            f"https://raw.githubusercontent.com/snakers4/silero-vad/{revision}/model.onnx",
            hashlib.sha256(model).hexdigest(),
            1024,
        ),
    )
    monkeypatch.setattr(
        downloader,
        "LICENSE_ARTIFACT",
        ArtifactSpec(
            "LICENSE",
            f"https://raw.githubusercontent.com/snakers4/silero-vad/{revision}/LICENSE",
            hashlib.sha256(license_text).hexdigest(),
            1024,
        ),
    )
    return model, license_text


def install_fake_network(
    monkeypatch: pytest.MonkeyPatch,
    payloads: Mapping[str, bytes | Exception],
    calls: list[str] | None = None,
) -> None:
    def fake_urlopen(request: Request, *, timeout: int) -> FakeResponse:
        assert timeout == 30
        assert request.get_header("Accept-encoding") == "identity"
        url = request.full_url
        if calls is not None:
            calls.append(url)
        result = payloads[url]
        if isinstance(result, Exception):
            raise result
        return FakeResponse(result, url, content_length=len(result))

    monkeypatch.setattr(downloader, "urlopen", fake_urlopen)


def payload_map(model: bytes, license_text: bytes) -> dict[str, bytes]:
    return {
        downloader.MODEL_ARTIFACT.url: model,
        downloader.LICENSE_ARTIFACT.url: license_text,
    }


class TestProvisioning:
    def test_correct_hashes_are_atomically_published(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        artifacts: tuple[bytes, bytes],
    ) -> None:
        model, license_text = artifacts
        install_fake_network(monkeypatch, payload_map(model, license_text))
        destination = tmp_path / "vad"

        provision_vad_model(destination)

        verify_vad_model(destination)
        assert (destination / "silero_vad.onnx").read_bytes() == model
        assert (destination / "LICENSE").read_bytes() == license_text
        assert list(tmp_path.glob(".vad.staging-*")) == []

    def test_parent_fsync_failure_after_rename_is_propagated(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        artifacts: tuple[bytes, bytes],
    ) -> None:
        model, license_text = artifacts
        install_fake_network(monkeypatch, payload_map(model, license_text))
        destination = tmp_path / "vad"

        def fail_parent_fsync(path: Path) -> None:
            assert path == tmp_path
            raise OSError("parent fsync failed")

        monkeypatch.setattr(atomic_directory, "fsync_directory", fail_parent_fsync)

        with pytest.raises(OSError, match="parent fsync failed"):
            provision_vad_model(destination)

        verify_vad_model(destination)
        assert list(tmp_path.glob(".vad.staging-*")) == []

    def test_a_genuine_concurrent_winner_is_verified_and_reused(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        artifacts: tuple[bytes, bytes],
    ) -> None:
        model, license_text = artifacts
        install_fake_network(monkeypatch, payload_map(model, license_text))
        destination = tmp_path / "vad"

        def lose_rename_race(staging: Path, target: Path) -> None:
            assert target == destination
            shutil.copytree(staging, target)
            raise FileExistsError("concurrent winner")

        monkeypatch.setattr(atomic_directory.os, "rename", lose_rename_race)  # type: ignore[attr-defined]

        provision_vad_model(destination)

        verify_vad_model(destination)
        assert (destination / "silero_vad.onnx").read_bytes() == model
        assert list(tmp_path.glob(".vad.staging-*")) == []

    def test_an_existing_valid_directory_is_idempotent_and_never_uses_network(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        artifacts: tuple[bytes, bytes],
    ) -> None:
        model, license_text = artifacts
        calls: list[str] = []
        install_fake_network(monkeypatch, payload_map(model, license_text), calls)
        destination = tmp_path / "vad"
        provision_vad_model(destination)
        assert len(calls) == 2

        provision_vad_model(destination)

        assert len(calls) == 2

    def test_corruption_fails_closed_without_replacing_or_redownloading(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        artifacts: tuple[bytes, bytes],
    ) -> None:
        model, license_text = artifacts
        calls: list[str] = []
        install_fake_network(monkeypatch, payload_map(model, license_text), calls)
        destination = tmp_path / "vad"
        provision_vad_model(destination)
        target = destination / "silero_vad.onnx"
        target.write_bytes(b"corrupt")

        with pytest.raises(VADModelValidationError, match="SHA-256 mismatch"):
            provision_vad_model(destination)

        assert target.read_bytes() == b"corrupt"
        assert len(calls) == 2

    def test_a_stream_larger_than_the_bound_is_rejected_and_removed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        artifacts: tuple[bytes, bytes],
    ) -> None:
        model, license_text = artifacts
        monkeypatch.setattr(
            downloader,
            "MODEL_ARTIFACT",
            ArtifactSpec(
                downloader.MODEL_ARTIFACT.filename,
                downloader.MODEL_ARTIFACT.url,
                hashlib.sha256(model).hexdigest(),
                len(model) - 1,
            ),
        )
        responses = payload_map(model, license_text)

        def fake_urlopen(request: Request, *, timeout: int) -> FakeResponse:
            assert timeout == 30
            payload = responses[request.full_url]
            return FakeResponse(payload, request.full_url)

        monkeypatch.setattr(downloader, "urlopen", fake_urlopen)
        destination = tmp_path / "vad"

        with pytest.raises(VADModelDownloadError, match="size limit"):
            provision_vad_model(destination)

        assert not destination.exists()
        assert list(tmp_path.glob(".vad.staging-*")) == []

    def test_a_failure_after_a_partial_download_cleans_staging(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        artifacts: tuple[bytes, bytes],
    ) -> None:
        model, _ = artifacts
        install_fake_network(
            monkeypatch,
            {
                downloader.MODEL_ARTIFACT.url: model,
                downloader.LICENSE_ARTIFACT.url: OSError("connection reset"),
            },
        )
        destination = tmp_path / "vad"

        with pytest.raises(VADModelDownloadError, match="download failed for LICENSE"):
            provision_vad_model(destination)

        assert not destination.exists()
        assert list(tmp_path.glob(".vad.staging-*")) == []

    def test_a_network_failure_never_publishes_a_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        artifacts: tuple[bytes, bytes],
    ) -> None:
        install_fake_network(
            monkeypatch,
            {downloader.MODEL_ARTIFACT.url: TimeoutError("offline")},
        )
        destination = tmp_path / "vad"

        with pytest.raises(VADModelDownloadError, match="TimeoutError"):
            provision_vad_model(destination)

        assert not destination.exists()
        assert list(tmp_path.glob(".vad.staging-*")) == []


class TestOfflineVerification:
    def test_offline_mode_verifies_without_opening_the_network(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        artifacts: tuple[bytes, bytes],
    ) -> None:
        model, license_text = artifacts
        install_fake_network(monkeypatch, payload_map(model, license_text))
        destination = tmp_path / "vad"
        provision_vad_model(destination)

        def forbidden(*arguments: Any, **keywords: Any) -> None:
            raise AssertionError("offline verification attempted network access")

        monkeypatch.setattr(downloader, "urlopen", forbidden)
        provision_vad_model(destination, offline=True)

    def test_offline_mode_rejects_a_missing_directory_without_network(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        artifacts: tuple[bytes, bytes],
    ) -> None:
        def forbidden(*arguments: Any, **keywords: Any) -> None:
            raise AssertionError("offline verification attempted network access")

        monkeypatch.setattr(downloader, "urlopen", forbidden)
        with pytest.raises(VADModelValidationError, match="offline verification mode"):
            provision_vad_model(tmp_path / "absent", offline=True)

    @pytest.mark.parametrize("link_target", ["directory", "model"])
    def test_symlinks_are_rejected(
        self,
        link_target: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        artifacts: tuple[bytes, bytes],
    ) -> None:
        model, license_text = artifacts
        real = tmp_path / "real"
        real.mkdir()
        (real / "silero_vad.onnx").write_bytes(model)
        (real / "LICENSE").write_bytes(license_text)
        if link_target == "directory":
            destination = tmp_path / "linked"
            destination.symlink_to(real, target_is_directory=True)
        else:
            destination = real
            target = destination / "silero_vad.onnx"
            target.unlink()
            target.symlink_to(tmp_path / "outside.onnx")

        with pytest.raises(VADModelValidationError, match="real directory|regular file"):
            verify_vad_model(destination)
