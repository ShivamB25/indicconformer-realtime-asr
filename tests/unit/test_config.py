"""Contract tests for environment-backed settings.

Settings decide whether the service may start with a mock engine, whether it
may reach the network, and what limits requests are held to, so every branch of
the validator is pinned here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings, read_api_key
from app.core.types import EngineKind


def clear_asr_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for name in list(os.environ):
        if name.startswith("ASR_"):
            monkeypatch.delenv(name, raising=False)


def local_artifacts(tmp_path: Path) -> dict[str, Any]:
    manifest = tmp_path / "model-manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    return {
        "model_dir": tmp_path,
        "model_repo_id": "ai4bharat/indic-conformer-600m-multilingual",
        "model_revision": "0123456789abcdef0123456789abcdef01234567",
        "model_manifest": manifest,
    }


class TestDefaults:
    def test_defaults_are_safe_for_a_developer_machine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clear_asr_environment(monkeypatch)
        settings = Settings()
        assert settings.environment == "development"
        assert settings.engine is EngineKind.MOCK
        assert settings.host == "127.0.0.1"
        assert settings.port == 8000
        assert settings.offline is True
        assert settings.require_cuda is True
        assert settings.sample_rate == 16_000
        assert settings.model_dir is None

    def test_a_dotenv_file_in_the_working_directory_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        clear_asr_environment(monkeypatch)
        (tmp_path / ".env").write_text("ASR_ENGINE=ort\nASR_PORT=9999\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        settings = Settings()
        assert settings.engine is EngineKind.MOCK
        assert settings.port == 8000

    def test_environment_variables_are_case_insensitive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clear_asr_environment(monkeypatch)
        monkeypatch.setenv("asr_port", "8123")
        assert Settings().port == 8123

    def test_an_unrecognised_asr_variable_changes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prefixed variables that match no field are ignored, not applied."""

        clear_asr_environment(monkeypatch)
        monkeypatch.setenv("ASR_SECRET_BACKDOOR", "1")
        monkeypatch.setenv("ASR_PORTT", "9999")
        settings = Settings()
        assert settings.port == 8000
        assert settings.model_extra is None

    def test_unknown_constructor_arguments_are_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clear_asr_environment(monkeypatch)
        with pytest.raises(ValidationError) as caught:
            Settings(secret_backdoor=1)  # type: ignore[call-arg]
        assert {error["type"] for error in caught.value.errors()} == {"extra_forbidden"}


class TestEngineSelection:
    def test_engine_kind_set_is_closed(self) -> None:
        assert [kind.value for kind in EngineKind] == ["mock", "official", "ort"]

    @pytest.mark.parametrize("engine", ["whisper", "MOCK", "onnx", ""])
    def test_unknown_engines_are_rejected(
        self, monkeypatch: pytest.MonkeyPatch, engine: str
    ) -> None:
        clear_asr_environment(monkeypatch)
        monkeypatch.setenv("ASR_ENGINE", engine)
        with pytest.raises(ValidationError):
            Settings()

    def test_production_refuses_the_mock_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clear_asr_environment(monkeypatch)
        monkeypatch.setenv("ASR_ENVIRONMENT", "production")
        monkeypatch.setenv("ASR_ENGINE", "mock")
        with pytest.raises(ValidationError, match="mock engine is not permitted in production"):
            Settings()

    def test_mock_engine_is_allowed_outside_production(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clear_asr_environment(monkeypatch)
        monkeypatch.setenv("ASR_ENVIRONMENT", "test")
        assert Settings().engine is EngineKind.MOCK

    @pytest.mark.parametrize("engine", ["official", "ort"])
    def test_real_engines_require_every_pinned_artifact(
        self, monkeypatch: pytest.MonkeyPatch, engine: str
    ) -> None:
        clear_asr_environment(monkeypatch)
        monkeypatch.setenv("ASR_ENGINE", engine)
        with pytest.raises(ValidationError) as caught:
            Settings()
        message = str(caught.value)
        assert "model_dir" in message
        assert "model_repo_id" in message
        assert "model_revision" in message
        assert "model_manifest" in message

    @pytest.mark.parametrize(
        "omitted", ["model_dir", "model_repo_id", "model_revision", "model_manifest"]
    )
    def test_a_single_missing_artifact_is_named(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, omitted: str
    ) -> None:
        clear_asr_environment(monkeypatch)
        monkeypatch.setenv("ASR_ENGINE", "ort")
        for name, value in local_artifacts(tmp_path).items():
            if name != omitted:
                monkeypatch.setenv(f"ASR_{name.upper()}", str(value))
        with pytest.raises(ValidationError) as caught:
            Settings()
        message = str(caught.value)
        assert omitted in message
        assert "missing" in message

    def test_real_engines_must_stay_offline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        clear_asr_environment(monkeypatch)
        monkeypatch.setenv("ASR_ENGINE", "ort")
        for name, value in local_artifacts(tmp_path).items():
            monkeypatch.setenv(f"ASR_{name.upper()}", str(value))
        monkeypatch.setenv("ASR_OFFLINE", "false")
        with pytest.raises(ValidationError, match="offline=true"):
            Settings()

    def test_a_fully_pinned_offline_engine_validates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        clear_asr_environment(monkeypatch)
        monkeypatch.setenv("ASR_ENGINE", "official")
        artifacts = local_artifacts(tmp_path)
        for name, value in artifacts.items():
            monkeypatch.setenv(f"ASR_{name.upper()}", str(value))
        settings = Settings()
        assert settings.engine is EngineKind.OFFICIAL
        assert settings.offline is True
        assert settings.model_dir == artifacts["model_dir"]
        assert settings.model_manifest == artifacts["model_manifest"]

    def test_production_requires_an_absolute_api_key_path(self, tmp_path: Path) -> None:
        artifacts = local_artifacts(tmp_path)
        with pytest.raises(ValidationError, match="api_key_file"):
            Settings(
                environment="production",
                engine=EngineKind.ORT,
                api_key_file=Path("relative-token"),
                **artifacts,
            )

        settings = Settings(
            environment="production",
            engine=EngineKind.ORT,
            api_key_file=Path("/run/secrets/api_key"),
            **artifacts,
        )
        assert settings.api_key_file == Path("/run/secrets/api_key")

    def test_the_mock_engine_needs_no_artifacts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clear_asr_environment(monkeypatch)
        monkeypatch.setenv("ASR_OFFLINE", "false")
        settings = Settings()
        assert settings.engine is EngineKind.MOCK
        assert settings.offline is False


class TestApiKeyFile:
    TOKEN = "a-secure-service-api-key-that-is-long-enough"

    def test_one_bounded_ascii_token_is_loaded(self, tmp_path: Path) -> None:
        key_file = tmp_path / "api.key"
        key_file.write_text(f"{self.TOKEN}\n", encoding="utf-8")

        assert read_api_key(key_file) == self.TOKEN

    @pytest.mark.parametrize(
        "content",
        [
            "too-short",
            "x" * 4097,
            "x" * 31 + " embedded-space",
            "é" * 32,
        ],
        ids=["too_short", "too_long", "whitespace", "non_ascii"],
    )
    def test_invalid_or_non_comparable_tokens_are_rejected(
        self, tmp_path: Path, content: str
    ) -> None:
        key_file = tmp_path / "api.key"
        key_file.write_text(content, encoding="utf-8")

        with pytest.raises(ValueError, match="32-4096 character ASCII token"):
            read_api_key(key_file)

    def test_non_regular_key_paths_are_rejected(self, tmp_path: Path) -> None:
        target = tmp_path / "real.key"
        target.write_text(self.TOKEN, encoding="utf-8")
        link = tmp_path / "linked.key"
        link.symlink_to(target)

        with pytest.raises(ValueError, match="regular file"):
            read_api_key(link)
        with pytest.raises(ValueError, match="regular file"):
            read_api_key(tmp_path)


class TestLimits:
    @pytest.mark.parametrize("port", [0, -1, 65_536, 70_000])
    def test_port_must_be_a_usable_tcp_port(
        self, monkeypatch: pytest.MonkeyPatch, port: int
    ) -> None:
        clear_asr_environment(monkeypatch)
        monkeypatch.setenv("ASR_PORT", str(port))
        with pytest.raises(ValidationError) as caught:
            Settings()
        assert error_fields(caught.value) == {("port",)}

    @pytest.mark.parametrize("value", [1, 0, -1, 500 * 1024 * 1024 + 1])
    def test_upload_limit_is_bounded(self, monkeypatch: pytest.MonkeyPatch, value: int) -> None:
        clear_asr_environment(monkeypatch)
        monkeypatch.setenv("ASR_MAX_UPLOAD_BYTES", str(value))
        with pytest.raises(ValidationError) as caught:
            Settings()
        assert error_fields(caught.value) == {("max_upload_bytes",)}

    @pytest.mark.parametrize("value", [0, -5, 3_601])
    def test_audio_duration_limit_is_bounded(
        self, monkeypatch: pytest.MonkeyPatch, value: int
    ) -> None:
        clear_asr_environment(monkeypatch)
        monkeypatch.setenv("ASR_MAX_AUDIO_SECONDS", str(value))
        with pytest.raises(ValidationError) as caught:
            Settings()
        assert error_fields(caught.value) == {("max_audio_seconds",)}

    @pytest.mark.parametrize("value", ["0", "-1", "3601"])
    def test_request_timeout_is_bounded(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        clear_asr_environment(monkeypatch)
        monkeypatch.setenv("ASR_REQUEST_TIMEOUT_SECONDS", value)
        with pytest.raises(ValidationError) as caught:
            Settings()
        assert error_fields(caught.value) == {("request_timeout_seconds",)}

    @pytest.mark.parametrize("rate", ["8000", "44100", "48000"])
    def test_sample_rate_is_pinned_to_sixteen_kilohertz(
        self, monkeypatch: pytest.MonkeyPatch, rate: str
    ) -> None:
        clear_asr_environment(monkeypatch)
        monkeypatch.setenv("ASR_SAMPLE_RATE", rate)
        with pytest.raises(ValidationError) as caught:
            Settings()
        assert error_fields(caught.value) == {("sample_rate",)}

    def test_limits_accept_their_documented_extremes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clear_asr_environment(monkeypatch)
        monkeypatch.setenv("ASR_PORT", "65535")
        monkeypatch.setenv("ASR_MAX_UPLOAD_BYTES", str(500 * 1024 * 1024))
        monkeypatch.setenv("ASR_MAX_AUDIO_SECONDS", "3600")
        monkeypatch.setenv("ASR_REQUEST_TIMEOUT_SECONDS", "3600")
        settings = Settings()
        assert settings.port == 65_535
        assert settings.max_audio_seconds == 3_600


class TestSettingsCache:
    def test_get_settings_is_cached_until_cleared(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clear_asr_environment(monkeypatch)
        get_settings.cache_clear()
        first = get_settings()
        assert get_settings() is first

        monkeypatch.setenv("ASR_PORT", "8321")
        assert get_settings() is first
        assert get_settings().port == 8000

        get_settings.cache_clear()
        assert get_settings().port == 8321


def error_fields(exc: ValidationError) -> set[tuple[int | str, ...]]:
    return {error["loc"] for error in exc.errors()}
