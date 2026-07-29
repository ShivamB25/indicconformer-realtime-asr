"""Environment-backed service configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.types import EngineKind


class Settings(BaseSettings):
    """Strict runtime settings, read from ``ASR_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="ASR_",
        env_file=None,
        case_sensitive=False,
        extra="forbid",
        validate_default=True,
    )

    environment: Literal["development", "test", "production"] = "development"
    engine: EngineKind = EngineKind.MOCK
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    model_dir: Path | None = None
    model_repo_id: str | None = None
    model_revision: str | None = None
    model_manifest: Path | None = None
    offline: bool = True
    require_cuda: bool = True

    sample_rate: Literal[16000] = 16000
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, ge=2, le=500 * 1024 * 1024)
    max_audio_seconds: int = Field(default=600, ge=1, le=3600)
    request_timeout_seconds: float = Field(default=120.0, gt=0, le=3600)
    websocket_allowed_origins: tuple[str, ...] = ()
    websocket_bearer_token_file: Path | None = None

    @model_validator(mode="after")
    def validate_engine_artifacts(self) -> Self:
        if self.environment == "production" and self.engine is EngineKind.MOCK:
            raise ValueError("the mock engine is not permitted in production")
        if self.environment == "production" and not self.require_cuda:
            raise ValueError("production engines require CUDA")
        if self.environment == "production":
            token_file = self.websocket_bearer_token_file
            if token_file is None or not token_file.is_absolute():
                raise ValueError("production requires an absolute websocket_bearer_token_file")
        if any(
            not origin.startswith(("http://", "https://"))
            for origin in self.websocket_allowed_origins
        ):
            raise ValueError("websocket_allowed_origins must contain exact HTTP origins")

        if self.engine is not EngineKind.MOCK:
            missing = [
                name
                for name, value in (
                    ("model_dir", self.model_dir),
                    ("model_repo_id", self.model_repo_id),
                    ("model_revision", self.model_revision),
                    ("model_manifest", self.model_manifest),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "production engines require pinned local artifacts; missing "
                    + ", ".join(missing)
                )
            if not self.offline:
                raise ValueError("production engines require offline=true")

        return self


def read_websocket_bearer_token(path: Path) -> str:
    """Load one bounded token from a real file without exposing its contents."""
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("websocket bearer token path must be a regular file")
        token = path.read_text(encoding="utf-8").strip()
    except ValueError:
        raise
    except (OSError, UnicodeError):
        raise ValueError("websocket bearer token file cannot be read") from None
    if len(token) < 32 or len(token) > 4096 or any(character.isspace() for character in token):
        raise ValueError("websocket bearer token must contain one 32-4096 character token")
    return token


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process settings without constructing any model objects."""

    return Settings()
