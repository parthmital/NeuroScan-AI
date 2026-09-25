"""Application settings, read once from the environment and validated at startup."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from cryptography.fernet import Fernet
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Values shipped in .env.example; refusing them stops a copied example from reaching production.
_EXAMPLE_SECRETS = {"change-me", "replace-with-a-random-secret"}


class Settings(BaseModel):
    environment: Literal["development", "production"] = "production"
    jwt_secret: str = Field(min_length=32)
    data_encryption_key: str
    database_url: str = f"sqlite:///{BACKEND_DIR / 'neuroscan.db'}"
    upload_dir: Path = BACKEND_DIR / "uploads"
    detection_weights: Path = BACKEND_DIR / "Binary Detection" / "EfficientNetB0.pth"
    classification_weights: Path = BACKEND_DIR / "Classification" / "EfficientNetB0.pth"
    segmentation_weights: Path = BACKEND_DIR / "Segmentation" / "VNet.pth"
    cors_origins: list[str] = []
    access_token_expire_minutes: int = Field(default=30, ge=1, le=24 * 60)
    cookie_secure: bool = True
    allow_registration: bool = True
    max_upload_file_mb: int = Field(default=300, ge=1, le=4096)
    max_pending_jobs: int = Field(default=8, ge=1, le=1000)
    auth_rate_limit_per_minute: int = Field(default=10, ge=1, le=1000)
    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    @field_validator("jwt_secret")
    @classmethod
    def _reject_example_secret(cls, value: str) -> str:
        if value in _EXAMPLE_SECRETS:
            raise ValueError("JWT_SECRET still has the example value; generate a new one")
        return value

    @field_validator("data_encryption_key")
    @classmethod
    def _valid_fernet_key(cls, value: str) -> str:
        try:
            Fernet(value.encode())
        except (ValueError, TypeError) as exc:
            raise ValueError("DATA_ENCRYPTION_KEY must be a Fernet key") from exc
        return value

    @field_validator(
        "upload_dir",
        "detection_weights",
        "classification_weights",
        "segmentation_weights",
    )
    @classmethod
    def _relative_to_backend(cls, value: Path) -> Path:
        return value if value.is_absolute() else BACKEND_DIR / value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @model_validator(mode="after")
    def _no_wildcard_origin(self) -> "Settings":
        if "*" in self.cors_origins:
            raise ValueError("CORS_ORIGINS cannot contain '*' because requests carry cookies")
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def max_upload_file_bytes(self) -> int:
        return self.max_upload_file_mb * 1024 * 1024


_ENV_FIELDS = {name: name.upper() for name in Settings.model_fields}


def load_settings() -> Settings:
    load_dotenv()
    values = {field: os.environ[env] for field, env in _ENV_FIELDS.items() if env in os.environ}
    return Settings(**values)


@lru_cache
def get_settings() -> Settings:
    return load_settings()
