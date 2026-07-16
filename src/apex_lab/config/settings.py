"""Configuration management for APEX Lab.

Provides typed configuration using Pydantic Settings.
Configuration is loaded from environment variables and .env files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import AliasChoices, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment.

    All sensitive values (API keys, tokens) should be set via environment
    variables or .env file. Never hardcode them.

    Attributes:
        kite_api_key: Zerodha Kite API key
        kite_api_secret: Zerodha Kite API secret
        kite_access_token: Zerodha Kite access token (obtained after login)
        data_dir: Root directory for all data storage.  Reads
            ``APEX_DATA_DIR`` or ``DATA_DIR`` from the environment;
            defaults to ``~/kite-test/apex-data-lake``.
        cache_dir: Directory for temporary caches
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        timezone: Timezone for timestamp handling (e.g., 'Asia/Kolkata')
        default_interval: Default candle interval for data fetch (e.g., '5m', '15m')
    """

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
    )

    # Zerodha Kite Connect
    kite_api_key: str
    kite_api_secret: str
    kite_access_token: str = ""

    # Data Storage — reads APEX_DATA_DIR or DATA_DIR from environment
    data_dir: Path = Field(
        default=Path("~/kite-test/apex-data-lake"),
        validation_alias=AliasChoices("apex_data_dir", "data_dir"),
    )
    cache_dir: Path = Path("~/kite-test/apex-data-lake/cache")

    # Logging
    log_level: str = "INFO"

    # System
    timezone: str = "Asia/Kolkata"
    default_interval: str = "5m"

    @field_validator("data_dir", "cache_dir", mode="before")
    @classmethod
    def expand_user(cls, v: Any) -> Path:
        """Expand ``~`` in directory paths."""
        return Path(v).expanduser()

    def __init__(self, **data: Any) -> None:
        """Initialize settings and create required directories."""
        super().__init__(**data)
        # Ensure data directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class _LazySettings:
    """Proxy that defers :class:`Settings` instantiation until first attribute access.

    This allows modules to import ``settings`` at the top of the file without
    triggering a :class:`Settings` instantiation (and the accompanying
    environment-variable validation) at import time.  The real ``Settings``
    object is created on the first attribute access, so offline test suites
    that never touch Kite credentials can import and exercise the library
    without a ``.env`` file.
    """

    def __getattr__(self, name: str) -> Any:
        # On first access, build and cache the real Settings instance.
        try:
            instance: Settings = object.__getattribute__(self, "_instance")
        except AttributeError:
            instance = Settings()
            object.__setattr__(self, "_instance", instance)
        return getattr(instance, name)

    def __repr__(self) -> str:
        try:
            instance = object.__getattribute__(self, "_instance")
            return repr(instance)
        except AttributeError:
            return "<Settings: not yet initialized>"


# Lazy global settings proxy — Settings() is only instantiated when a
# settings attribute is first accessed (e.g. settings.kite_api_key).
settings: _LazySettings = _LazySettings()
