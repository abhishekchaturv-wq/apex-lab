"""Refactored settings with nested models.

Provides typed configuration using nested Pydantic Settings models
for better organization and type safety.
"""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from apex_lab.exceptions import ConfigurationError


class KiteSettings(BaseSettings):
    """Zerodha Kite Connect configuration.

    Attributes:
        api_key: Kite API key from Zerodha dashboard
        api_secret: Kite API secret from Zerodha dashboard
        access_token: Access token obtained after authentication
    """

    api_key: str = Field(..., description="Kite API key")
    api_secret: str = Field(..., description="Kite API secret")
    access_token: str = Field(default="", description="Kite access token")


class DataSettings(BaseSettings):
    """Data storage configuration.

    Attributes:
        dir: Root directory for all data storage
        cache_dir: Directory for temporary caches
    """

    dir: Path = Field(default=Path("./data"), description="Root data directory")
    cache_dir: Path = Field(
        default=Path("./data/cache"), description="Cache directory"
    )

    @property
    def raw_path(self) -> Path:
        """Path to raw data directory."""
        return self.dir / "raw"

    @property
    def processed_path(self) -> Path:
        """Path to processed data directory."""
        return self.dir / "processed"

    @property
    def features_path(self) -> Path:
        """Path to features directory."""
        return self.processed_path / "features"

    @property
    def labels_path(self) -> Path:
        """Path to labels directory."""
        return self.processed_path / "labels"

    @property
    def models_path(self) -> Path:
        """Path to models directory."""
        return self.dir / "models"

    @property
    def reports_path(self) -> Path:
        """Path to reports directory."""
        return self.dir / "reports"


class LoggingSettings(BaseSettings):
    """Logging configuration.

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file: Log file path
        max_bytes: Maximum log file size in bytes before rotation
        backup_count: Number of backup log files to keep
    """

    level: str = Field(default="INFO", description="Logging level")
    file: Path = Field(default=Path("./logs/apex.log"), description="Log file path")
    max_bytes: int = Field(
        default=10485760, description="Max log file size (10MB default)"
    )
    backup_count: int = Field(
        default=5, description="Number of backup log files to keep"
    )

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


class ApplicationSettings(BaseSettings):
    """Application-level configuration.

    Attributes:
        timezone: Timezone for timestamp handling
        default_interval: Default candle interval for data fetch
    """

    timezone: str = Field(default="Asia/Kolkata", description="Timezone")
    default_interval: str = Field(default="5minute", description="Default interval")

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Validate timezone."""
        try:
            import pytz

            pytz.timezone(v)
            return v
        except Exception:
            raise ValueError(f"Invalid timezone: {v}")

    @field_validator("default_interval")
    @classmethod
    def validate_interval(cls, v: str) -> str:
        """Validate default interval."""
        valid_intervals = {
            "minute",
            "5minute",
            "15minute",
            "30minute",
            "60minute",
            "day",
            "week",
            "month",
        }
        if v not in valid_intervals:
            raise ValueError(
                f"Invalid interval: {v}. Must be one of {valid_intervals}"
            )
        return v


class Settings(BaseSettings):
    """Root application settings.

    Combines all configuration categories into a single settings object.
    """

    kite: KiteSettings = Field(default_factory=KiteSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    app: ApplicationSettings = Field(default_factory=ApplicationSettings)

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False

    def __init__(self, **data):
        """Initialize settings and create required directories."""
        super().__init__(**data)
        self._ensure_directories_exist()

    def _ensure_directories_exist(self) -> None:
        """Create all required directories."""
        directories = [
            self.data.dir,
            self.data.cache_dir,
            self.data.raw_path,
            self.data.processed_path,
            self.data.features_path,
            self.data.labels_path,
            self.data.models_path,
            self.data.reports_path,
            self.logging.file.parent,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """Validate all settings.

        Raises:
            ConfigurationError: If any validation fails
        """
        errors = []

        # Check required directories exist and are writable
        for directory in [
            self.data.dir,
            self.data.cache_dir,
        ]:
            if not directory.exists():
                errors.append(f"Directory does not exist: {directory}")
            elif not directory.is_dir():
                errors.append(f"Path is not a directory: {directory}")
            elif not directory.writable():
                errors.append(f"Directory is not writable: {directory}")

        # Check Kite credentials
        if not self.kite.api_key:
            errors.append("KITE_API_KEY is not configured")
        if not self.kite.api_secret:
            errors.append("KITE_API_SECRET is not configured")

        # Check logging file directory is writable
        if not self.logging.file.parent.writable():
            errors.append(f"Log directory is not writable: {self.logging.file.parent}")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            raise ConfigurationError(error_msg)

    # Legacy properties for backward compatibility
    @property
    def kite_api_key(self) -> str:
        """Legacy property for API key."""
        return self.kite.api_key

    @property
    def kite_api_secret(self) -> str:
        """Legacy property for API secret."""
        return self.kite.api_secret

    @property
    def kite_access_token(self) -> str:
        """Legacy property for access token."""
        return self.kite.access_token

    @property
    def data_dir(self) -> Path:
        """Legacy property for data directory."""
        return self.data.dir

    @property
    def log_level(self) -> str:
        """Legacy property for log level."""
        return self.logging.level

    @property
    def timezone(self) -> str:
        """Legacy property for timezone."""
        return self.app.timezone

    @property
    def default_interval(self) -> str:
        """Legacy property for default interval."""
        return self.app.default_interval


# Global settings instance
settings = Settings()
