"""
Configuration management for AgenticSQL.

All settings are loaded from environment variables (via .env file).
No credentials or connection details are hardcoded in source code.
"""

import os
import urllib.parse
from dataclasses import dataclass
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # API
    google_api_key: str = ""

    # Database
    db_user: str = ""
    db_password: str = ""
    db_server: str = "127.0.0.1"
    db_name: str = ""
    db_driver: str = "ODBC+Driver+17+for+SQL+Server"

    # LLM
    llm_model: str = "gemini-3.6-flash"
    llm_temperature: float = 0.0

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        load_dotenv()

        config = cls(
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            db_user=os.getenv("DB_USER", ""),
            db_password=os.getenv("DB_PASSWORD", ""),
            db_server=os.getenv("DB_SERVER", "127.0.0.1"),
            db_name=os.getenv("DB_NAME", ""),
            db_driver=os.getenv("DB_DRIVER", "ODBC+Driver+17+for+SQL+Server"),
            llm_model=os.getenv("LLM_MODEL", "gemini-3.6-flash"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            server_host=os.getenv("SERVER_HOST", "0.0.0.0"),
            server_port=int(os.getenv("SERVER_PORT", "8000")),
        )

        config.validate()
        return config

    def validate(self) -> None:
        """Validate that all required configuration values are present."""
        errors: list[str] = []

        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY is missing from your .env file")
        if not self.db_user:
            errors.append("DB_USER is missing from your .env file")
        if not self.db_password:
            errors.append("DB_PASSWORD is missing from your .env file")
        if not self.db_name:
            errors.append("DB_NAME is missing from your .env file")

        if errors:
            raise ValueError(
                "Configuration errors:\n" + "\n".join(f"  • {e}" for e in errors)
            )

    @property
    def connection_string(self) -> str:
        """
        Build a properly URL-encoded database connection string.

        Uses urllib.parse.quote_plus() to safely encode the password and
        username, preventing issues with special characters like @, :, /
        in credentials.
        """
        encoded_password = urllib.parse.quote_plus(self.db_password)
        encoded_user = urllib.parse.quote_plus(self.db_user)
        return (
            f"mssql+pyodbc://{encoded_user}:{encoded_password}"
            f"@{self.db_server}/{self.db_name}"
            f"?driver={self.db_driver}&TrustServerCertificate=yes"
        )
