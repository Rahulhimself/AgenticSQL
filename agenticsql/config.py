"""
Configuration management for AgenticSQL.

Supports:
- Direct connection URIs (DATABASE_URL / DB_URI) for managed and serverless cloud databases
  (Supabase, Neon, PlanetScale, CockroachDB, AWS RDS, GCP Cloud SQL, Azure SQL).
- Discrete parameters for PostgreSQL, MySQL/MariaDB, Microsoft SQL Server/Azure SQL, SQLite.
- URL-encoding of credentials to prevent syntax breakage from special characters.
"""

import os
import urllib.parse
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Standard default ports per database dialect
DEFAULT_PORTS: dict[str, int] = {
    "postgresql": 5432,
    "postgres": 5432,
    "mysql": 3306,
    "mariadb": 3306,
    "mssql": 1433,
    "sqlserver": 1433,
    "cockroachdb": 26257,
    "cockroach": 26257,
}


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # API & LLM Provider Settings
    llm_provider: str = "groq"  # 'groq', 'gemini', 'openai', 'mock'
    groq_api_key: str = ""
    google_api_key: str = ""
    openai_api_key: str = ""

    # Direct database connection URI (if provided, takes precedence over discrete fields)
    database_url: str = ""

    # Database discrete parameters
    db_type: str = "mssql"  # 'mssql', 'postgresql', 'mysql', 'sqlite', 'cockroachdb'
    db_user: str = ""
    db_password: str = ""
    db_server: str = "127.0.0.1"
    db_port: Optional[int] = None
    db_name: str = ""
    db_driver: str = "ODBC+Driver+17+for+SQL+Server"
    db_sslmode: str = ""  # 'require', 'prefer', 'disable', etc.
    db_extra_params: str = ""

    # LLM
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0

    # Self-Healing & Optimization (Phase 4b)
    max_retries: int = 3
    enable_self_healing: bool = True
    enable_schema_pruning: bool = True

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        load_dotenv()

        port_str = os.getenv("DB_PORT", "").strip()
        db_port = int(port_str) if port_str.isdigit() else None

        raw_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        google_key = os.getenv("GOOGLE_API_KEY", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()

        if raw_provider in ("groq", "gemini", "openai", "mock", "fake"):
            if raw_provider == "groq" and not groq_key and google_key:
                provider = "gemini"
            else:
                provider = raw_provider
        elif groq_key:
            provider = "groq"
        elif google_key:
            provider = "gemini"
        elif openai_key:
            provider = "openai"
        else:
            provider = "groq"

        default_model = "llama-3.3-70b-versatile" if provider == "groq" else (
            "gemini-2.5-flash" if provider == "gemini" else "gpt-4o-mini"
        )
        llm_model = os.getenv("LLM_MODEL") or default_model

        config = cls(
            llm_provider=provider,
            groq_api_key=groq_key,
            google_api_key=google_key,
            openai_api_key=openai_key,
            database_url=os.getenv("DATABASE_URL") or os.getenv("DB_URI", ""),
            db_type=os.getenv("DB_TYPE") or os.getenv("DB_DIALECT", "mssql"),
            db_user=os.getenv("DB_USER", ""),
            db_password=os.getenv("DB_PASSWORD", ""),
            db_server=os.getenv("DB_SERVER", "127.0.0.1"),
            db_port=db_port,
            db_name=os.getenv("DB_NAME", ""),
            db_driver=os.getenv("DB_DRIVER", "ODBC+Driver+17+for+SQL+Server"),
            db_sslmode=os.getenv("DB_SSLMODE", ""),
            db_extra_params=os.getenv("DB_EXTRA_PARAMS", ""),
            llm_model=llm_model,
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            enable_self_healing=os.getenv("ENABLE_SELF_HEALING", "true").lower() in ("true", "1", "yes"),
            enable_schema_pruning=os.getenv("ENABLE_SCHEMA_PRUNING", "true").lower() in ("true", "1", "yes"),
            server_host=os.getenv("SERVER_HOST", "0.0.0.0"),
            server_port=int(os.getenv("SERVER_PORT", "8000")),
        )

        config.validate()
        return config

    def validate(self) -> None:
        """Validate that all required configuration values are present."""
        errors: list[str] = []

        # Validate LLM API key based on selected provider
        provider = self.llm_provider.lower().strip()
        if provider == "groq":
            if not self.groq_api_key:
                errors.append("GROQ_API_KEY is missing from your .env file")
        elif provider == "gemini":
            if not self.google_api_key:
                errors.append("GOOGLE_API_KEY is missing from your .env file")
        elif provider == "openai":
            if not self.openai_api_key:
                errors.append("OPENAI_API_KEY is missing from your .env file")
        elif provider not in ("mock", "fake", "none"):
            if not (self.groq_api_key or self.google_api_key or self.openai_api_key):
                errors.append(f"API key missing for LLM provider '{self.llm_provider}'")

        # If direct DATABASE_URL is provided, discrete connection fields are optional
        if self.database_url:
            if errors:
                raise ValueError(
                    "Configuration errors:\n" + "\n".join(f"  • {e}" for e in errors)
                )
            return

        # SQLite only requires database file name/path
        db_type_lower = self.db_type.lower().strip()
        if db_type_lower in ("sqlite", "sqlite3"):
            if not self.db_name:
                errors.append("DB_NAME (SQLite file path) is missing from your .env file")
        else:
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
        Build a properly formatted and URL-encoded database connection string.

        Supports direct DATABASE_URL strings as well as discrete parameters
        for PostgreSQL, MySQL, MSSQL, SQLite, and CockroachDB.
        """
        # 1. Direct DATABASE_URL override
        if self.database_url:
            url = self.database_url.strip()
            # Normalize postgres:// scheme commonly used by cloud hosts (Heroku, Supabase, Neon)
            if url.startswith("postgres://"):
                url = "postgresql+psycopg2://" + url[len("postgres://"):]
            elif url.startswith("postgresql://") and "+psycopg2" not in url:
                url = "postgresql+psycopg2://" + url[len("postgresql://"):]
            elif url.startswith("mysql://") and "+pymysql" not in url:
                url = "mysql+pymysql://" + url[len("mysql://"):]
            return url

        db_type_lower = self.db_type.lower().strip()

        # 2. SQLite
        if db_type_lower in ("sqlite", "sqlite3"):
            return f"sqlite:///{self.db_name}"

        encoded_user = urllib.parse.quote_plus(self.db_user)
        encoded_password = urllib.parse.quote_plus(self.db_password)
        port = self.db_port or DEFAULT_PORTS.get(db_type_lower)

        # 3. PostgreSQL (Supabase, Neon, AWS RDS Postgres, Google Cloud SQL Postgres, Azure Postgres, Aiven)
        if db_type_lower in ("postgresql", "postgres"):
            port_str = f":{port}" if port else ""
            query_params: list[str] = []
            if self.db_sslmode:
                query_params.append(f"sslmode={self.db_sslmode}")
            if self.db_extra_params:
                query_params.append(self.db_extra_params.lstrip("?&"))
            query_str = f"?{'&'.join(query_params)}" if query_params else ""
            return f"postgresql+psycopg2://{encoded_user}:{encoded_password}@{self.db_server}{port_str}/{self.db_name}{query_str}"

        # 4. MySQL / MariaDB (PlanetScale, AWS RDS MySQL/Aurora, Cloud SQL MySQL, Azure MySQL, Aiven)
        if db_type_lower in ("mysql", "mariadb"):
            port_str = f":{port}" if port else ""
            query_params = []
            if self.db_sslmode:
                query_params.append(f"ssl_mode={self.db_sslmode}")
            if self.db_extra_params:
                query_params.append(self.db_extra_params.lstrip("?&"))
            query_str = f"?{'&'.join(query_params)}" if query_params else ""
            return f"mysql+pymysql://{encoded_user}:{encoded_password}@{self.db_server}{port_str}/{self.db_name}{query_str}"

        # 5. CockroachDB
        if db_type_lower in ("cockroachdb", "cockroach"):
            port_str = f":{port}" if port else ":26257"
            query_params = []
            if self.db_sslmode:
                query_params.append(f"sslmode={self.db_sslmode}")
            if self.db_extra_params:
                query_params.append(self.db_extra_params.lstrip("?&"))
            query_str = f"?{'&'.join(query_params)}" if query_params else ""
            return f"cockroachdb+psycopg2://{encoded_user}:{encoded_password}@{self.db_server}{port_str}/{self.db_name}{query_str}"

        # 6. Microsoft SQL Server / Azure SQL / Cloud SQL SQL Server (default)
        port_str = f":{port}" if self.db_port else ""
        query_params = [f"driver={self.db_driver}", "TrustServerCertificate=yes"]
        if self.db_extra_params:
            query_params.append(self.db_extra_params.lstrip("?&"))
        return (
            f"mssql+pyodbc://{encoded_user}:{encoded_password}"
            f"@{self.db_server}{port_str}/{self.db_name}"
            f"?{'&'.join(query_params)}"
        )
