"""
Tests for agenticsql.config module.

Validates:
- Configuration loading from environment variables
- Validation logic for direct DATABASE_URL vs discrete parameters
- Multi-cloud and multi-dialect connection string generation:
  - Direct URI (Supabase, Neon, PlanetScale, CockroachDB)
  - PostgreSQL (AWS RDS, Cloud SQL, Neon, Supabase)
  - MySQL / MariaDB (PlanetScale, AWS Aurora, Cloud SQL)
  - Microsoft SQL Server / Azure SQL
  - SQLite local files
  - Special character URL encoding
"""

import os
import pytest
from unittest.mock import patch

from agenticsql.config import Config


class TestConfigValidation:
    """Test configuration validation logic."""

    def test_missing_groq_api_key_raises(self):
        """Config validation should fail when GROQ_API_KEY is missing for Groq provider."""
        config = Config(
            llm_provider="groq",
            groq_api_key="",
            db_user="user",
            db_password="pass",
            db_name="testdb",
        )
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            config.validate()

    def test_missing_google_api_key_raises(self):
        """Config validation should fail when GOOGLE_API_KEY is missing for Gemini provider."""
        config = Config(
            llm_provider="gemini",
            google_api_key="",
            db_user="user",
            db_password="pass",
            db_name="testdb",
        )
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            config.validate()

    def test_missing_db_user_raises(self):
        """Config validation should fail when DB_USER is missing for discrete config."""
        config = Config(
            llm_provider="groq",
            groq_api_key="gsk_test",
            db_user="",
            db_password="pass",
            db_name="testdb",
        )
        with pytest.raises(ValueError, match="DB_USER"):
            config.validate()

    def test_missing_db_password_raises(self):
        """Config validation should fail when DB_PASSWORD is missing for discrete config."""
        config = Config(
            llm_provider="groq",
            groq_api_key="gsk_test",
            db_user="user",
            db_password="",
            db_name="testdb",
        )
        with pytest.raises(ValueError, match="DB_PASSWORD"):
            config.validate()

    def test_missing_db_name_raises(self):
        """Config validation should fail when DB_NAME is missing."""
        config = Config(
            llm_provider="groq",
            groq_api_key="gsk_test",
            db_user="user",
            db_password="pass",
            db_name="",
        )
        with pytest.raises(ValueError, match="DB_NAME"):
            config.validate()

    def test_multiple_missing_fields_reported(self):
        """All missing fields should be reported in a single error."""
        config = Config(llm_provider="groq")  # All fields empty
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        error_msg = str(exc_info.value)
        assert "GROQ_API_KEY" in error_msg
        assert "DB_USER" in error_msg
        assert "DB_PASSWORD" in error_msg
        assert "DB_NAME" in error_msg

    def test_valid_discrete_config_passes(self):
        """Config with all required fields should pass validation."""
        config = Config(
            llm_provider="groq",
            groq_api_key="gsk_test",
            db_user="testuser",
            db_password="testpass",
            db_name="testdb",
        )
        config.validate()

    def test_direct_database_url_passes_without_user_pass(self):
        """When DATABASE_URL is present, discrete user/password/server are not required."""
        config = Config(
            llm_provider="groq",
            groq_api_key="gsk_test",
            database_url="postgresql://postgres:secret@db.supabase.co:5432/postgres?sslmode=require",
        )
        config.validate()

    def test_sqlite_only_requires_db_name(self):
        """SQLite requires only db_name and api_key."""
        config = Config(
            llm_provider="groq",
            groq_api_key="gsk_test",
            db_type="sqlite",
            db_name="local_data.db",
        )
        config.validate()


class TestConnectionString:
    """Test connection string building across dialects and cloud providers."""

    def test_basic_mssql_connection_string(self):
        """MSSQL connection string should be properly formatted."""
        config = Config(
            google_api_key="key",
            db_type="mssql",
            db_user="user",
            db_password="password",
            db_server="127.0.0.1",
            db_name="mydb",
            db_driver="ODBC+Driver+17+for+SQL+Server",
        )
        cs = config.connection_string
        assert "mssql+pyodbc://" in cs
        assert "user:password@127.0.0.1/mydb" in cs
        assert "driver=ODBC+Driver+17+for+SQL+Server" in cs
        assert "TrustServerCertificate=yes" in cs

    def test_postgresql_connection_string(self):
        """PostgreSQL string for AWS RDS / Supabase / Neon."""
        config = Config(
            google_api_key="key",
            db_type="postgresql",
            db_user="postgres",
            db_password="secret@password",
            db_server="ep-cool-dawn.us-east-2.aws.neon.tech",
            db_port=5432,
            db_name="neondb",
            db_sslmode="require",
        )
        cs = config.connection_string
        assert cs.startswith("postgresql+psycopg2://")
        assert "postgres:secret%40password@ep-cool-dawn.us-east-2.aws.neon.tech:5432/neondb" in cs
        assert "sslmode=require" in cs

    def test_mysql_connection_string(self):
        """MySQL string for PlanetScale / AWS RDS MySQL."""
        config = Config(
            google_api_key="key",
            db_type="mysql",
            db_user="root",
            db_password="mypassword",
            db_server="mydb.rds.amazonaws.com",
            db_port=3306,
            db_name="sales",
        )
        cs = config.connection_string
        assert cs.startswith("mysql+pymysql://")
        assert "root:mypassword@mydb.rds.amazonaws.com:3306/sales" in cs

    def test_sqlite_connection_string(self):
        """SQLite connection string."""
        config = Config(
            google_api_key="key",
            db_type="sqlite",
            db_name="test.db",
        )
        assert config.connection_string == "sqlite:///test.db"

    def test_direct_database_url_normalization(self):
        """Direct postgres:// URL should be normalized to postgresql+psycopg2://."""
        config = Config(
            google_api_key="key",
            database_url="postgres://user:pass@ep-cool-dawn.aws.neon.tech/neondb?sslmode=require",
        )
        assert config.connection_string.startswith("postgresql+psycopg2://user:pass@ep-cool-dawn.aws.neon.tech/neondb")

    def test_special_chars_in_password_are_encoded(self):
        """Passwords with @ : / and other special chars must be URL-encoded."""
        config = Config(
            google_api_key="key",
            db_type="postgresql",
            db_user="user",
            db_password="P@ss:w/rd!",
            db_server="127.0.0.1",
            db_name="mydb",
        )
        cs = config.connection_string
        assert "P%40ss%3Aw%2Frd%21" in cs
        assert "%40127.0.0.1" not in cs  # Server should NOT be encoded


class TestConfigFromEnv:
    """Test loading configuration from environment variables."""

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "groq",
        "GROQ_API_KEY": "gsk-test-api-key",
        "DATABASE_URL": "postgresql://user:pass@host:5432/dbname?sslmode=require",
        "LLM_MODEL": "llama-3.3-70b-versatile",
        "LLM_TEMPERATURE": "0.5",
        "SERVER_HOST": "localhost",
        "SERVER_PORT": "9000",
    }, clear=False)
    def test_loads_direct_database_url_groq(self):
        """DATABASE_URL should load and validate properly from env for Groq."""
        config = Config.from_env()
        assert config.groq_api_key == "gsk-test-api-key"
        assert config.llm_provider == "groq"
        assert config.database_url.startswith("postgresql://")
        assert config.llm_model == "llama-3.3-70b-versatile"
        assert config.llm_temperature == 0.5
        assert config.server_host == "localhost"
        assert config.server_port == 9000

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "gemini",
        "GOOGLE_API_KEY": "test-api-key",
        "DATABASE_URL": "postgresql://user:pass@host:5432/dbname?sslmode=require",
        "LLM_MODEL": "gemini-2.5-pro",
        "LLM_TEMPERATURE": "0.5",
        "SERVER_HOST": "localhost",
        "SERVER_PORT": "9000",
    }, clear=False)
    def test_loads_direct_database_url_gemini(self):
        """DATABASE_URL should load and validate properly from env for Gemini."""
        config = Config.from_env()
        assert config.google_api_key == "test-api-key"
        assert config.llm_provider == "gemini"
        assert config.database_url.startswith("postgresql://")
        assert config.llm_model == "gemini-2.5-pro"
        assert config.llm_temperature == 0.5
