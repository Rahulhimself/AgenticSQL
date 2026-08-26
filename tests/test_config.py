"""
Tests for agenticsql.config module.

Validates configuration loading, validation, and connection string building.
"""

import os
import pytest
from unittest.mock import patch

from agenticsql.config import Config


class TestConfigValidation:
    """Test configuration validation logic."""

    def test_missing_api_key_raises(self):
        """Config validation should fail when GOOGLE_API_KEY is missing."""
        config = Config(
            google_api_key="",
            db_user="user",
            db_password="pass",
            db_name="testdb",
        )
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            config.validate()

    def test_missing_db_user_raises(self):
        """Config validation should fail when DB_USER is missing."""
        config = Config(
            google_api_key="key",
            db_user="",
            db_password="pass",
            db_name="testdb",
        )
        with pytest.raises(ValueError, match="DB_USER"):
            config.validate()

    def test_missing_db_password_raises(self):
        """Config validation should fail when DB_PASSWORD is missing."""
        config = Config(
            google_api_key="key",
            db_user="user",
            db_password="",
            db_name="testdb",
        )
        with pytest.raises(ValueError, match="DB_PASSWORD"):
            config.validate()

    def test_missing_db_name_raises(self):
        """Config validation should fail when DB_NAME is missing."""
        config = Config(
            google_api_key="key",
            db_user="user",
            db_password="pass",
            db_name="",
        )
        with pytest.raises(ValueError, match="DB_NAME"):
            config.validate()

    def test_multiple_missing_fields_reported(self):
        """All missing fields should be reported in a single error."""
        config = Config()  # All fields empty
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        error_msg = str(exc_info.value)
        assert "GOOGLE_API_KEY" in error_msg
        assert "DB_USER" in error_msg
        assert "DB_PASSWORD" in error_msg
        assert "DB_NAME" in error_msg

    def test_valid_config_passes(self):
        """Config with all required fields should pass validation."""
        config = Config(
            google_api_key="test-key",
            db_user="testuser",
            db_password="testpass",
            db_name="testdb",
        )
        # Should not raise
        config.validate()


class TestConnectionString:
    """Test connection string building with URL encoding."""

    def test_basic_connection_string(self):
        """Connection string should be properly formatted."""
        config = Config(
            google_api_key="key",
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

    def test_special_chars_in_password_are_encoded(self):
        """Passwords with @ : / and other special chars must be URL-encoded."""
        config = Config(
            google_api_key="key",
            db_user="user",
            db_password="P@ss:w/rd!",
            db_server="127.0.0.1",
            db_name="mydb",
        )
        cs = config.connection_string
        # The raw @ should NOT appear between user: and @server
        # Instead it should be encoded as %40
        assert "P%40ss%3Aw%2Frd%21" in cs
        # The structural @ (separating creds from host) should still be present
        assert "%40127.0.0.1" not in cs  # Server should NOT be encoded

    def test_at_sign_in_password_encoded(self):
        """The specific bug case: password with @ should be properly encoded."""
        config = Config(
            google_api_key="key",
            db_user="langchain_agent",
            db_password="Rahul@093609",
            db_server="127.0.0.1",
            db_name="sql_practise",
        )
        cs = config.connection_string
        # @ in password should be encoded as %40
        assert "Rahul%40093609" in cs
        # Should still have exactly one structural @ for user:pass@host
        parts = cs.split("//")[1]  # Remove scheme
        at_count = parts.count("@")
        assert at_count == 1, f"Expected 1 structural '@' but found {at_count} in: {parts}"


class TestConfigFromEnv:
    """Test loading configuration from environment variables."""

    @patch.dict(os.environ, {
        "GOOGLE_API_KEY": "test-api-key",
        "DB_USER": "test-user",
        "DB_PASSWORD": "test-pass",
        "DB_SERVER": "192.168.1.1",
        "DB_NAME": "test_db",
        "DB_DRIVER": "ODBC+Driver+18+for+SQL+Server",
        "LLM_MODEL": "gemini-2.5-pro",
        "LLM_TEMPERATURE": "0.5",
        "SERVER_HOST": "localhost",
        "SERVER_PORT": "9000",
    }, clear=False)
    def test_loads_all_env_vars(self):
        """All env vars should be loaded into the config."""
        config = Config.from_env()
        assert config.google_api_key == "test-api-key"
        assert config.db_user == "test-user"
        assert config.db_password == "test-pass"
        assert config.db_server == "192.168.1.1"
        assert config.db_name == "test_db"
        assert config.db_driver == "ODBC+Driver+18+for+SQL+Server"
        assert config.llm_model == "gemini-2.5-pro"
        assert config.llm_temperature == 0.5
        assert config.server_host == "localhost"
        assert config.server_port == 9000
