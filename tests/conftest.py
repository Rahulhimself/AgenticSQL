"""
Pytest configuration and global environment fixtures for AgenticSQL test suite.
"""

import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up safe fallback environment variables for test execution."""
    os.environ.setdefault("LLM_PROVIDER", "mock")
    os.environ.setdefault("GROQ_API_KEY", "gsk_test_mock_key_12345")
    os.environ.setdefault("DB_TYPE", "sqlite")
    os.environ.setdefault("DB_NAME", "data/test_agenticsql.db")
    os.environ.setdefault("DB_USER", "test_user")
    os.environ.setdefault("DB_PASSWORD", "test_password")
    os.environ.setdefault("DB_SERVER", "127.0.0.1")
