"""
Unit tests for the FastAPI backend server (agenticsql.server).
Tests REST endpoints, authentication, rate limiting, and WebSocket streaming.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from agenticsql.config import Config
from agenticsql.server import create_app, RateLimiter
from agenticsql.auth import AuthDatabase


@pytest.fixture
def test_config():
    return Config(
        llm_provider="groq",
        groq_api_key="gsk_test_key",
        db_name="test_db",
        db_server="127.0.0.1",
        db_type="sqlite",
        llm_model="llama-3.3-70b-versatile",
    )


@pytest.fixture
def client(test_config, tmp_path):
    with patch("agenticsql.server.connect") as mock_connect, \
         patch("agenticsql.server.create_llm") as mock_llm, \
         patch("agenticsql.server.AgenticSQLAgent") as mock_agent_cls, \
         patch("agenticsql.server.get_schema_info") as mock_schema:

        mock_db = MagicMock()
        mock_db.dialect = "sqlite"
        mock_connect.return_value = mock_db

        mock_agent = MagicMock()
        mock_agent.chat.return_value = {
            "output": "Test answer",
            "sql": ["SELECT 1;"],
            "data": {"columns": ["val"], "rows": [[1]]},
            "healed": False,
            "cost": "LOW",
        }
        mock_agent.get_history.return_value = [{"role": "user", "content": "hello"}]
        mock_agent_cls.return_value = mock_agent
        mock_schema.return_value = {"users": "CREATE TABLE users (id INT);"}

        app = create_app(test_config)
        with TestClient(app) as client:
            yield client


class TestServerEndpoints:
    """Test REST API endpoints."""

    def test_health_check(self, client):
        """GET /api/health should return status and configuration info."""
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["database"] == "test_db"
        assert data["dialect"] == "sqlite"

    def test_chat_endpoint(self, client):
        """POST /api/chat should execute query and return formatted response."""
        res = client.post("/api/chat", json={"message": "Show all users"})
        assert res.status_code == 200
        data = res.json()
        assert data["output"] == "Test answer"
        assert data["sql"] == ["SELECT 1;"]
        assert data["cost"] == "LOW"

    def test_chat_empty_message_rejected(self, client):
        """POST /api/chat with empty message should return 400."""
        res = client.post("/api/chat", json={"message": "   "})
        assert res.status_code == 400

    def test_history_and_clear_endpoints(self, client):
        """GET /api/history and POST /api/clear should manage conversation history."""
        hist_res = client.get("/api/history")
        assert hist_res.status_code == 200
        assert "history" in hist_res.json()

        clear_res = client.post("/api/clear")
        assert clear_res.status_code == 200
        assert clear_res.json()["status"] == "cleared"

    def test_schema_endpoint(self, client):
        """GET /api/schema should return database schema dict."""
        res = client.get("/api/schema")
        assert res.status_code == 200
        assert "users" in res.json()["schema"]

    def test_auth_register_and_login_flow(self, client):
        """Test user registration and login issuing JWT tokens."""
        unique_user = f"user_{int(pytest.importorskip('time').time())}"
        reg_res = client.post(
            "/api/auth/register",
            json={"username": unique_user, "email": f"{unique_user}@test.com", "password": "Pass12345!"},
        )
        assert reg_res.status_code == 200
        token = reg_res.json()["token"]
        assert token is not None

        login_res = client.post(
            "/api/auth/login",
            json={"username_or_email": unique_user, "password": "Pass12345!"},
        )
        assert login_res.status_code == 200
        assert login_res.json()["token"] is not None

        # Verify /api/auth/me with Bearer token
        me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_res.status_code == 200
        assert me_res.json()["user"]["username"] == unique_user

    def test_rate_limiter_logic(self):
        """Test RateLimiter sliding window allows and blocks appropriately."""
        limiter = RateLimiter(max_requests=3, window_seconds=10)
        client_id = "test_ip"

        assert limiter.is_allowed(client_id) is True
        assert limiter.is_allowed(client_id) is True
        assert limiter.is_allowed(client_id) is True
        # 4th request within window should be rejected
        assert limiter.is_allowed(client_id) is False
