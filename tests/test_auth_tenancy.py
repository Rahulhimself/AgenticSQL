"""
Unit tests for Phase 4c: Auth, Multi-Tenancy, Per-User History, and Admin Analytics.
"""

import time
import pytest
from pathlib import Path

from agenticsql.auth import (
    hash_password,
    verify_password,
    create_jwt_token,
    decode_jwt_token,
    AuthDatabase,
    User,
    UserConnection,
)
from agenticsql.tenancy import TenantManager


class TestCryptographyAndJWT:
    """Test PBKDF2 password hashing and JWT token signing."""

    def test_hash_and_verify_password(self):
        """Test salted password hashing and verification."""
        pwd = "SecurePassword@2026"
        pwd_hash, salt = hash_password(pwd)

        assert pwd_hash != pwd
        assert len(salt) == 32
        assert verify_password(pwd, pwd_hash, salt) is True
        assert verify_password("WrongPassword", pwd_hash, salt) is False

    def test_jwt_create_and_decode_valid(self):
        """Test generating and decoding a valid JWT token."""
        payload = {"sub": 42, "username": "alice", "role": "admin"}
        secret = "test-secret-key-123"

        token = create_jwt_token(payload, secret=secret, expires_in_seconds=3600)
        assert isinstance(token, str)
        assert token.count(".") == 2

        decoded = decode_jwt_token(token, secret=secret)
        assert decoded is not None
        assert decoded["sub"] == 42
        assert decoded["username"] == "alice"
        assert decoded["role"] == "admin"

    def test_jwt_rejects_tampered_signature(self):
        """Test that tampered tokens are rejected."""
        payload = {"sub": 1, "role": "user"}
        token = create_jwt_token(payload, secret="secret-a")

        # Attempt to decode with wrong secret
        decoded = decode_jwt_token(token, secret="secret-b")
        assert decoded is None

    def test_jwt_rejects_expired_token(self):
        """Test that expired tokens are rejected."""
        payload = {"sub": 1}
        token = create_jwt_token(payload, secret="secret", expires_in_seconds=-10)

        decoded = decode_jwt_token(token, secret="secret")
        assert decoded is None


class TestAuthDatabase:
    """Test user registration, authentication, and metadata persistence."""

    def test_bootstrap_default_admin(self, tmp_path):
        """Test that an initial admin account is created if DB is fresh."""
        db_file = tmp_path / "meta_test.db"
        auth_db = AuthDatabase(db_path=db_file)

        admin_user = auth_db.authenticate_user("admin", "Admin@12345")
        assert admin_user is not None
        assert admin_user.username == "admin"
        assert admin_user.role == "admin"

    def test_register_and_authenticate_user(self, tmp_path):
        """Test registering a new standard user and authenticating."""
        db_file = tmp_path / "meta_test.db"
        auth_db = AuthDatabase(db_path=db_file)

        user = auth_db.register_user(
            username="bob",
            email="bob@example.com",
            password="BobPassword123!",
            role="user",
        )
        assert user.id is not None
        assert user.username == "bob"
        assert user.role == "user"

        # Authenticate via username
        auth_by_user = auth_db.authenticate_user("bob", "BobPassword123!")
        assert auth_by_user is not None
        assert auth_by_user.id == user.id

        # Authenticate via email
        auth_by_email = auth_db.authenticate_user("bob@example.com", "BobPassword123!")
        assert auth_by_email is not None

        # Rejects bad password
        assert auth_db.authenticate_user("bob", "WrongPassword") is None

    def test_prevent_duplicate_registration(self, tmp_path):
        """Test that duplicate usernames/emails raise ValueError."""
        db_file = tmp_path / "meta_test.db"
        auth_db = AuthDatabase(db_path=db_file)

        auth_db.register_user("charlie", "charlie@test.com", "pass123")
        with pytest.raises(ValueError):
            auth_db.register_user("charlie", "other@test.com", "pass123")


class TestMultiTenancyAndHistory:
    """Test per-user database connections, query isolation, and admin metrics."""

    def test_per_user_database_connections(self, tmp_path):
        """Test registering and listing database connections per user."""
        db_file = tmp_path / "meta_test.db"
        auth_db = AuthDatabase(db_path=db_file)

        u1 = auth_db.register_user("user1", "u1@test.com", "pass1")
        u2 = auth_db.register_user("user2", "u2@test.com", "pass2")

        conn1 = auth_db.add_user_connection(
            user_id=u1.id,
            name="U1 Analytics DB",
            db_type="sqlite",
            db_uri="sqlite:///u1.db",
        )
        conn2 = auth_db.add_user_connection(
            user_id=u2.id,
            name="U2 Postgres DB",
            db_type="postgresql",
            db_uri="postgresql://user:pass@host/db",
        )

        u1_conns = auth_db.get_user_connections(u1.id)
        assert len(u1_conns) == 1
        assert u1_conns[0].name == "U1 Analytics DB"

        u2_conns = auth_db.get_user_connections(u2.id)
        assert len(u2_conns) == 1
        assert u2_conns[0].name == "U2 Postgres DB"

    def test_per_user_query_history_isolation(self, tmp_path):
        """Test that query history is strictly isolated per tenant."""
        db_file = tmp_path / "meta_test.db"
        auth_db = AuthDatabase(db_path=db_file)

        u1 = auth_db.register_user("user_a", "ua@test.com", "pass")
        u2 = auth_db.register_user("user_b", "ub@test.com", "pass")

        # Log queries for User A
        auth_db.log_query(u1.id, "Total sales?", "SELECT SUM(amount) FROM sales;", latency=0.45)
        auth_db.log_query(u1.id, "Active customers?", "SELECT COUNT(*) FROM users;", latency=0.30)

        # Log queries for User B
        auth_db.log_query(u2.id, "Server health?", "SELECT 1;", latency=0.05)

        u1_history = auth_db.get_user_query_history(u1.id)
        assert len(u1_history) == 2
        assert all(h.user_id == u1.id for h in u1_history)

        u2_history = auth_db.get_user_query_history(u2.id)
        assert len(u2_history) == 1
        assert u2_history[0].question == "Server health?"

    def test_admin_aggregate_statistics(self, tmp_path):
        """Test admin metrics calculation across users, connections, and queries."""
        db_file = tmp_path / "meta_test.db"
        auth_db = AuthDatabase(db_path=db_file)

        u = auth_db.register_user("analyst", "analyst@test.com", "pass")
        auth_db.add_user_connection(u.id, "Sales DB", "mssql", "mssql://localhost/sales")
        auth_db.add_user_connection(u.id, "Logs DB", "sqlite", "sqlite:///logs.db")

        auth_db.log_query(u.id, "Query 1", "SELECT 1;", latency=1.20, status="SUCCESS")
        auth_db.log_query(u.id, "Bad Query", "DROP TABLE users;", latency=0.01, status="BLOCKED")

        stats = auth_db.get_admin_stats()
        # default admin + 1 registered user = 2
        assert stats["total_users"] == 2
        assert stats["total_connections"] == 2
        assert stats["total_queries"] == 2
        assert stats["blocked_queries"] == 1
        assert stats["safe_queries"] == 1
        assert stats["avg_latency_seconds"] > 0
        assert len(stats["database_dialects"]) == 2


class TestTenantManager:
    """Test TenantManager dynamic database connection handling."""

    def test_tenant_manager_connection_registration(self, tmp_path):
        """Test registering and retrieving connections via TenantManager."""
        db_file = tmp_path / "meta_test.db"
        auth_db = AuthDatabase(db_path=db_file)
        tenant_mgr = TenantManager(auth_db=auth_db)

        user = auth_db.register_user("dave", "dave@test.com", "pass")
        conn = tenant_mgr.register_connection(
            user_id=user.id,
            name="Dave SQLite",
            db_type="sqlite",
            db_uri=f"sqlite:///{tmp_path}/dave.db",
        )

        user_conns = tenant_mgr.get_user_connections(user.id)
        assert len(user_conns) == 1
        assert user_conns[0].id == conn.id
