"""
Authentication, Password Security, JWT, and Metadata Store for AgenticSQL (Phase 4c).

Provides:
- PBKDF2-HMAC-SHA256 salted password hashing & verification
- HMAC-SHA256 JWT generation and validation
- SQLite metadata database for user accounts, registered tenant databases,
  per-user query history, and admin usage statistics.
"""

import os
import hmac
import time
import json
import base64
import hashlib
import secrets
import sqlite3
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)

DEFAULT_META_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "agenticsql_meta.db"
DEFAULT_JWT_SECRET = os.getenv("JWT_SECRET", "agenticsql-super-secret-jwt-key-2026")


# --- Security & Cryptography ---


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """
    Hash a password with PBKDF2-HMAC-SHA256 and a random salt.

    Returns:
        (password_hash, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)

    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000,
    ).hex()
    return pwd_hash, salt


def verify_password(password: str, expected_hash: str, salt: str) -> bool:
    """Verify a plain password against an expected PBKDF2 hash and salt."""
    actual_hash, _ = hash_password(password, salt=salt)
    return hmac.compare_digest(actual_hash, expected_hash)


def create_jwt_token(payload: dict, secret: str = DEFAULT_JWT_SECRET, expires_in_seconds: int = 86400) -> str:
    """
    Create a signed JWT token using HMAC-SHA256.
    """
    header = {"alg": "HS256", "typ": "JWT"}
    payload_copy = payload.copy()
    payload_copy["exp"] = int(time.time()) + expires_in_seconds
    payload_copy["iat"] = int(time.time())

    def b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

    header_b64 = b64url(json.dumps(header).encode("utf-8"))
    payload_b64 = b64url(json.dumps(payload_copy).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}"

    signature = hmac.new(secret.encode("utf-8"), signing_input.encode("utf-8"), hashlib.sha256).digest()
    sig_b64 = b64url(signature)

    return f"{signing_input}.{sig_b64}"


def decode_jwt_token(token: str, secret: str = DEFAULT_JWT_SECRET) -> Optional[dict]:
    """
    Verify and decode a signed JWT token.

    Returns payload dict if valid and unexpired, or None if invalid.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}"

        # Verify signature
        expected_sig = hmac.new(secret.encode("utf-8"), signing_input.encode("utf-8"), hashlib.sha256).digest()

        def b64url_decode(s: str) -> bytes:
            padding = 4 - (len(s) % 4)
            if padding != 4:
                s += "=" * padding
            return base64.urlsafe_b64decode(s.encode("utf-8"))

        actual_sig = b64url_decode(sig_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            logger.warning("JWT signature verification failed.")
            return None

        payload_bytes = b64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))

        # Verify expiration
        exp = payload.get("exp", 0)
        if time.time() > exp:
            logger.warning("JWT token has expired.")
            return None

        return payload
    except Exception as e:
        logger.warning("JWT decode error: %s", e)
        return None


# --- Data Models ---


@dataclass
class User:
    """User account data model."""
    id: int
    username: str
    email: str
    role: str = "user"  # "user" or "admin"
    created_at: str = ""


@dataclass
class UserConnection:
    """A registered database connection for a user."""
    id: int
    user_id: int
    name: str
    db_type: str  # "mssql", "postgresql", "mysql", "sqlite"
    db_uri: str
    db_server: str = ""
    db_name: str = ""
    is_default: bool = False
    created_at: str = ""


@dataclass
class UserQueryRecord:
    """A historical query executed by a user."""
    id: int
    user_id: int
    connection_id: Optional[int]
    question: str
    sql: str
    latency: float
    status: str = "SUCCESS"  # "SUCCESS", "BLOCKED", "ERROR"
    created_at: str = ""


# --- Metadata Database Store ---


class AuthDatabase:
    """
    SQLite-backed store for user authentication, per-user database connections,
    and query execution history.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_META_DB_PATH
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize metadata tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL
                );
            """)

            # User registered database connections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_connections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    db_type TEXT NOT NULL,
                    db_uri TEXT NOT NULL,
                    db_server TEXT,
                    db_name TEXT,
                    is_default INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
            """)

            # User query history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    connection_id INTEGER,
                    question TEXT NOT NULL,
                    sql TEXT NOT NULL,
                    latency REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
            """)
            conn.commit()

        # Seed initial admin account if users table is empty
        self._seed_default_admin()

    def _seed_default_admin(self) -> None:
        """Create initial admin account (admin / Admin@12345) if no users exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users;")
            count = cursor.fetchone()[0]
            if count == 0:
                pwd_hash, salt = hash_password("Admin@12345")
                now = datetime.now(timezone.utc).isoformat()
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash, salt, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    ("admin", "admin@agenticsql.ai", pwd_hash, salt, "admin", now),
                )
                conn.commit()
                logger.info("Initialized default administrator account (username: admin).")

    # --- User Management ---

    def register_user(self, username: str, email: str, password: str, role: str = "user") -> User:
        """Register a new user account."""
        username_clean = username.strip().lower()
        email_clean = email.strip().lower()

        pwd_hash, salt = hash_password(password)
        now = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash, salt, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (username_clean, email_clean, pwd_hash, salt, role, now),
                )
                conn.commit()
                user_id = cursor.lastrowid
                return User(id=user_id, username=username_clean, email=email_clean, role=role, created_at=now)
            except sqlite3.IntegrityError as e:
                raise ValueError(f"Username or email already registered: {e}")

    def authenticate_user(self, username_or_email: str, password: str) -> Optional[User]:
        """Authenticate user credentials and return User object if valid."""
        clean_input = username_or_email.strip().lower()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, email, password_hash, salt, role, created_at FROM users WHERE username = ? OR email = ?",
                (clean_input, clean_input),
            )
            row = cursor.fetchone()
            if not row:
                return None

            if verify_password(password, row["password_hash"], row["salt"]):
                return User(
                    id=row["id"],
                    username=row["username"],
                    email=row["email"],
                    role=row["role"],
                    created_at=row["created_at"],
                )
            return None

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Fetch user by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, email, role, created_at FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return User(
                    id=row["id"],
                    username=row["username"],
                    email=row["email"],
                    role=row["role"],
                    created_at=row["created_at"],
                )
            return None

    # --- Per-User Database Connections ---

    def add_user_connection(
        self,
        user_id: int,
        name: str,
        db_type: str,
        db_uri: str,
        db_server: str = "",
        db_name: str = "",
        is_default: bool = False,
    ) -> UserConnection:
        """Register a new database connection under a user's account."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO user_connections (user_id, name, db_type, db_uri, db_server, db_name, is_default, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, name.strip(), db_type.lower().strip(), db_uri.strip(), db_server.strip(), db_name.strip(), 1 if is_default else 0, now),
            )
            conn.commit()
            conn_id = cursor.lastrowid
            return UserConnection(
                id=conn_id,
                user_id=user_id,
                name=name.strip(),
                db_type=db_type.lower().strip(),
                db_uri=db_uri.strip(),
                db_server=db_server.strip(),
                db_name=db_name.strip(),
                is_default=is_default,
                created_at=now,
            )

    def get_user_connections(self, user_id: int) -> list[UserConnection]:
        """Fetch all database connections registered by a specific user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, user_id, name, db_type, db_uri, db_server, db_name, is_default, created_at FROM user_connections WHERE user_id = ? ORDER BY id DESC",
                (user_id,),
            )
            return [
                UserConnection(
                    id=row["id"],
                    user_id=row["user_id"],
                    name=row["name"],
                    db_type=row["db_type"],
                    db_uri=row["db_uri"],
                    db_server=row["db_server"] or "",
                    db_name=row["db_name"] or "",
                    is_default=bool(row["is_default"]),
                    created_at=row["created_at"],
                )
                for row in cursor.fetchall()
            ]

    def delete_user_connection(self, user_id: int, connection_id: int) -> bool:
        """Delete a database connection registered by a user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM user_connections WHERE id = ? AND user_id = ?",
                (connection_id, user_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    # --- Per-User Query History ---

    def log_query(
        self,
        user_id: int,
        question: str,
        sql: str,
        latency: float,
        connection_id: Optional[int] = None,
        status: str = "SUCCESS",
    ) -> UserQueryRecord:
        """Log a user's executed query into the metadata store."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO user_queries (user_id, connection_id, question, sql, latency, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, connection_id, question, sql, latency, status, now),
            )
            conn.commit()
            record_id = cursor.lastrowid
            return UserQueryRecord(
                id=record_id,
                user_id=user_id,
                connection_id=connection_id,
                question=question,
                sql=sql,
                latency=latency,
                status=status,
                created_at=now,
            )

    def get_user_query_history(self, user_id: int, limit: int = 50) -> list[UserQueryRecord]:
        """Fetch the recent query history for a user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, user_id, connection_id, question, sql, latency, status, created_at FROM user_queries WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            )
            return [
                UserQueryRecord(
                    id=row["id"],
                    user_id=row["user_id"],
                    connection_id=row["connection_id"],
                    question=row["question"],
                    sql=row["sql"],
                    latency=row["latency"],
                    status=row["status"],
                    created_at=row["created_at"],
                )
                for row in cursor.fetchall()
            ]

    # --- Admin Usage Statistics ---

    def get_admin_stats(self) -> dict:
        """Calculate system-wide aggregate usage statistics for administrators."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM users;")
            total_users = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM user_connections;")
            total_connections = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM user_queries;")
            total_queries = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM user_queries WHERE status = 'BLOCKED';")
            blocked_queries = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(latency) FROM user_queries WHERE latency > 0;")
            avg_latency_row = cursor.fetchone()[0]
            avg_latency = round(avg_latency_row, 2) if avg_latency_row is not None else 0.0

            cursor.execute("""
                SELECT u.username, COUNT(q.id) as query_count
                FROM users u
                LEFT JOIN user_queries q ON u.id = q.user_id
                GROUP BY u.id
                ORDER BY query_count DESC
                LIMIT 10;
            """)
            user_activity = [{"username": row["username"], "queries": row["query_count"]} for row in cursor.fetchall()]

            cursor.execute("""
                SELECT db_type, COUNT(*) as count
                FROM user_connections
                GROUP BY db_type;
            """)
            db_distribution = [{"dialect": row["db_type"], "count": row["count"]} for row in cursor.fetchall()]

            return {
                "total_users": total_users,
                "total_connections": total_connections,
                "total_queries": total_queries,
                "blocked_queries": blocked_queries,
                "safe_queries": max(0, total_queries - blocked_queries),
                "avg_latency_seconds": avg_latency,
                "top_users": user_activity,
                "database_dialects": db_distribution,
            }
