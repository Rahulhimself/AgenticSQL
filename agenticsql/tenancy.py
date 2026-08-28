"""
Multi-Tenancy Connection Manager & Session Context for AgenticSQL (Phase 4c).

Manages per-user database connections, dynamic switching, agent caching,
and query history logging for isolated tenant environments.
"""

import logging
from typing import Optional, Any
from pathlib import Path

# pyrefly: ignore [missing-import]
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

from .auth import AuthDatabase, User, UserConnection, UserQueryRecord
from .config import Config
from .database import connect as default_connect
from .llm import create_llm
from .agent import AgenticSQLAgent

logger = logging.getLogger(__name__)


class TenantManager:
    """
    Manages tenant contexts, dynamic database connections, and per-user agent caching.
    """

    def __init__(self, auth_db: Optional[AuthDatabase] = None):
        self.auth_db = auth_db or AuthDatabase()
        # In-memory agent cache: key is (user_id, connection_id_or_default)
        self._agent_cache: dict[tuple[int, Optional[int]], AgenticSQLAgent] = {}
        self._db_cache: dict[tuple[int, Optional[int]], SQLDatabase] = {}

    def get_user_connections(self, user_id: int) -> list[UserConnection]:
        """Fetch all registered database connections for a user."""
        return self.auth_db.get_user_connections(user_id)

    def register_connection(
        self,
        user_id: int,
        name: str,
        db_type: str,
        db_uri: str,
        db_server: str = "",
        db_name: str = "",
        is_default: bool = False,
    ) -> UserConnection:
        """Register a new database connection for a user."""
        return self.auth_db.add_user_connection(
            user_id=user_id,
            name=name,
            db_type=db_type,
            db_uri=db_uri,
            db_server=db_server,
            db_name=db_name,
            is_default=is_default,
        )

    def delete_connection(self, user_id: int, connection_id: int) -> bool:
        """Delete a user connection and evict from cached agents and engines."""
        cache_key = (user_id, connection_id)
        self._db_cache.pop(cache_key, None)
        self._agent_cache.pop(cache_key, None)
        return self.auth_db.delete_user_connection(user_id, connection_id)

    def connect_tenant_db(
        self,
        user: User,
        connection_id: Optional[int] = None,
        config: Optional[Config] = None,
    ) -> SQLDatabase:
        """
        Connect to the target database for the tenant.

        If connection_id is None, uses the system default database from Config.
        """
        cache_key = (user.id, connection_id)
        if cache_key in self._db_cache:
            return self._db_cache[cache_key]

        cfg = config or Config.from_env()

        if connection_id is None:
            # Use system default database
            db = default_connect(cfg)
            self._db_cache[cache_key] = db
            return db

        # Look up user connection
        connections = self.auth_db.get_user_connections(user.id)
        matched = next((c for c in connections if c.id == connection_id), None)
        if not matched:
            raise ValueError(f"Connection ID {connection_id} not found for user {user.username}")

        # Connect to tenant database URI
        try:
            logger.info("Connecting to custom database '%s' for user '%s'", matched.name, user.username)
            engine = create_engine(matched.db_uri)
            db = SQLDatabase(engine)
            # Tag dialect attribute for downstream prompt helpers
            db.dialect = matched.db_type
            self._db_cache[cache_key] = db
            return db
        except Exception as e:
            logger.error("Failed to connect to tenant database '%s': %s", matched.name, e)
            raise RuntimeError(f"Could not connect to database '{matched.name}': {e}")

    def get_agent(
        self,
        user: User,
        connection_id: Optional[int] = None,
        config: Optional[Config] = None,
    ) -> AgenticSQLAgent:
        """
        Retrieve or initialize an AgenticSQLAgent configured for the tenant.
        """
        cache_key = (user.id, connection_id)
        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]

        cfg = config or Config.from_env()
        db = self.connect_tenant_db(user, connection_id=connection_id, config=cfg)
        llm = create_llm(cfg)

        agent = AgenticSQLAgent(
            llm=llm,
            db=db,
            verbose=False,
            max_retries=cfg.max_retries,
            enable_self_healing=cfg.enable_self_healing,
            enable_schema_pruning=cfg.enable_schema_pruning,
        )

        self._agent_cache[cache_key] = agent
        return agent

    def execute_tenant_chat(
        self,
        user: User,
        question: str,
        connection_id: Optional[int] = None,
        config: Optional[Config] = None,
    ) -> dict:
        """
        Execute a natural language query for a user and log execution into history.
        """
        import time

        agent = self.get_agent(user, connection_id=connection_id, config=config)

        start_t = time.time()
        response = agent.chat(question)
        latency = round(time.time() - start_t, 2)

        sql_executed = "; ".join(response.get("sql", [])) if response.get("sql") else ""
        status = "BLOCKED" if "BLOCKED" in response.get("output", "") else ("ERROR" if "[ERROR]" in response.get("output", "") else "SUCCESS")

        # Persist query to tenant history
        self.auth_db.log_query(
            user_id=user.id,
            connection_id=connection_id,
            question=question,
            sql=sql_executed,
            latency=latency,
            status=status,
        )

        return response
