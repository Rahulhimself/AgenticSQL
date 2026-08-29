"""
Unit and integration tests for Dynamic Database Connection Switching & Schema Isolation.

Verifies:
1. Dynamic connection switching across multiple distinct databases.
2. Tenant schema isolation (tables from DB A do not appear in DB B).
3. Dialect detection and schema introspection for different databases.
4. Cache invalidation on connection switch and deletion.
5. New account isolation (new accounts start clean with zero inherited connections).
6. URI scheme normalization for cloud & local database drivers.
"""

import os
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch
from sqlalchemy import create_engine, text

from agenticsql.auth import AuthDatabase, User
from agenticsql.tenancy import TenantManager
from agenticsql.database import get_schema_info, normalize_connection_uri, connect_from_uri


def create_sqlite_database_with_schema(db_path: Path, table_defs: dict[str, list[str]]) -> None:
    """Helper to populate a SQLite database with specific tables and columns."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    for table_name, cols in table_defs.items():
        col_defs = ", ".join(cols)
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs});")
    conn.commit()
    conn.close()


@patch.dict(os.environ, {
    "DB_TYPE": "sqlite",
    "DB_NAME": "data/test_agenticsql.db",
    "DB_USER": "test_user",
    "DB_PASSWORD": "test_password",
    "LLM_PROVIDER": "mock",
    "GROQ_API_KEY": "gsk_test_key",
})
class TestDynamicDatabaseSwitching:
    """Verify multi-tenant dynamic database connection switching and schema inspection."""

    def test_uri_normalization(self):
        """Test URL scheme normalizations across PostgreSQL, MySQL, MSSQL, and SQLite."""
        assert normalize_connection_uri("postgres://user:pass@localhost:5432/mydb") == "postgresql+psycopg2://user:pass@localhost:5432/mydb"
        assert normalize_connection_uri("postgresql://user:pass@localhost:5432/mydb") == "postgresql+psycopg2://user:pass@localhost:5432/mydb"
        assert normalize_connection_uri("mysql://root:pass@127.0.0.1:3306/appdb") == "mysql+pymysql://root:pass@127.0.0.1:3306/appdb"
        assert normalize_connection_uri("sqlite:///data/test.db") == "sqlite:///data/test.db"
        assert normalize_connection_uri("mssql+pyodbc://user:pass@server/db?driver=ODBC+Driver+17+for+SQL+Server") == "mssql+pyodbc://user:pass@server/db?driver=ODBC+Driver+17+for+SQL+Server"

    def test_dynamic_switching_and_schema_isolation(self, tmp_path):
        """
        Create two distinct SQLite databases with different schemas and verify
        that switching connections dynamically loads the exact corresponding tables.
        """
        # DB 1: E-commerce DB
        db1_path = tmp_path / "ecommerce.db"
        create_sqlite_database_with_schema(db1_path, {
            "ecommerce_orders": ["order_id INTEGER PRIMARY KEY", "customer_name TEXT", "total_amount REAL"],
            "ecommerce_products": ["product_id INTEGER PRIMARY KEY", "title TEXT", "price REAL"],
        })

        # DB 2: HR Management DB
        db2_path = tmp_path / "human_resources.db"
        create_sqlite_database_with_schema(db2_path, {
            "hr_employees": ["emp_id INTEGER PRIMARY KEY", "full_name TEXT", "salary REAL"],
            "hr_departments": ["dept_id INTEGER PRIMARY KEY", "dept_name TEXT"],
            "hr_payrolls": ["payroll_id INTEGER PRIMARY KEY", "emp_id INTEGER", "amount REAL"],
        })

        meta_db = tmp_path / "meta_switching.db"
        auth_db = AuthDatabase(db_path=meta_db)
        tenant_mgr = TenantManager(auth_db=auth_db)

        user = auth_db.register_user("dev_user", "dev@example.com", "DevPassword123!")

        # Register both connections
        conn1 = tenant_mgr.register_connection(
            user_id=user.id,
            name="E-Commerce Database",
            db_type="sqlite",
            db_uri=f"sqlite:///{db1_path.as_posix()}",
            db_name="ecommerce.db",
        )

        conn2 = tenant_mgr.register_connection(
            user_id=user.id,
            name="Human Resources Database",
            db_type="sqlite",
            db_uri=f"sqlite:///{db2_path.as_posix()}",
            db_name="human_resources.db",
        )

        # 1. Connect to DB 1 (E-Commerce) and inspect schema
        db1 = tenant_mgr.connect_tenant_db(user, connection_id=conn1.id)
        schema1 = get_schema_info(db1)

        assert "ecommerce_orders" in schema1
        assert "ecommerce_products" in schema1
        assert "hr_employees" not in schema1
        assert "hr_departments" not in schema1

        # 2. Switch to DB 2 (Human Resources) and inspect schema
        db2 = tenant_mgr.connect_tenant_db(user, connection_id=conn2.id)
        schema2 = get_schema_info(db2)

        assert "hr_employees" in schema2
        assert "hr_departments" in schema2
        assert "hr_payrolls" in schema2
        assert "ecommerce_orders" not in schema2
        assert "ecommerce_products" not in schema2

        # 3. Switch back to DB 1
        db1_again = tenant_mgr.connect_tenant_db(user, connection_id=conn1.id)
        schema1_again = get_schema_info(db1_again)
        assert "ecommerce_orders" in schema1_again
        assert "hr_employees" not in schema1_again

    def test_new_user_starts_with_isolated_connections(self, tmp_path):
        """Verify that newly registered users do not inherit connections from other users."""
        meta_db = tmp_path / "meta_new_user.db"
        auth_db = AuthDatabase(db_path=meta_db)
        tenant_mgr = TenantManager(auth_db=auth_db)

        user_alice = auth_db.register_user("alice", "alice@example.com", "PasswordAlice123!")
        user_bob = auth_db.register_user("bob", "bob@example.com", "PasswordBob123!")

        # Alice registers a connection
        tenant_mgr.register_connection(
            user_id=user_alice.id,
            name="Alice Private DB",
            db_type="sqlite",
            db_uri=f"sqlite:///{tmp_path}/alice.db",
        )

        alice_conns = tenant_mgr.get_user_connections(user_alice.id)
        bob_conns = tenant_mgr.get_user_connections(user_bob.id)

        assert len(alice_conns) == 1
        assert alice_conns[0].name == "Alice Private DB"
        # Bob must have 0 custom connections
        assert len(bob_conns) == 0

    def test_delete_connection_evicts_cache(self, tmp_path):
        """Verify that deleting a connection evicts it from the cache and database store."""
        db_path = tmp_path / "temp_cache.db"
        create_sqlite_database_with_schema(db_path, {"temp_tbl": ["id INT"]})

        meta_db = tmp_path / "meta_cache.db"
        auth_db = AuthDatabase(db_path=meta_db)
        tenant_mgr = TenantManager(auth_db=auth_db)

        user = auth_db.register_user("carol", "carol@example.com", "CarolPass123!")
        conn = tenant_mgr.register_connection(
            user_id=user.id,
            name="Carol DB",
            db_type="sqlite",
            db_uri=f"sqlite:///{db_path.as_posix()}",
        )

        # Connect to populate cache
        db = tenant_mgr.connect_tenant_db(user, connection_id=conn.id)
        assert (user.id, conn.id) in tenant_mgr._db_cache

        # Delete connection
        deleted = tenant_mgr.delete_connection(user.id, conn.id)
        assert deleted is True
        assert (user.id, conn.id) not in tenant_mgr._db_cache
        assert len(tenant_mgr.get_user_connections(user.id)) == 0

        # Attempting to connect to deleted connection ID raises ValueError
        with pytest.raises(ValueError, match="not found"):
            tenant_mgr.connect_tenant_db(user, connection_id=conn.id)
