"""
Database connection and schema management.

Provides connection initialization with multi-dialect support (PostgreSQL,
MySQL, SQL Server, SQLite, CockroachDB), cloud provider diagnostics,
health checks, and schema introspection helpers.
"""

import logging
from typing import Optional
# pyrefly: ignore [missing-import]
from langchain_community.utilities import SQLDatabase

from .config import Config

logger = logging.getLogger(__name__)


def _get_troubleshooting_tips(config: Config, error: Exception) -> list[str]:
    """Generate dialect- and provider-aware troubleshooting steps for connection errors."""
    db_type = config.db_type.lower().strip()
    is_uri = bool(config.database_url)

    if is_uri:
        return [
            "Verify your DATABASE_URL format and credentials.",
            "If using Supabase/Neon/CockroachDB, ensure '?sslmode=require' is included.",
            "Check if your IP address is allowed in your cloud provider's network/firewall settings.",
            "Ensure the required driver is installed (psycopg2-binary for Postgres, pymysql for MySQL).",
        ]

    if db_type in ("sqlite", "sqlite3"):
        return [
            f"Does the SQLite database file '{config.db_name}' exist?",
            "Do you have read/write file system permissions for the file location?",
        ]

    if db_type in ("postgresql", "postgres", "cockroachdb", "cockroach"):
        return [
            f"Is the PostgreSQL/CockroachDB server reachable at {config.db_server}:{config.db_port or 5432}?",
            "If connecting to Supabase/Neon/AWS RDS/Cloud SQL, set DB_SSLMODE='require'.",
            "Are DB_USER and DB_PASSWORD correct in .env?",
            "Is 'psycopg2-binary' installed? (run: pip install psycopg2-binary)",
            "Check cloud provider firewall / inbound IP allowlists (e.g. AWS Security Group, GCP Authorized Networks).",
        ]

    if db_type in ("mysql", "mariadb"):
        return [
            f"Is the MySQL/MariaDB server reachable at {config.db_server}:{config.db_port or 3306}?",
            "If connecting to PlanetScale or AWS Aurora, verify SSL connection parameters.",
            "Are DB_USER and DB_PASSWORD correct in .env?",
            "Is 'pymysql' installed? (run: pip install pymysql cryptography)",
            "Check if user has remote host connection privileges (e.g. 'user'@'%').",
        ]

    # Default: MS SQL Server / Azure SQL
    return [
        f"Is SQL Server / Azure SQL reachable at {config.db_server}?",
        "Is SQL Server Authentication enabled (for local instances)?",
        "Are DB_USER and DB_PASSWORD correct in .env?",
        f"Is '{config.db_driver}' installed on your system?",
        "For Azure SQL, ensure client IP is added to Azure Firewall rules.",
        "Can you connect using SSMS, Azure Data Studio, or sqlcmd?",
    ]


def connect(config: Config) -> SQLDatabase:
    """
    Connect to the database with error handling and a health check.

    Supports local instances and cloud databases across AWS RDS, Azure SQL,
    Google Cloud SQL, Supabase, Neon, PlanetScale, and CockroachDB.

    Args:
        config: Application configuration with database connection details.

    Returns:
        An initialized SQLDatabase instance.

    Raises:
        ConnectionError: If the connection fails, with tailored troubleshooting tips.
    """
    target_desc = config.database_url or f"{config.db_server}/{config.db_name} ({config.db_type})"
    # Obfuscate credentials for logging
    logger.info("Connecting to database: %s (user=%s)...", config.db_server or "direct-URI", config.db_user or "URI-defined")

    try:
        db = SQLDatabase.from_uri(config.connection_string)

        # Health check
        db.run("SELECT 1")
        logger.info("Database connection successful. Dialect: %s", db.dialect)
        return db

    except Exception as e:
        tips = _get_troubleshooting_tips(config, e)
        tips_formatted = "\n".join(f"  {i+1}. {tip}" for i, tip in enumerate(tips))
        raise ConnectionError(
            f"Failed to connect to database '{config.db_name or 'target'}' "
            f"via '{config.db_server or 'URI'}': {e}\n\n"
            f"Troubleshooting:\n{tips_formatted}"
        ) from e


def normalize_connection_uri(uri: str) -> str:
    """
    Normalize connection URI schemes so SQLAlchemy uses the appropriate drivers.
    
    Transforms:
      - postgres:// -> postgresql+psycopg2://
      - postgresql:// (without driver) -> postgresql+psycopg2://
      - mysql:// (without driver) -> mysql+pymysql://
    """
    url = (uri or "").strip()
    if url.startswith("postgres://"):
        url = "postgresql+psycopg2://" + url[len("postgres://"):]
    elif url.startswith("postgresql://") and "+psycopg2" not in url and "+asyncpg" not in url and "+pg8000" not in url:
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]
    elif url.startswith("mysql://") and "+pymysql" not in url and "+mysqldb" not in url and "+mariadbconnector" not in url:
        url = "mysql+pymysql://" + url[len("mysql://"):]
    return url


def connect_from_uri(uri: str) -> SQLDatabase:
    """
    Connect to a database via URI string with error handling and a health check.

    Args:
        uri: Database connection URI string.

    Returns:
        An initialized SQLDatabase instance.
    """
    norm_uri = normalize_connection_uri(uri)
    logger.info("Connecting to database from URI...")
    try:
        db = SQLDatabase.from_uri(norm_uri)
        db.run("SELECT 1")
        logger.info("Database connection successful. Dialect: %s", getattr(db, "dialect", "unknown"))
        return db
    except Exception as e:
        logger.error("Failed to connect to database via URI: %s", e)
        raise ConnectionError(f"Could not connect to database via URI: {e}") from e


def get_schema_info(db: SQLDatabase) -> dict[str, str]:
    """
    Get structured schema information from the database.

    Returns a dict mapping table names to their schema descriptions
    (CREATE TABLE statements with column types).
    """
    tables = db.get_usable_table_names()
    schema: dict[str, str] = {}

    for table in tables:
        try:
            info = db.get_table_info(table_names=[table])
            schema[table] = info
        except Exception as e:
            logger.warning("Could not get schema for table '%s': %s", table, e)
            schema[table] = f"Error retrieving schema: {e}"

    return schema

