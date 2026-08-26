"""
Database connection and schema management.

Provides connection initialization with error handling, retry logic,
and schema introspection helpers.
"""

import logging
from typing import Optional
# pyrefly: ignore [missing-import]
from langchain_community.utilities import SQLDatabase

from .config import Config

logger = logging.getLogger(__name__)


def connect(config: Config) -> SQLDatabase:
    """
    Connect to the database with error handling and a health check.

    Args:
        config: Application configuration with database connection details.

    Returns:
        An initialized SQLDatabase instance.

    Raises:
        ConnectionError: If the connection fails, with troubleshooting tips.
    """
    try:
        logger.info(
            "Connecting to %s/%s as %s...",
            config.db_server, config.db_name, config.db_user,
        )
        db = SQLDatabase.from_uri(config.connection_string)

        # Verify connection is actually alive
        db.run("SELECT 1")
        logger.info("Database connection successful.")
        return db

    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to database '{config.db_name}' "
            f"at '{config.db_server}': {e}\n\n"
            f"Troubleshooting:\n"
            f"  1. Is SQL Server running?\n"
            f"  2. Is SQL Server Authentication enabled?\n"
            f"  3. Are DB_USER and DB_PASSWORD correct in .env?\n"
            f"  4. Is '{config.db_driver}' installed?\n"
            f"  5. Can you connect using SSMS or sqlcmd?"
        ) from e


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
