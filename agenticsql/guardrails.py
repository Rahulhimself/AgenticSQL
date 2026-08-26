"""
SQL query validation and safety guardrails.

Blocks destructive or data-modifying SQL operations and maintains
an audit log of all query execution attempts.
"""

import re
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# SQL patterns that indicate destructive or data-modifying operations
BLOCKED_PATTERNS: list[tuple[str, str]] = [
    (r"\bDROP\s+(TABLE|DATABASE|SCHEMA|INDEX|VIEW|PROCEDURE|FUNCTION|TRIGGER)\b", "DROP statement"),
    (r"\bTRUNCATE\s+TABLE\b", "TRUNCATE TABLE"),
    (r"\bDELETE\s+FROM\b", "DELETE FROM"),
    (r"\bALTER\s+(TABLE|DATABASE|SCHEMA)\b", "ALTER statement"),
    (r"\bINSERT\s+INTO\b", "INSERT INTO"),
    (r"\bUPDATE\s+\w+\s+SET\b", "UPDATE statement"),
    (r"\bEXEC(UTE)?\s+", "EXEC/EXECUTE statement"),
    (r"\bCREATE\s+(TABLE|DATABASE|SCHEMA|INDEX|VIEW|PROCEDURE|FUNCTION|TRIGGER)\b", "CREATE statement"),
    (r"\bGRANT\s+", "GRANT statement"),
    (r"\bREVOKE\s+", "REVOKE statement"),
    (r"\bSHUTDOWN\b", "SHUTDOWN command"),
    (r"\bxp_cmdshell\b", "xp_cmdshell (dangerous system procedure)"),
    (r"\bsp_configure\b", "sp_configure (server configuration)"),
    (r"\bBULK\s+INSERT\b", "BULK INSERT"),
    (r"\bOPENROWSET\b", "OPENROWSET (external data access)"),
    (r"\bOPENDATASOURCE\b", "OPENDATASOURCE (external data access)"),
]


class QueryAuditLog:
    """Logs all SQL queries to a file for auditing and forensics."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "query_audit.log"

    def log_query(self, sql: str, status: str, reason: str = "") -> None:
        """
        Log a query execution attempt.

        Args:
            sql: The SQL query string.
            status: 'ALLOWED' or 'BLOCKED'.
            reason: Reason for blocking (if blocked).
        """
        timestamp = datetime.now().isoformat()
        # Normalize whitespace for cleaner logs
        clean_sql = " ".join(sql.strip().split())
        entry = f"[{timestamp}] [{status}] {clean_sql}"
        if reason:
            entry += f" | Reason: {reason}"
        entry += "\n"

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(entry)
        except OSError as e:
            logger.warning(f"Could not write to audit log: {e}")


# Module-level audit log instance (lazy init)
_audit_log: Optional[QueryAuditLog] = None


def get_audit_log() -> QueryAuditLog:
    """Get or create the global audit log."""
    global _audit_log
    if _audit_log is None:
        _audit_log = QueryAuditLog()
    return _audit_log


def validate_sql(sql: str) -> tuple[bool, str]:
    """
    Validate a SQL query against safety rules.

    Args:
        sql: The SQL query to validate.

    Returns:
        A tuple of (is_safe, reason).
        is_safe is True if the query is safe to execute.
        reason contains the blocking reason if not safe, or empty string.
    """
    sql_stripped = sql.strip()

    # Allow empty queries
    if not sql_stripped:
        return True, ""

    # Check against blocked patterns (case-insensitive)
    for pattern, description in BLOCKED_PATTERNS:
        if re.search(pattern, sql_stripped, re.IGNORECASE):
            reason = f"Blocked: {description} detected in query"
            get_audit_log().log_query(sql_stripped, "BLOCKED", reason)
            logger.warning(f"Query blocked — {reason}")
            return False, reason

    # Log allowed queries
    get_audit_log().log_query(sql_stripped, "ALLOWED")
    return True, ""


def format_blocked_message(reason: str) -> str:
    """Format a user-friendly message when a query is blocked."""
    return (
        f"[BLOCKED] Query Blocked: {reason}\n\n"
        "This agent is configured for READ-ONLY access. "
        "Destructive or data-modifying operations (DROP, DELETE, UPDATE, INSERT, ALTER, etc.) "
        "are not permitted.\n\n"
        "If you need to modify data, please use a database management tool directly."
    )
