"""
SQL query validation and safety guardrails.

Uses sqlglot Abstract Syntax Tree (AST) inspection to enforce read-only
data access across multiple dialects (T-SQL, PostgreSQL, MySQL, SQLite, etc.)
and maintains an audit log of all query execution attempts.
"""

import re
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

# pyrefly: ignore [missing-import]
import sqlglot
# pyrefly: ignore [missing-import]
from sqlglot import exp, errors

logger = logging.getLogger(__name__)

# Disallowed AST node classes anywhere in the parsed query tree
FORBIDDEN_AST_NODES = (
    exp.Drop,
    exp.Delete,
    exp.Insert,
    exp.Update,
    exp.Alter,
    exp.Create,
    exp.TruncateTable,
    exp.Grant,
    exp.Revoke,
    exp.Command,
    exp.Execute,
    exp.Merge,
    exp.Kill,
)

# Permitted top-level statement node types
ALLOWED_TOP_LEVEL_NODES = (
    exp.Selectable,  # Covers exp.Select and exp.Union
    exp.Describe,
    exp.Show,
    exp.Pragma,
)

# Dangerous procedural commands or system functions blocked regardless of dialect
FORBIDDEN_PROCEDURES: list[tuple[str, str]] = [
    ("xp_cmdshell", "xp_cmdshell (dangerous system procedure)"),
    ("sp_configure", "sp_configure (server configuration)"),
    ("sp_executesql", "sp_executesql (dynamic SQL execution)"),
    ("bulk insert", "BULK INSERT"),
    ("openrowset", "OPENROWSET (external data access)"),
    ("opendatasource", "OPENDATASOURCE (external data access)"),
    ("shutdown", "SHUTDOWN command"),
]

# Legacy regex fallback patterns in case of unexpected dialect syntax anomalies
FALLBACK_BLOCKED_PATTERNS: list[tuple[str, str]] = [
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
]


def normalize_dialect(dialect: Optional[str] = None) -> str:
    """
    Map dialect names to canonical sqlglot dialect identifiers.
    Normalizes PostgreSQL, MySQL, T-SQL, SQLite, and cloud DB variants.
    """
    if not dialect:
        return "tsql"
    d = dialect.lower().strip()
    if any(k in d for k in ["postgres", "cockroach", "neon", "supabase"]):
        return "postgres"
    if any(k in d for k in ["mysql", "mariadb", "planetscale"]):
        return "mysql"
    if "sqlite" in d:
        return "sqlite"
    if "oracle" in d:
        return "oracle"
    if "snowflake" in d:
        return "snowflake"
    if "bigquery" in d:
        return "bigquery"
    return "tsql"


class QueryAuditLog:
    """
    Persistent audit logger for all query execution attempts.
    Records timestamps, queries, decisions (ALLOWED/BLOCKED), and security reasons.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "query_audit.log"

    def log_query(self, sql: str, status: str, reason: str = "") -> None:
        """
        Log a single query attempt with timestamp and security classification.
        Appends entry safely to the query audit log file.
        """
        timestamp = datetime.now().isoformat()

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
    """
    Retrieve or lazily initialize the singleton QueryAuditLog instance.
    """
    global _audit_log
    if _audit_log is None:
        _audit_log = QueryAuditLog()
    return _audit_log


def validate_sql(sql: str, dialect: Optional[str] = None) -> tuple[bool, str]:
    """
    Validate a SQL query against safety rules using sqlglot AST inspection.
    Returns (is_safe, reason) ensuring strict read-only compliance and no data modifications.
    """
    sql_stripped = sql.strip()

    # Allow empty or whitespace-only queries
    if not sql_stripped:
        return True, ""

    # 1. Check forbidden system procedures / administrative commands
    sql_lower = sql_stripped.lower()
    for kw, description in FORBIDDEN_PROCEDURES:
        if re.search(rf"\b{re.escape(kw)}\b", sql_lower):
            reason = f"Blocked: {description} detected in query"
            get_audit_log().log_query(sql_stripped, "BLOCKED", reason)
            logger.warning(f"Query blocked — {reason}")
            return False, reason

    # 2. Parse into Abstract Syntax Tree (AST) with sqlglot
    glot_dialect = normalize_dialect(dialect)
    try:
        parsed = sqlglot.parse(sql_stripped, read=glot_dialect)
        statements = [s for s in parsed if s is not None]

        if not statements:
            get_audit_log().log_query(sql_stripped, "ALLOWED")
            return True, ""

        for stmt in statements:
            # Check top-level statement type
            if not isinstance(stmt, ALLOWED_TOP_LEVEL_NODES):
                stmt_name = type(stmt).__name__.upper()
                reason = f"Blocked: Non-SELECT/read-only statement ({stmt_name}) is not permitted"
                get_audit_log().log_query(sql_stripped, "BLOCKED", reason)
                logger.warning(f"Query blocked — {reason}")
                return False, reason

            # Recursively walk the statement AST to check for forbidden operations
            for node in stmt.walk():
                if isinstance(node, FORBIDDEN_AST_NODES):
                    node_name = type(node).__name__.upper()
                    reason = f"Blocked: Forbidden {node_name} operation detected in query AST"
                    get_audit_log().log_query(sql_stripped, "BLOCKED", reason)
                    logger.warning(f"Query blocked — {reason}")
                    return False, reason

        # Query passed AST inspection
        get_audit_log().log_query(sql_stripped, "ALLOWED")
        return True, ""

    except (errors.ParseError, errors.SqlglotError) as e:
        logger.debug("sqlglot AST parse warning (%s), checking secondary regex fallback...", e)

        # Fallback to secondary regex scan for unparseable dialect extensions
        for pattern, description in FALLBACK_BLOCKED_PATTERNS:
            if re.search(pattern, sql_stripped, re.IGNORECASE):
                reason = f"Blocked: {description} detected in query"
                get_audit_log().log_query(sql_stripped, "BLOCKED", reason)
                logger.warning(f"Query blocked (fallback) — {reason}")
                return False, reason

        # If no destructive pattern is matched in fallback, allow
        get_audit_log().log_query(sql_stripped, "ALLOWED")
        return True, ""

    except Exception as e:
        logger.error("Unexpected error during SQL AST validation: %s", e)
        # Defense in depth: fallback regex validation
        for pattern, description in FALLBACK_BLOCKED_PATTERNS:
            if re.search(pattern, sql_stripped, re.IGNORECASE):
                reason = f"Blocked: {description} detected in query"
                get_audit_log().log_query(sql_stripped, "BLOCKED", reason)
                return False, reason

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

