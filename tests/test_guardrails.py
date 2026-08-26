"""
Tests for agenticsql.guardrails module.

Validates that destructive SQL is blocked and safe SQL is allowed.
"""


# pyrefly: ignore [missing-import]
import pytest
from agenticsql.guardrails import validate_sql, format_blocked_message


class TestValidateSQL:
    """Test SQL validation against safety rules."""

    # --- Blocked queries ---

    @pytest.mark.parametrize("sql,description", [
        ("DROP TABLE users", "DROP TABLE"),
        ("drop table users", "DROP TABLE (lowercase)"),
        ("DROP DATABASE production", "DROP DATABASE"),
        ("DROP VIEW my_view", "DROP VIEW"),
        ("DROP PROCEDURE sp_test", "DROP PROCEDURE"),
        ("TRUNCATE TABLE orders", "TRUNCATE TABLE"),
        ("DELETE FROM customers", "DELETE FROM"),
        ("DELETE FROM customers WHERE 1=1", "DELETE FROM with WHERE"),
        ("ALTER TABLE users ADD COLUMN age INT", "ALTER TABLE"),
        ("ALTER DATABASE mydb SET SINGLE_USER", "ALTER DATABASE"),
        ("INSERT INTO users VALUES (1, 'test')", "INSERT INTO"),
        ("UPDATE users SET name='hacked'", "UPDATE SET"),
        ("EXEC sp_executesql N'SELECT 1'", "EXEC"),
        ("EXECUTE xp_cmdshell 'dir'", "EXECUTE"),
        ("CREATE TABLE new_table (id INT)", "CREATE TABLE"),
        ("CREATE DATABASE newdb", "CREATE DATABASE"),
        ("GRANT SELECT ON users TO hacker", "GRANT"),
        ("REVOKE ALL FROM public", "REVOKE"),
        ("SHUTDOWN", "SHUTDOWN"),
        ("xp_cmdshell 'whoami'", "xp_cmdshell"),
        ("sp_configure 'show advanced options', 1", "sp_configure"),
        ("BULK INSERT users FROM 'file.csv'", "BULK INSERT"),
        ("SELECT * FROM OPENROWSET('SQLNCLI', 'Server=evil')", "OPENROWSET"),
    ])
    def test_blocks_destructive_sql(self, sql, description):
        """Destructive SQL statements should be blocked."""
        is_safe, reason = validate_sql(sql)
        assert not is_safe, f"Expected '{description}' to be blocked: {sql}"
        assert reason, f"Blocked query should have a reason: {sql}"

    # --- Allowed queries ---

    @pytest.mark.parametrize("sql,description", [
        ("SELECT * FROM users", "Basic SELECT"),
        ("SELECT COUNT(*) FROM orders", "SELECT COUNT"),
        ("SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id",
         "SELECT with JOIN"),
        ("SELECT TOP 10 * FROM products ORDER BY price DESC", "SELECT TOP"),
        ("SELECT DISTINCT category FROM products", "SELECT DISTINCT"),
        ("""
            SELECT
                category,
                AVG(price) as avg_price,
                COUNT(*) as count
            FROM products
            GROUP BY category
            HAVING COUNT(*) > 5
            ORDER BY avg_price DESC
        """, "Complex aggregation query"),
        ("", "Empty query"),
        ("   ", "Whitespace-only query"),
    ])
    def test_allows_safe_sql(self, sql, description):
        """Safe SELECT queries should be allowed."""
        is_safe, reason = validate_sql(sql)
        assert is_safe, f"Expected '{description}' to be allowed: {sql} (reason: {reason})"

    # --- Case sensitivity ---

    def test_case_insensitive_blocking(self):
        """Blocking should work regardless of case."""
        variations = [
            "DROP TABLE users",
            "drop table users",
            "Drop Table Users",
            "dRoP tAbLe users",
        ]
        for sql in variations:
            is_safe, _ = validate_sql(sql)
            assert not is_safe, f"Should block case variation: {sql}"


class TestFormatBlockedMessage:
    """Test blocked message formatting."""

    def test_includes_reason(self):
        """Blocked message should include the reason."""
        msg = format_blocked_message("DROP statement detected")
        assert "DROP statement detected" in msg

    def test_includes_read_only_notice(self):
        """Blocked message should mention read-only access."""
        msg = format_blocked_message("test reason")
        assert "READ-ONLY" in msg

    def test_includes_blocked_marker(self):
        """Blocked message should include the blocked marker."""
        msg = format_blocked_message("test")
        assert "[BLOCKED]" in msg
