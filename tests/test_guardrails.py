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


class TestASTGuardrailAdvancements:
    """Tests specifically validating AST capabilities over naive regex."""

    def test_allows_select_with_dangerous_words_in_string_literals(self):
        """Keywords inside string literals must not be falsely blocked."""
        queries = [
            "SELECT * FROM articles WHERE title = 'How to DROP a table in SQL'",
            "SELECT * FROM logs WHERE message LIKE '%DELETE FROM orders%'",
            "SELECT * FROM support_tickets WHERE description = 'Need to UPDATE my address'",
            "SELECT 'ALTER TABLE accounts ADD COLUMN test' AS command_text FROM dual",
        ]
        for sql in queries:
            is_safe, reason = validate_sql(sql)
            assert is_safe, f"Safe query was falsely blocked: {sql} (reason: {reason})"

    def test_blocks_multi_statement_injection(self):
        """Multi-statement queries containing mutations must be blocked."""
        injections = [
            "SELECT 1; DROP TABLE users;",
            "SELECT id FROM users; DELETE FROM orders WHERE 1=1;",
            "SELECT name FROM products; UPDATE users SET role = 'admin';",
            "SELECT count(*) FROM items; INSERT INTO logs VALUES ('hacked');",
        ]
        for sql in injections:
            is_safe, reason = validate_sql(sql)
            assert not is_safe, f"Multi-statement injection was not blocked: {sql}"
            assert reason, f"Should provide blocking reason for: {sql}"

    def test_allows_safe_multi_select(self):
        """Multi-statement queries with only safe SELECTs should be allowed."""
        is_safe, _ = validate_sql("SELECT * FROM users; SELECT * FROM orders;")
        assert is_safe

    def test_blocks_cte_concealed_mutations(self):
        """Mutations concealed inside or after CTE expressions must be blocked."""
        cte_mutations = [
            "WITH temp_ids AS (SELECT id FROM users) DELETE FROM users WHERE id IN (SELECT id FROM temp_ids)",
            "WITH temp_data AS (SELECT 1 AS x) UPDATE accounts SET balance = 0",
            "WITH cte AS (SELECT * FROM old_table) INSERT INTO new_table SELECT * FROM cte",
        ]
        for sql in cte_mutations:
            is_safe, reason = validate_sql(sql)
            assert not is_safe, f"CTE mutation was not blocked: {sql}"

    def test_allows_safe_cte_queries(self):
        """Read-only CTE queries should be allowed."""
        safe_ctes = [
            "WITH ranked AS (SELECT id, ROW_NUMBER() OVER (ORDER BY created_at) as rn FROM orders) SELECT * FROM ranked WHERE rn <= 10",
            "WITH sales_summary AS (SELECT product_id, SUM(amount) as total FROM sales GROUP BY product_id) SELECT * FROM sales_summary ORDER BY total DESC",
        ]
        for sql in safe_ctes:
            is_safe, reason = validate_sql(sql)
            assert is_safe, f"Safe CTE query was blocked: {sql} (reason: {reason})"

    def test_allows_queries_with_sql_comments(self):
        """Safe queries with comments should be permitted."""
        comment_queries = [
            "/* Query written by analyst to check dropouts */ SELECT * FROM students",
            "-- Filter by active users\nSELECT name FROM customers WHERE status = 'active'",
            "SELECT id, name /* internal id */ FROM products",
        ]
        for sql in comment_queries:
            is_safe, reason = validate_sql(sql)
            assert is_safe, f"Safe commented query was blocked: {sql} (reason: {reason})"

    def test_blocks_ddl_with_comments(self):
        """Destructive DDL disguised with comments must be blocked."""
        ddl_comments = [
            "/* safe looking comment */ DROP TABLE audit_log;",
            "-- admin maintenance\nTRUNCATE TABLE users;",
        ]
        for sql in ddl_comments:
            is_safe, reason = validate_sql(sql)
            assert not is_safe, f"Comment-disguised DDL was not blocked: {sql}"


class TestMultiDialectASTValidation:
    """Tests validating AST checks across discrete SQL dialects."""

    def test_postgresql_dialect(self):
        """PostgreSQL dialect specific queries."""
        assert validate_sql('SELECT * FROM "public"."users" LIMIT 10 OFFSET 5', dialect="postgresql")[0]
        assert not validate_sql('DROP TABLE "public"."users"', dialect="postgresql")[0]

    def test_mysql_dialect(self):
        """MySQL dialect specific queries."""
        assert validate_sql("SELECT * FROM `orders` LIMIT 10", dialect="mysql")[0]
        assert validate_sql("SHOW TABLES", dialect="mysql")[0]
        assert validate_sql("DESCRIBE users", dialect="mysql")[0]
        assert not validate_sql("DROP TABLE `orders`", dialect="mysql")[0]

    def test_sqlite_dialect(self):
        """SQLite dialect specific queries."""
        assert validate_sql("SELECT * FROM items LIMIT 5", dialect="sqlite")[0]
        assert validate_sql("PRAGMA table_info(users)", dialect="sqlite")[0]
        assert not validate_sql("DROP TABLE items", dialect="sqlite")[0]

    def test_tsql_dialect(self):
        """T-SQL / MSSQL dialect specific queries."""
        assert validate_sql("SELECT TOP 10 * FROM [dbo].[customers]", dialect="mssql")[0]
        assert not validate_sql("DROP TABLE [dbo].[customers]", dialect="mssql")[0]
