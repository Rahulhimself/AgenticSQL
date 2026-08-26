"""
Tests for agenticsql.agent module.

Validates:
- Agent initialization with tool-calling and fallback
- Guardrails intercepting SQL execution on db.run()
- Robust SQL extraction from intermediate agent steps
- Conversation memory tracking and history trimming
- Schema retrieval
"""

import sqlite3
# pyrefly: ignore [missing-import]
import pytest
from unittest.mock import MagicMock, patch
# pyrefly: ignore [missing-import]
from langchain_community.utilities import SQLDatabase
from agenticsql.agent import AgenticSQLAgent, _apply_guardrails


@pytest.fixture
def mock_db():
    """Create an in-memory SQLite database fixture with pre-populated tables."""
    conn = sqlite3.connect("file:agent_test_db?mode=memory&cache=shared", uri=True)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS test_users (id INTEGER PRIMARY KEY, name TEXT, role TEXT);")
    cursor.execute("DELETE FROM test_users;")
    cursor.execute("INSERT INTO test_users (id, name, role) VALUES (1, 'Alice', 'Admin');")
    conn.commit()

    db = SQLDatabase.from_uri("sqlite:///file:agent_test_db?mode=memory&cache=shared&uri=true")
    yield db
    conn.close()


@pytest.fixture
def mock_llm():
    """Create a mock LLM instance."""
    llm = MagicMock()
    llm.model = "gemini-2.5-flash"
    llm.temperature = 0.0
    return llm


class TestApplyGuardrails:
    """Test monkey-patching guardrails onto SQLDatabase."""

    def test_blocks_destructive_query_via_run(self, mock_db):
        """db.run should block destructive queries and return blocked message."""
        guarded_db = _apply_guardrails(mock_db)
        result = guarded_db.run("DROP TABLE test_users")
        assert "[BLOCKED]" in result
        assert "DROP" in result

    def test_allows_safe_query_via_run(self, mock_db):
        """db.run should execute safe SELECT queries."""
        guarded_db = _apply_guardrails(mock_db)
        result = guarded_db.run("SELECT * FROM test_users")
        assert "Alice" in result


class TestAgenticSQLExtractSQL:
    """Test SQL query extraction from intermediate agent steps."""

    def test_extract_sql_from_string_tool_input(self):
        """Extract SQL from string tool_input."""
        action = MagicMock()
        action.tool = "sql_db_query"
        action.tool_input = "SELECT id, name FROM test_users"

        response = {
            "output": "Found 1 user.",
            "intermediate_steps": [(action, "[(1, 'Alice')]")],
        }

        extracted = AgenticSQLAgent._extract_sql(response)
        assert len(extracted) == 1
        assert extracted[0] == "SELECT id, name FROM test_users"

    def test_extract_sql_from_dict_tool_input(self):
        """Extract SQL from dict tool_input with 'query' key."""
        action = MagicMock()
        action.tool = "sql_db_query"
        action.tool_input = {"query": "SELECT COUNT(*) FROM test_users"}

        response = {
            "output": "There is 1 user.",
            "intermediate_steps": [(action, "[(1,)]")],
        }

        extracted = AgenticSQLAgent._extract_sql(response)
        assert len(extracted) == 1
        assert extracted[0] == "SELECT COUNT(*) FROM test_users"

    def test_extract_sql_strips_markdown_code_fences(self):
        """Extract SQL should clean markdown code fences if present."""
        action = MagicMock()
        action.tool = "sql_db_query"
        action.tool_input = "```sql\nSELECT * FROM test_users\n```"

        response = {
            "output": "Results",
            "intermediate_steps": [(action, "...")],
        }

        extracted = AgenticSQLAgent._extract_sql(response)
        assert len(extracted) == 1
        assert extracted[0] == "SELECT * FROM test_users"

    def test_extract_sql_ignores_non_sql_tools(self):
        """Extract SQL should ignore unrelated tools."""
        action = MagicMock()
        action.tool = "calculator"
        action.tool_input = "2 + 2"

        response = {
            "output": "4",
            "intermediate_steps": [(action, "4")],
        }

        extracted = AgenticSQLAgent._extract_sql(response)
        assert len(extracted) == 0


class TestAgenticSQLAgentLifecycle:
    """Test AgenticSQLAgent interaction, memory, and lifecycle."""

    @patch("agenticsql.agent.create_sql_agent")
    def test_agent_initialization(self, mock_create_agent, mock_llm, mock_db):
        """Agent should initialize with tool-calling by default."""
        agent = AgenticSQLAgent(llm=mock_llm, db=mock_db, memory_window=5)
        assert agent.agent_type == "tool-calling"
        assert agent.memory_window == 5
        assert len(agent.chat_history) == 0
        assert agent.last_sql is None
        mock_create_agent.assert_called_once()

    @patch("agenticsql.agent.create_sql_agent")
    def test_chat_updates_history_and_last_sql(self, mock_create_agent, mock_llm, mock_db):
        """chat() should invoke agent and record conversation turns."""
        mock_executor = MagicMock()
        action = MagicMock()
        action.tool = "sql_db_query"
        action.tool_input = "SELECT name FROM test_users"
        mock_executor.invoke.return_value = {
            "output": "The user is Alice.",
            "intermediate_steps": [(action, "[(Alice,)]")],
        }
        mock_create_agent.return_value = mock_executor

        agent = AgenticSQLAgent(llm=mock_llm, db=mock_db)
        result = agent.chat("Who is in the database?")

        assert result["output"] == "The user is Alice."
        assert result["sql"] == ["SELECT name FROM test_users"]
        assert agent.last_sql == "SELECT name FROM test_users"

        history = agent.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Who is in the database?"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "The user is Alice."

    @patch("agenticsql.agent.create_sql_agent")
    def test_memory_window_trimming(self, mock_create_agent, mock_llm, mock_db):
        """Memory window should trim older entries beyond the configured size."""
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {"output": "Response", "intermediate_steps": []}
        mock_create_agent.return_value = mock_executor

        agent = AgenticSQLAgent(llm=mock_llm, db=mock_db, memory_window=2)

        for i in range(5):
            agent.chat(f"Question {i}")

        history = agent.get_history()
        # memory_window=2 => max 4 entries (2 pairs)
        assert len(history) == 4
        assert history[-2]["content"] == "Question 4"

    @patch("agenticsql.agent.create_sql_agent")
    def test_clear_history(self, mock_create_agent, mock_llm, mock_db):
        """clear_history() should empty history and reset last_sql."""
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {"output": "Response", "intermediate_steps": []}
        mock_create_agent.return_value = mock_executor

        agent = AgenticSQLAgent(llm=mock_llm, db=mock_db)
        agent.chat("Test question")
        assert len(agent.get_history()) == 2

        agent.clear_history()
        assert len(agent.get_history()) == 0
        assert agent.last_sql is None

    def test_get_schema(self, mock_llm, mock_db):
        """get_schema() should return table definitions."""
        with patch("agenticsql.agent.create_sql_agent"):
            agent = AgenticSQLAgent(llm=mock_llm, db=mock_db)
            schema = agent.get_schema()
            assert "test_users" in schema
            assert "CREATE TABLE" in schema

    def test_execute_sql_returns_dataframe(self, mock_llm, mock_db):
        """execute_sql() should run valid queries and return a DataFrame."""
        with patch("agenticsql.agent.create_sql_agent"):
            agent = AgenticSQLAgent(llm=mock_llm, db=mock_db)
            df = agent.execute_sql("SELECT name, role FROM test_users")
            assert df is not None
            assert len(df) == 1
            assert df.iloc[0]["name"] == "Alice"

    def test_execute_sql_blocks_destructive_query(self, mock_llm, mock_db):
        """execute_sql() should return None for blocked queries."""
        with patch("agenticsql.agent.create_sql_agent"):
            agent = AgenticSQLAgent(llm=mock_llm, db=mock_db)
            df = agent.execute_sql("DROP TABLE test_users")
            assert df is None


class TestDialectPrompt:
    """Test dialect-aware system prompt generation."""

    def test_postgresql_prompt(self):
        """PostgreSQL prompt should specify LIMIT, double quotes, and ILIKE."""
        from agenticsql.agent import get_dialect_prompt
        prompt = get_dialect_prompt("postgresql")
        assert "POSTGRESQL" in prompt
        assert "LIMIT" in prompt
        assert "ILIKE" in prompt
        assert "TOP (N) instead of LIMIT" not in prompt

    def test_mysql_prompt(self):
        """MySQL prompt should specify backticks and LIMIT."""
        from agenticsql.agent import get_dialect_prompt
        prompt = get_dialect_prompt("mysql")
        assert "MYSQL" in prompt
        assert "LIMIT" in prompt
        assert "backticks" in prompt

    def test_mssql_prompt(self):
        """MSSQL prompt should specify TOP and square brackets."""
        from agenticsql.agent import get_dialect_prompt
        prompt = get_dialect_prompt("mssql")
        assert "T-SQL / MSSQL" in prompt
        assert "TOP" in prompt
        assert "square brackets" in prompt

    def test_sqlite_prompt(self):
        """SQLite prompt should specify LIMIT."""
        from agenticsql.agent import get_dialect_prompt
        prompt = get_dialect_prompt("sqlite")
        assert "SQLITE" in prompt
        assert "LIMIT" in prompt

