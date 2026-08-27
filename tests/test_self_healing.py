"""
Unit tests for Phase 4b: Autonomous Self-Healing Query Pipeline, Error Reflection,
Schema Pruning, and SQL Explanation Breakdown.
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from agenticsql.agent import AgenticSQLAgent
from agenticsql.config import Config


@pytest.fixture
def mock_sql_db():
    """Create a mock SQLDatabase instance."""
    db = MagicMock()
    db.dialect = "sqlite"
    db.get_usable_table_names.return_value = ["users", "orders", "products", "order_items", "customers", "logs"]
    db.get_table_info.return_value = (
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT);\n"
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total FLOAT, created_at TEXT);"
    )
    # Mock underlying SQLAlchemy engine
    engine = MagicMock()
    db._engine = engine
    return db


@pytest.fixture
def mock_llm():
    """Create a mock ChatGoogleGenerativeAI instance."""
    llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "SELECT COUNT(*) FROM users;"
    llm.invoke.return_value = mock_response
    return llm


class TestSelfHealingMethods:
    """Test individual Phase 4b self-healing, explanation, and pruning methods."""

    @patch("agenticsql.agent.create_sql_agent")
    def test_prune_schema_tables_filters_relevant(self, mock_create_agent, mock_llm, mock_sql_db):
        """Test that schema pruner selects matching tables for the query."""
        agent = AgenticSQLAgent(
            llm=mock_llm,
            db=mock_sql_db,
            enable_schema_pruning=True,
        )

        relevant = agent.prune_schema_tables("How many users registered this year?")
        assert "users" in relevant
        assert len(relevant) < len(mock_sql_db.get_usable_table_names())

    @patch("agenticsql.agent.create_sql_agent")
    def test_prune_schema_disabled(self, mock_create_agent, mock_llm, mock_sql_db):
        """Test that disabling schema pruning returns all available tables."""
        agent = AgenticSQLAgent(
            llm=mock_llm,
            db=mock_sql_db,
            enable_schema_pruning=False,
        )
        assert len(agent.prune_schema_tables("Show users")) == 6

    @patch("agenticsql.agent.create_sql_agent")
    def test_self_heal_sql_cleans_markdown_fences(self, mock_create_agent, mock_llm, mock_sql_db):
        """Test that self_heal_sql strips markdown code blocks from LLM output."""
        mock_llm.invoke.return_value.content = "```sql\nSELECT id, name FROM users;\n```"

        agent = AgenticSQLAgent(llm=mock_llm, db=mock_sql_db)
        healed = agent.self_heal_sql(
            failed_sql="SELECT user_id, user_name FROM users",
            error_msg="no such column: user_name",
            user_question="List all users with their names",
        )

        assert healed == "SELECT id, name FROM users;"

    @patch("agenticsql.agent.create_sql_agent")
    def test_explain_sql_generates_breakdown(self, mock_create_agent, mock_llm, mock_sql_db):
        """Test that explain_sql queries LLM for educational breakdown."""
        mock_llm.invoke.return_value.content = "- Target Tables: users\n- Aggregation: COUNT(*)"

        agent = AgenticSQLAgent(llm=mock_llm, db=mock_sql_db)
        explanation = agent.explain_sql("SELECT COUNT(*) FROM users")

        assert "Target Tables" in explanation
        assert mock_llm.invoke.called

    @patch("agenticsql.agent.create_sql_agent")
    def test_execute_sql_with_error_blocks_destructive(self, mock_create_agent, mock_llm, mock_sql_db):
        """Test that execute_sql_with_error enforces safety guardrails."""
        agent = AgenticSQLAgent(llm=mock_llm, db=mock_sql_db)

        df, err = agent.execute_sql_with_error("DROP TABLE users;")
        assert df is None
        assert "guardrails" in err.lower()


class TestSelfHealingChatLoop:
    """Test full chat execution loop with simulated self-healing repair."""

    @patch("agenticsql.agent.create_sql_agent")
    def test_chat_triggers_self_healing_on_execution_failure(self, mock_create_agent, mock_llm, mock_sql_db):
        """Test that when an initial SQL query fails, self-healing reflection generates a repaired query."""
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor

        # Mock initial agent step returning bad SQL
        action = MagicMock()
        action.tool = "sql_db_query"
        action.tool_input = "SELECT bad_col FROM users;"
        mock_executor.invoke.return_value = {
            "output": "Here are the users.",
            "intermediate_steps": [(action, "Error: no such column bad_col")],
        }

        agent = AgenticSQLAgent(
            llm=mock_llm,
            db=mock_sql_db,
            max_retries=3,
            enable_self_healing=True,
        )

        # Mock execute_sql_with_error: first call fails, second call (repaired) succeeds
        valid_df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        agent.execute_sql_with_error = MagicMock(side_effect=[
            (None, "no such column: bad_col"),  # Attempt 1 fails
            (valid_df, None),                  # Attempt 2 succeeds
        ])
        agent.self_heal_sql = MagicMock(return_value="SELECT id, name FROM users;")
        agent.explain_sql = MagicMock(return_value="Explaining fixed query.")

        res = agent.chat("Show all users")

        assert res["healed"] is True
        assert res["attempts"] == 2
        assert res["data"] is not None
        assert "SELECT id, name FROM users;" in res["sql"]
        assert res["explanation"] == "Explaining fixed query."
