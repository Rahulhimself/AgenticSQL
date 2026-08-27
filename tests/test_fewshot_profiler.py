"""
Unit tests for Phase 4c: Semantic Few-Shot RAG Exemplar Store,
AST Query Performance Profiler, and Dynamic Exemplar Learning.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from agenticsql.fewshot import FewShotStore, FewShotExample
from agenticsql.profiler import QueryProfiler, ProfileReport
from agenticsql.agent import AgenticSQLAgent


class TestFewShotStore:
    """Test Few-Shot Store loading, retrieval ranking, and exemplar learning."""

    def test_seed_exemplars_loaded(self, tmp_path):
        """Test that default seed exemplars are initialized if storage file is new."""
        temp_file = tmp_path / "fewshot_test.json"
        store = FewShotStore(storage_path=temp_file)

        assert len(store.examples) > 0
        assert any(e.dialect == "mssql" for e in store.examples)
        assert any(e.dialect == "postgresql" for e in store.examples)

    def test_retrieve_relevant_dialect_filtering(self, tmp_path):
        """Test that retrieval filters and prioritizes target dialect."""
        temp_file = tmp_path / "fewshot_test.json"
        store = FewShotStore(storage_path=temp_file)

        # Retrieve for PostgreSQL
        pg_results = store.retrieve_relevant("Show customers with highest orders", dialect="postgresql", top_k=2)
        assert len(pg_results) <= 2
        assert all(e.dialect == "postgresql" for e in pg_results)

        # Retrieve for MSSQL
        ms_results = store.retrieve_relevant("Show transactions from last 30 days", dialect="mssql", top_k=2)
        assert len(ms_results) <= 2
        assert all(e.dialect == "mssql" for e in ms_results)

    def test_add_example_persistence(self, tmp_path):
        """Test adding and persisting a new golden exemplar."""
        temp_file = tmp_path / "fewshot_test.json"
        store = FewShotStore(storage_path=temp_file)
        initial_count = len(store.examples)

        store.add_example(
            question="Calculate 90-day moving average volume",
            sql="SELECT AVG(Volume) OVER (ORDER BY Date ROWS 90 PRECEDING) FROM StockHistory;",
            dialect="mssql",
            category="window_function",
        )

        assert len(store.examples) == initial_count + 1

        # Reload from disk
        reloaded_store = FewShotStore(storage_path=temp_file)
        assert len(reloaded_store.examples) == initial_count + 1
        assert any("moving average" in e.question for e in reloaded_store.examples)

    def test_format_examples_for_prompt(self, tmp_path):
        """Test formatting retrieved exemplars as a prompt block."""
        temp_file = tmp_path / "fewshot_test.json"
        store = FewShotStore(storage_path=temp_file)

        prompt_block = store.format_examples_for_prompt("Find top customers", dialect="mssql", top_k=1)
        assert "RELEVANT GOLDEN QUERY EXAMPLES" in prompt_block
        assert "```sql" in prompt_block


class TestQueryProfiler:
    """Test SQLGlot AST query performance profiler and anti-pattern detection."""

    def test_profiles_bounded_optimized_query_as_low_cost(self):
        """Test that a bounded query with explicit columns is rated LOW cost."""
        profiler = QueryProfiler(dialect="mssql")
        report = profiler.profile("SELECT TOP (10) user_id, email FROM users WHERE is_active = 1;")

        assert report.cost_rating == "LOW"
        assert report.complexity_score <= 2

    def test_detects_unbounded_select_star(self):
        """Test detection of unbounded SELECT * queries."""
        profiler = QueryProfiler(dialect="mssql")
        report = profiler.profile("SELECT * FROM large_orders_table WHERE status = 'PENDING';")

        assert report.cost_rating in ("MEDIUM", "HIGH")
        assert any("Unbounded Query" in w for w in report.warnings)
        assert any("SELECT *" in w for w in report.warnings)

    def test_detects_leading_wildcard_like(self):
        """Test detection of leading wildcard LIKE '%...' patterns."""
        profiler = QueryProfiler(dialect="postgresql")
        report = profiler.profile("SELECT id, name FROM users WHERE email LIKE '%@gmail.com' LIMIT 10;")

        assert any("Leading Wildcard" in w for w in report.warnings)
        assert any("full table scan" in w for w in report.warnings)

    def test_detects_cartesian_join(self):
        """Test detection of unconstrained cross joins."""
        profiler = QueryProfiler(dialect="mssql")
        report = profiler.profile("SELECT u.name, o.id FROM users u CROSS JOIN orders o;")

        assert report.cost_rating in ("MEDIUM", "HIGH")
        assert any("Cartesian" in w for w in report.warnings)


class TestAgentPhase4cIntegration:
    """Test FewShotStore and QueryProfiler integration into AgenticSQLAgent."""

    @patch("agenticsql.agent.create_sql_agent")
    def test_agent_chat_includes_cost_and_profiling(self, mock_create_agent):
        """Test that AgenticSQLAgent.chat() returns cost, profiling_tips, and few-shot context."""
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor

        action = MagicMock()
        action.tool = "sql_db_query"
        action.tool_input = "SELECT TOP (5) id, name FROM users;"
        mock_executor.invoke.return_value = {
            "output": "Here are top 5 users.",
            "intermediate_steps": [(action, "success")],
        }

        mock_db = MagicMock()
        mock_db.dialect = "mssql"
        mock_db.get_usable_table_names.return_value = ["users"]
        mock_db.get_table_info.return_value = "CREATE TABLE users (id INT, name VARCHAR(50));"
        mock_db._engine = MagicMock()

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Query explanation."

        agent = AgenticSQLAgent(llm=mock_llm, db=mock_db)
        agent.execute_sql_with_error = MagicMock(return_value=(
            pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}),
            None,
        ))

        res = agent.chat("Show top 5 users")

        assert res["cost"] in ("LOW", "MEDIUM", "HIGH")
        assert isinstance(res["profiling_tips"], list)
        assert isinstance(res["profiling_warnings"], list)
        assert mock_executor.invoke.called

    @patch("agenticsql.agent.create_sql_agent")
    def test_agent_add_golden_example(self, mock_create_agent):
        """Test saving a golden exemplar via agent method."""
        mock_db = MagicMock()
        mock_db.dialect = "mssql"
        mock_llm = MagicMock()

        agent = AgenticSQLAgent(llm=mock_llm, db=mock_db)
        initial_len = len(agent.fewshot_store.examples)

        unique_q = "Unique Top Products Query 999"
        agent.add_golden_example(unique_q, "SELECT TOP (10) * FROM products;")
        assert any(e.question == unique_q for e in agent.fewshot_store.examples)
