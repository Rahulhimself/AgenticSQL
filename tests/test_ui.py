"""
Unit tests for the Phase 4a Web Dashboard UI (agenticsql.ui).
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import streamlit as st

from agenticsql.ui import (
    init_session_state,
    load_backend,
    handle_user_query,
)
from agenticsql.cli import parse_args


class TestUIState:
    """Test session state initialization and handling."""

    def test_init_session_state(self):
        """Verify session state defaults are populated correctly."""
        # Clear existing session state if any
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        init_session_state()

        assert "messages" in st.session_state
        assert st.session_state.messages == []
        assert "last_sql" in st.session_state
        assert st.session_state.last_sql is None
        assert "last_df" in st.session_state
        assert st.session_state.last_df is None
        assert "db_connected" in st.session_state
        assert st.session_state.db_connected is False


class TestUIBackendLoading:
    """Test backend loading with mock and missing credentials."""

    def test_load_backend_missing_google_api_key(self):
        """Should return error if GOOGLE_API_KEY is empty."""
        # Reset state
        st.session_state.agent = None
        st.session_state.db = None
        st.session_state.config = MagicMock()
        st.session_state.config.google_api_key = ""

        agent, db, err = load_backend()
        assert agent is None
        assert db is None
        assert err is not None
        assert "GOOGLE_API_KEY is missing" in err

    @patch("agenticsql.ui.connect")
    @patch("agenticsql.ui.create_llm")
    @patch("agenticsql.ui.AgenticSQLAgent")
    @patch("agenticsql.ui.get_schema_info")
    def test_load_backend_success(self, mock_schema, mock_agent_cls, mock_llm, mock_connect):
        """Should connect and cache agent and schema when config is valid."""
        st.session_state.agent = None
        st.session_state.db = None
        st.session_state.config = MagicMock()
        st.session_state.config.google_api_key = "dummy_key"

        mock_db_instance = MagicMock()
        mock_connect.return_value = mock_db_instance
        mock_agent_instance = MagicMock()
        mock_agent_cls.return_value = mock_agent_instance
        mock_schema.return_value = {"users": "CREATE TABLE users (id INT);"}

        agent, db, err = load_backend()
        assert err is None
        assert agent is mock_agent_instance
        assert db is mock_db_instance
        assert st.session_state.db_connected is True
        assert "users" in st.session_state.schema_info


class TestUIQueryHandling:
    """Test user query execution and session state mutation."""

    def test_handle_user_query_updates_session_state(self):
        """handle_user_query should invoke agent and store output and DataFrame."""
        init_session_state()

        mock_agent = MagicMock()
        mock_agent.chat.return_value = {
            "output": "Found 2 customers.",
            "sql": ["SELECT * FROM customers;"],
            "data": {
                "columns": ["id", "name"],
                "rows": [[1, "Alice"], [2, "Bob"]],
            },
        }

        handle_user_query("Show customers", mock_agent)

        assert len(st.session_state.messages) == 2
        # User message
        assert st.session_state.messages[0]["role"] == "user"
        assert st.session_state.messages[0]["content"] == "Show customers"
        # Assistant message
        assert st.session_state.messages[1]["role"] == "assistant"
        assert st.session_state.messages[1]["content"] == "Found 2 customers."
        assert st.session_state.messages[1]["sql"] == ["SELECT * FROM customers;"]

        # DataFrame state
        assert st.session_state.last_df is not None
        assert isinstance(st.session_state.last_df, pd.DataFrame)
        assert len(st.session_state.last_df) == 2
        assert list(st.session_state.last_df.columns) == ["id", "name"]
        assert st.session_state.last_sql == "SELECT * FROM customers;"


class TestCLIParsing:
    """Test CLI argument parsing for UI mode."""

    @patch("sys.argv", ["main.py", "--ui", "--port", "8501"])
    def test_cli_parses_ui_flag(self):
        """CLI should recognize --ui flag and optional port."""
        args = parse_args()
        assert args.ui is True
        assert args.port == 8501
