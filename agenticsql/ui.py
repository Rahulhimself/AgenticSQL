"""
AgenticSQL — Interactive Web Dashboard & AI Studio (Phase 4c: Auth & Multi-Tenancy).

Features:
1. Multi-User Authentication & JWT Engine (Login / Register / Role-based access).
2. Multi-Tenant Database Connection Registry & Dynamic Switcher.
3. Conversational Chat & Natural Language to SQL Studio.
4. Collapsible SQL query inspection with syntax highlighting & query explanation breakdown.
5. Interactive Data Table with instant CSV/JSON exports.
6. Auto-Visualization Studio with dynamic Plotly charts (Bar, Line, Area, Pie, Scatter).
7. Autonomous Self-Healing Query Execution (automated error reflection & retry).
8. AST-based Query Performance Profiler & Index Optimizer.
9. Semantic Few-Shot RAG Exemplar Store with 1-click bookmarking.
10. Live Database Schema Explorer (tables, column types, row counts, preview).
11. User-isolated Query Execution History with latency tracking.
12. Safety Guardrails & Real-Time Audit Log Monitor.
13. Executive Admin Analytics & Usage KPI Dashboard.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agenticsql.config import Config
from agenticsql.database import connect, get_schema_info
from agenticsql.llm import create_llm
from agenticsql.agent import AgenticSQLAgent
from agenticsql.auth import AuthDatabase, User, UserConnection
from agenticsql.tenancy import TenantManager
from agenticsql.visualization import (
    is_chartable,
    dataframe_to_csv,
    dataframe_to_json,
    dataframe_to_markdown,
)
from agenticsql import guardrails

logger = logging.getLogger(__name__)

# --- Custom Styling & Theme (Google Gemini UI Aesthetic) ---
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Base Theme (Gemini Deep Black/Grey Aesthetic) */
    html, body, [class*="css"], .stApp {
        font-family: 'Google Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: #131314 !important;
        color: #e3e3e3 !important;
    }
    
    /* Code blocks */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #1e1f20 !important;
        color: #e3e3e3 !important;
        border: 1px solid #2d2f31 !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Sidebar Gemini Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e1f20 !important;
        border-right: 1px solid #2d2f31 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #e3e3e3 !important;
    }

    /* Gemini Header Banner */
    .agent-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.1rem 1.6rem;
        background: linear-gradient(135deg, rgba(30, 31, 32, 0.95) 0%, rgba(19, 19, 20, 0.98) 100%);
        border: 1px solid #2d2f31;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .agent-title {
        font-size: 1.65rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a8c7fa 0%, #d3e3fd 50%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .agent-subtitle {
        color: #c4c7c5;
        font-size: 0.85rem;
        margin-top: 3px;
    }

    /* Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        background: rgba(30, 31, 32, 0.9);
        border: 1px solid #3c4043;
        border-radius: 9999px;
        color: #a8c7fa;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #34a853;
        border-radius: 50%;
        box-shadow: 0 0 8px #34a853;
    }

    /* Healed Badge */
    .healed-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        background: rgba(168, 199, 250, 0.12);
        border: 1px solid rgba(168, 199, 250, 0.3);
        border-radius: 8px;
        color: #a8c7fa;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Metric Cards */
    .metric-card {
        background: #1e1f20 !important;
        border: 1px solid #2d2f31 !important;
        border-radius: 14px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff !important;
    }
    .metric-lbl {
        font-size: 0.75rem;
        color: #8e918f !important;
        text-transform: uppercase;
        margin-top: 4px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Gemini Chat Messages */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0.75rem 0.5rem;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background-color: #1e1f20 !important;
        border-radius: 18px !important;
        padding: 0.85rem 1.2rem !important;
        border: 1px solid #2d2f31 !important;
        margin-bottom: 0.75rem;
    }

    /* Streamlit Chat Input (Gemini Rounded Floating Bar) */
    div[data-testid="stChatInput"] {
        background-color: #1e1f20 !important;
        border: 1px solid #3c4043 !important;
        border-radius: 28px !important;
        color: #ffffff !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #ffffff !important;
    }

    /* Gemini Pill Buttons */
    div.stButton > button {
        border-radius: 9999px !important;
        background-color: #1e1f20 !important;
        border: 1px solid #3c4043 !important;
        color: #e3e3e3 !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background-color: #282a2c !important;
        border-color: #a8c7fa !important;
        color: #ffffff !important;
        box-shadow: 0 0 12px rgba(168, 199, 250, 0.2) !important;
    }

    /* Prompt Suggestion Chips */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 9999px !important;
        font-size: 0.8rem !important;
        border: 1px solid #2d2f31 !important;
        background: #1e1f20 !important;
        color: #c4c7c5 !important;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        border-color: #a8c7fa !important;
        color: #a8c7fa !important;
        background: #282a2c !important;
    }

    /* Form Inputs */
    input, select, textarea, div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: #1e1f20 !important;
        border-color: #3c4043 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
    }

    /* Expanders & Tabs */
    div[data-testid="stExpander"] {
        background-color: #1e1f20 !important;
        border: 1px solid #2d2f31 !important;
        border-radius: 12px !important;
    }
    div[data-baseweb="tab-list"] {
        background-color: #131314 !important;
        border-bottom: 1px solid #2d2f31 !important;
    }
    button[data-baseweb="tab"] {
        color: #8e918f !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #a8c7fa !important;
        border-bottom-color: #a8c7fa !important;
    }
</style>
"""


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_sql" not in st.session_state:
        st.session_state.last_sql = None
    if "last_df" not in st.session_state:
        st.session_state.last_df = None
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "db" not in st.session_state:
        st.session_state.db = None
    if "config" not in st.session_state:
        try:
            st.session_state.config = Config.from_env()
        except Exception:
            st.session_state.config = None
    if "schema_info" not in st.session_state:
        st.session_state.schema_info = None
    if "db_connected" not in st.session_state:
        st.session_state.db_connected = False

    # Phase 4c: Auth & Multi-Tenancy State
    if "auth_db" not in st.session_state:
        st.session_state.auth_db = AuthDatabase()
    if "tenant_mgr" not in st.session_state:
        st.session_state.tenant_mgr = TenantManager(st.session_state.auth_db)
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "active_connection_id" not in st.session_state:
        st.session_state.active_connection_id = None


def render_login_screen() -> None:
    """Render authentication modal when user is not logged in."""
    st.markdown(
        """
        <div style="text-align: center; margin-top: 2rem; margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(90deg, #a8c7fa 0%, #d3e3fd 50%, #ffffff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">✨ AgenticSQL Studio</h1>
            <p style="color: #c4c7c5;">Autonomous Database Intelligence • Multi-Tenant Security</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_left, c_mid, c_right = st.columns([1, 2, 1])

    with c_mid:
        auth_tab_login, auth_tab_reg = st.tabs(["🔑 Sign In", "📝 Create Account"])

        with auth_tab_login:
            st.markdown("#### Welcome Back")
            st.caption("Sign in with your registered account credentials.")

            with st.form("login_form"):
                username_input = st.text_input("Username or Email", placeholder="e.g. admin or user@domain.com")
                password_input = st.text_input("Password", type="password", placeholder="••••••••")
                submit_login = st.form_submit_button("Sign In", use_container_width=True)

                if submit_login:
                    if not username_input or not password_input:
                        st.error("Please provide both username and password.")
                    else:
                        user = st.session_state.auth_db.authenticate_user(username_input, password_input)
                        if user:
                            st.session_state.current_user = user
                            st.toast(f"Welcome back, {user.username}!", icon="👋")
                            st.rerun()
                        else:
                            st.error("Invalid username/email or password.")

            st.info("💡 **Default Demo Account:** `admin` / `Admin@12345`")

        with auth_tab_reg:
            st.markdown("#### New Account Registration")
            st.caption("Register an isolated tenant workspace.")

            with st.form("reg_form"):
                reg_user = st.text_input("Username", placeholder="e.g. john_doe")
                reg_email = st.text_input("Email", placeholder="e.g. john@example.com")
                reg_pass = st.text_input("Password", type="password", placeholder="••••••••")
                reg_role = st.selectbox("Role", options=["user", "admin"], index=0)
                submit_reg = st.form_submit_button("Register Account", use_container_width=True)

                if submit_reg:
                    if not reg_user or not reg_email or not reg_pass:
                        st.error("All fields are required.")
                    else:
                        try:
                            new_user = st.session_state.auth_db.register_user(
                                username=reg_user,
                                email=reg_email,
                                password=reg_pass,
                                role=reg_role,
                            )
                            st.session_state.current_user = new_user
                            st.success(f"Account created successfully for {new_user.username}!")
                            st.rerun()
                        except ValueError as e:
                            st.error(str(e))


def load_backend_for_tenant() -> tuple[Optional[AgenticSQLAgent], Optional[Any], Optional[str]]:
    """Load or switch the active database and agent for the logged-in tenant."""
    if st.session_state.agent is not None and st.session_state.db is not None:
        return st.session_state.agent, st.session_state.db, None

    user = st.session_state.current_user
    if not user:
        return None, None, "User not authenticated."

    conn_id = st.session_state.active_connection_id

    try:
        config = st.session_state.config or Config.from_env()
        st.session_state.config = config

        provider = (config.llm_provider or "groq").lower().strip()
        if provider == "groq" and not config.groq_api_key:
            return None, None, "GROQ_API_KEY is missing from environment. Please add it to your .env file."
        elif provider == "gemini" and not config.google_api_key:
            return None, None, "GOOGLE_API_KEY is missing from environment. Please add it to your .env file."
        elif provider == "openai" and not config.openai_api_key:
            return None, None, "OPENAI_API_KEY is missing from environment. Please add it to your .env file."

        if conn_id is None:
            db = connect(config)
            llm = create_llm(config)
            agent = AgenticSQLAgent(
                llm=llm,
                db=db,
                verbose=False,
                max_retries=config.max_retries,
                enable_self_healing=config.enable_self_healing,
                enable_schema_pruning=config.enable_schema_pruning,
            )
        else:
            agent = st.session_state.tenant_mgr.get_agent(user, connection_id=conn_id, config=config)
            db = st.session_state.tenant_mgr.connect_tenant_db(user, connection_id=conn_id, config=config)

        st.session_state.agent = agent
        st.session_state.db = db
        st.session_state.db_connected = True
        st.session_state.schema_info = get_schema_info(db)

        return agent, db, None

    except Exception as e:
        logger.error("Failed to load tenant backend: %s", e)
        st.session_state.db_connected = False
        return None, None, str(e)


def load_backend() -> tuple[Optional[AgenticSQLAgent], Optional[Any], Optional[str]]:
    """Default backend loader fallback."""
    if st.session_state.get("current_user") is None:
        st.session_state.current_user = User(id=1, username="admin", email="admin@agenticsql.ai", role="admin")
    return load_backend_for_tenant()


def render_header(config: Optional[Config], is_connected: bool) -> None:
    """Render top header bar with branding and live status."""
    user = st.session_state.current_user
    dialect_str = (getattr(st.session_state.db, "dialect", config.db_type) if config else "SQL").upper()
    provider_str = (config.llm_provider if config else "Groq").upper()
    model_str = config.llm_model if config else "openai/gpt-oss-120b"

    status_html = (
        f'<div class="status-pill"><span class="status-dot"></span>{dialect_str} ONLINE</div>'
        if is_connected
        else '<div class="status-pill offline"><span class="status-dot"></span>OFFLINE</div>'
    )

    role_badge = (
        f'<span style="padding: 4px 12px; font-size: 0.75rem; font-weight: 600; border-radius: 9999px; background: rgba(168, 199, 250, 0.12); color: #a8c7fa; border: 1px solid rgba(168, 199, 250, 0.3); letter-spacing: 0.03em;">👤 {user.username} ({user.role.upper()})</span>'
        if user
        else ""
    )

    provider_badge = f'<span style="padding: 4px 10px; font-size: 0.75rem; font-weight: 600; border-radius: 9999px; background: rgba(74, 222, 128, 0.12); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.3);">{provider_str}</span>'

    st.markdown(
        f"""
        <div class="agent-header">
            <div>
                <h1 class="agent-title">✨ AgenticSQL Studio</h1>
                <div class="agent-subtitle">Autonomous Database Intelligence • Powered by {model_str}</div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                {provider_badge}
                {role_badge}
                {status_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(config: Optional[Config], db: Optional[Any]) -> None:
    """Render sidebar with user profile, database connection switcher, and connection manager."""
    user = st.session_state.current_user

    with st.sidebar:
        st.markdown(f"### 👤 Tenant Profile")
        if user:
            st.markdown(f"**User:** `{user.username}`")
            st.markdown(f"**Email:** `{user.email}`")
            st.markdown(f"**Role:** `{user.role.upper()}`")

            if st.button("🚪 Sign Out", use_container_width=True):
                st.session_state.current_user = None
                st.session_state.messages = []
                st.session_state.active_connection_id = None
                st.rerun()

        st.markdown("---")
        st.markdown("### 🗄️ Database Switcher")

        # Fetch registered connections for user
        user_conns = st.session_state.auth_db.get_user_connections(user.id) if user else []

        db_options = {"default": f"Default Database ({config.db_name or config.db_type.upper()})"}
        for c in user_conns:
            db_options[c.id] = f"{c.name} ({c.db_type.upper()})"

        current_choice = st.session_state.active_connection_id if st.session_state.active_connection_id in db_options else "default"
        selected_choice = st.selectbox(
            "Active Database",
            options=list(db_options.keys()),
            format_func=lambda x: db_options[x],
            index=list(db_options.keys()).index(current_choice),
        )

        target_conn_id = None if selected_choice == "default" else selected_choice
        if target_conn_id != st.session_state.active_connection_id:
            st.session_state.active_connection_id = target_conn_id
            st.session_state.agent = None
            st.session_state.db = None
            st.rerun()

        # Quick Delete Active Custom Database
        if selected_choice != "default" and user:
            if st.button("🗑️ Delete Selected Database", use_container_width=True, help="Remove this custom database connection"):
                st.session_state.tenant_mgr.delete_connection(user.id, selected_choice)
                st.session_state.active_connection_id = None
                st.session_state.agent = None
                st.session_state.db = None
                st.success("Database connection deleted.")
                st.rerun()

        # Register & Manage Custom Databases
        with st.expander("⚙️ Manage & Register Databases", expanded=False):
            tab_reg, tab_manage = st.tabs(["➕ Add New", "📋 Saved Connections"])

            with tab_reg:
                mode = st.radio("Configuration Mode", ["Quick Form (Recommended)", "Direct URI"], horizontal=True)

                if mode == "Quick Form (Recommended)":
                    with st.form("quick_db_conn_form"):
                        conn_name = st.text_input("Connection Label", placeholder="e.g. Rahul June 2026 DB")
                        conn_type = st.selectbox("Dialect", options=["mssql", "postgresql", "mysql", "sqlite"])
                        conn_dbname = st.text_input("Database Name", placeholder="e.g. RAHUL_B_PRT_JUNE_2026 or my_db")

                        if conn_type == "sqlite":
                            conn_server = ""
                            conn_user = ""
                            conn_pass = ""
                            conn_driver = ""
                        else:
                            conn_server = st.text_input("Server / Host", value=config.db_server if config else "127.0.0.1")
                            conn_user = st.text_input("Username", value=config.db_user if config else "")
                            conn_pass = st.text_input("Password", value=config.db_password if config else "", type="password")
                            conn_driver = st.text_input("Driver (for MSSQL)", value=config.db_driver if config else "ODBC+Driver+17+for+SQL+Server") if conn_type == "mssql" else ""

                        submit_btn = st.form_submit_button("Test & Save Connection")

                        if submit_btn:
                            if not conn_name or not conn_dbname:
                                st.error("Please provide both Connection Label and Database Name.")
                            else:
                                import urllib.parse
                                from sqlalchemy import create_engine

                                enc_user = urllib.parse.quote_plus(conn_user)
                                enc_pass = urllib.parse.quote_plus(conn_pass)
                                enc_db = urllib.parse.quote_plus(conn_dbname)

                                if conn_type == "sqlite":
                                    uri = f"sqlite:///{conn_dbname}"
                                elif conn_type == "postgresql":
                                    uri = f"postgresql+psycopg2://{enc_user}:{enc_pass}@{conn_server}:5432/{enc_db}"
                                elif conn_type == "mysql":
                                    uri = f"mysql+pymysql://{enc_user}:{enc_pass}@{conn_server}:3306/{enc_db}"
                                else:  # mssql
                                    uri = f"mssql+pyodbc://{enc_user}:{enc_pass}@{conn_server}/{enc_db}?driver={conn_driver}&TrustServerCertificate=yes"

                                try:
                                    # Pre-flight connection test
                                    test_engine = create_engine(uri)
                                    with test_engine.connect() as test_conn:
                                        pass

                                    new_c = st.session_state.tenant_mgr.register_connection(
                                        user_id=user.id,
                                        name=conn_name,
                                        db_type=conn_type,
                                        db_uri=uri,
                                        db_server=conn_server,
                                        db_name=conn_dbname,
                                    )
                                    st.success(f"Connected and registered '{new_c.name}' successfully!")
                                    st.session_state.active_connection_id = new_c.id
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Connection test failed: {e}")

                else:  # Direct URI
                    with st.form("uri_db_conn_form"):
                        conn_name = st.text_input("Connection Name", placeholder="e.g. Supabase Production")
                        conn_type = st.selectbox("Dialect", options=["mssql", "postgresql", "mysql", "sqlite"])
                        conn_uri = st.text_input("Connection URI", placeholder="e.g. sqlite:///data/my_app.db or postgresql://...")
                        submit_uri = st.form_submit_button("Test & Save URI")

                        if submit_uri:
                            if not conn_name or not conn_uri:
                                st.error("Name and Connection URI are required.")
                            else:
                                from sqlalchemy import create_engine
                                try:
                                    test_engine = create_engine(conn_uri)
                                    with test_engine.connect() as test_conn:
                                        pass

                                    new_c = st.session_state.tenant_mgr.register_connection(
                                        user_id=user.id,
                                        name=conn_name,
                                        db_type=conn_type,
                                        db_uri=conn_uri,
                                    )
                                    st.success(f"Registered connection '{new_c.name}'!")
                                    st.session_state.active_connection_id = new_c.id
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Connection failed: {e}")

            with tab_manage:
                if not user_conns:
                    st.info("No custom databases registered yet.")
                else:
                    for c in user_conns:
                        col_info, col_del = st.columns([3, 1])
                        with col_info:
                            st.markdown(f"**{c.name}** (`{c.db_type.upper()}`)")
                            if c.db_name:
                                st.caption(f"DB: `{c.db_name}`")
                        with col_del:
                            if st.button("🗑️", key=f"del_conn_{c.id}", help=f"Delete {c.name}"):
                                st.session_state.tenant_mgr.delete_connection(user.id, c.id)
                                if st.session_state.active_connection_id == c.id:
                                    st.session_state.active_connection_id = None
                                    st.session_state.agent = None
                                    st.session_state.db = None
                                st.success(f"Deleted {c.name}")
                                st.rerun()

        st.markdown("---")
        st.markdown("### ⚙️ Session Controls")

        if st.button("🧹 Clear Chat & Reset Memory", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_sql = None
            st.session_state.last_df = None
            if st.session_state.agent:
                st.session_state.agent.chat_history.clear()
            st.rerun()

        if st.button("🔄 Refresh Schema & Config", use_container_width=True):
            st.session_state.config = Config.from_env()
            st.session_state.agent = None
            st.session_state.db = None
            st.session_state.schema_info = None
            st.success("Config & database schema reloaded!")
            st.rerun()

        st.markdown("---")
        st.caption("AgenticSQL v1.0.0 • Read-Only Safe")


def render_chart_studio(df: pd.DataFrame) -> None:
    """Render dynamic, interactive Plotly charting studio for tabular results."""
    st.markdown("#### 📊 Auto-Visualization Studio")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()

    if not all_cols:
        st.info("No columns available for charting.")
        return

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        chart_type = st.selectbox(
            "Chart Type",
            options=["Bar Chart", "Line Chart", "Area Chart", "Pie Chart", "Scatter Plot"],
            key="ui_chart_type",
        )

    with c2:
        x_default = all_cols[0] if all_cols else None
        x_col = st.selectbox("X-Axis (Category / Dimension)", options=all_cols, index=0, key="ui_x_col")

    with c3:
        y_options = numeric_cols if numeric_cols else all_cols
        y_default_idx = 0 if y_options else 0
        if len(y_options) > 1 and x_col in y_options and y_options[0] == x_col:
            y_default_idx = 1
        y_col = st.selectbox("Y-Axis (Metric / Value)", options=y_options, index=y_default_idx, key="ui_y_col")

    color_discrete = ["#f59e0b", "#38bdf8", "#34d399", "#a855f7", "#f43f5e", "#fbbf24"]

    try:
        if chart_type == "Bar Chart":
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                title=f"{y_col} by {x_col}",
                template="plotly_dark",
                color_discrete_sequence=color_discrete,
            )
        elif chart_type == "Line Chart":
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                title=f"{y_col} Trend by {x_col}",
                template="plotly_dark",
                markers=True,
                color_discrete_sequence=color_discrete,
            )
        elif chart_type == "Area Chart":
            fig = px.area(
                df,
                x=x_col,
                y=y_col,
                title=f"{y_col} Distribution across {x_col}",
                template="plotly_dark",
                color_discrete_sequence=color_discrete,
            )
        elif chart_type == "Pie Chart":
            fig = px.pie(
                df,
                names=x_col,
                values=y_col if y_col in numeric_cols else None,
                title=f"Share of {y_col} by {x_col}",
                template="plotly_dark",
                color_discrete_sequence=color_discrete,
            )
        else:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{x_col} vs {y_col}",
                template="plotly_dark",
                color_discrete_sequence=color_discrete,
            )

        fig.update_layout(
            paper_bgcolor="#1e1f20",
            plot_bgcolor="#131314",
            font_family="Google Sans, Inter, sans-serif",
            font_color="#e3e3e3",
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not render chart: {e}")


def handle_user_query(user_query: str, agent: AgenticSQLAgent) -> None:
    """Execute user query via tenant agent and append to session state & persistent history."""
    user = st.session_state.current_user

    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })

    with st.spinner("🤖 AgenticSQL is analyzing schema, generating SQL, and querying database..."):
        start_t = time.time()
        try:
            response = agent.chat(user_query)
            latency = round(time.time() - start_t, 2)

            output_text = response.get("output", "No response received.")
            sql_list = response.get("sql", [])
            data_dict = response.get("data", None)
            healed = response.get("healed", False)
            attempts = response.get("attempts", 1)
            explanation = response.get("explanation", "")
            cost = response.get("cost", "LOW")
            profiling_tips = response.get("profiling_tips", [])
            profiling_warnings = response.get("profiling_warnings", [])

            # Store in session state
            st.session_state.last_sql = sql_list[-1] if sql_list else None

            # Reconstruct DataFrame if present
            if data_dict and "columns" in data_dict and "rows" in data_dict:
                df = pd.DataFrame(data_dict["rows"], columns=data_dict["columns"])
                st.session_state.last_df = df
            else:
                st.session_state.last_df = None

            # Persist query to metadata database for tenant
            if user:
                sql_str = "; ".join(sql_list) if sql_list else ""
                status = "BLOCKED" if "BLOCKED" in output_text else ("ERROR" if "[ERROR]" in output_text else "SUCCESS")
                st.session_state.auth_db.log_query(
                    user_id=user.id,
                    connection_id=st.session_state.active_connection_id,
                    question=user_query,
                    sql=sql_str,
                    latency=latency,
                    status=status,
                )

            st.session_state.messages.append({
                "role": "assistant",
                "content": output_text,
                "sql": sql_list,
                "latency": latency,
                "data": data_dict,
                "healed": healed,
                "attempts": attempts,
                "explanation": explanation,
                "cost": cost,
                "profiling_tips": profiling_tips,
                "profiling_warnings": profiling_warnings,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })

        except Exception as e:
            logger.error("Error running agent chat: %s", e)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ An error occurred while executing the query: {e}",
                "sql": [],
                "latency": 0.0,
                "data": None,
                "healed": False,
                "attempts": 1,
                "explanation": "",
                "cost": "LOW",
                "profiling_tips": [],
                "profiling_warnings": [],
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })


def render_chat_tab(agent: Optional[AgenticSQLAgent]) -> None:
    """Render Tab 1: Conversational Chat & Analysis Studio."""
    st.markdown("**💡 Quick Suggestions:**")
    q_cols = st.columns(4)
    samples = [
        "Show all tables and their row counts",
        "List top 5 records from the primary table",
        "Summarize revenue or activity by category",
        "Find records created in the last 30 days",
    ]

    for i, sample_q in enumerate(samples):
        with q_cols[i]:
            if st.button(sample_q, key=f"quick_btn_{i}", use_container_width=True):
                if agent:
                    handle_user_query(sample_q, agent)
                    st.rerun()
                else:
                    st.error("Database agent not connected.")

    st.markdown("---")

    # Chat Messages History
    for idx, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "⚡"):
            st.markdown(msg["content"])

            # Self-healing indicator
            if msg.get("healed"):
                st.markdown(
                    f'<div class="healed-badge">🔄 Self-Healed (Repaired in {msg.get("attempts", 2)} attempts)</div>',
                    unsafe_allow_html=True,
                )

            # Query Performance Rating Badge (Phase 4c)
            if role == "assistant" and msg.get("cost"):
                cost = msg.get("cost", "LOW")
                badge_bg = "rgba(16, 185, 129, 0.15)" if cost == "LOW" else ("rgba(245, 158, 11, 0.15)" if cost == "MEDIUM" else "rgba(239, 68, 68, 0.15)")
                badge_color = "#34d399" if cost == "LOW" else ("#fbbf24" if cost == "MEDIUM" else "#f87171")
                st.markdown(
                    f'<span style="display:inline-block; margin-top:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; border-radius:4px; background:{badge_bg}; color:{badge_color}; border:1px solid {badge_color}40;">⚡ Query Cost: {cost}</span>',
                    unsafe_allow_html=True,
                )

            # Render SQL if present
            if role == "assistant" and msg.get("sql"):
                for s_idx, q in enumerate(msg["sql"]):
                    with st.expander(f"🔍 Generated SQL Query #{s_idx + 1}", expanded=False):
                        st.code(q, language="sql")

            # Render Educational Explanation (Phase 4b)
            if role == "assistant" and msg.get("explanation"):
                with st.expander("💡 Plain-English Query Breakdown", expanded=False):
                    st.markdown(msg["explanation"])

            # Render Performance Optimization Tips (Phase 4c)
            tips = msg.get("profiling_tips", [])
            warnings = msg.get("profiling_warnings", [])
            if role == "assistant" and (tips or warnings):
                with st.expander("⚙️ AST Performance Diagnostics & Optimization Tips", expanded=False):
                    if warnings:
                        st.markdown("**⚠️ Warnings:**")
                        for w in warnings:
                            st.markdown(f"- {w}")
                    if tips:
                        st.markdown("**💡 Optimization Recommendations:**")
                        for t in tips:
                            st.markdown(f"- {t}")

            # Bookmark as Golden Exemplar button (Phase 4c)
            if role == "assistant" and msg.get("sql") and agent:
                col_btn, _ = st.columns([2, 5])
                with col_btn:
                    if st.button("⭐ Bookmark as Golden Exemplar", key=f"btn_save_ex_{idx}"):
                        user_query_text = st.session_state.messages[idx - 1]["content"] if idx > 0 else "Query"
                        agent.add_golden_example(user_query_text, msg["sql"][-1])
                        st.toast("✓ Golden Exemplar saved to Few-Shot repository!", icon="⭐")

            if "latency" in msg and msg["latency"]:
                st.caption(f"⏱️ Executed in `{msg['latency']}s` at `{msg.get('timestamp', '')}`")

    # Dynamic Data Table & Chart Studio for the latest result
    if st.session_state.last_df is not None and not st.session_state.last_df.empty:
        df = st.session_state.last_df
        st.markdown("---")
        st.markdown("### 📋 Query Results & Visual Studio")

        tab_data, tab_chart = st.tabs(["📄 Data Table", "📈 Dynamic Chart Studio"])

        with tab_data:
            st.dataframe(df, use_container_width=True, height=280)
            st.caption(f"Total Rows: **{len(df)}** | Total Columns: **{len(df.columns)}**")

            # 1-Click Export Buttons
            exp_c1, exp_c2, exp_c3 = st.columns([1, 1, 3])
            with exp_c1:
                csv_data = dataframe_to_csv(df)
                st.download_button(
                    "📥 Download CSV",
                    data=csv_data,
                    file_name=f"agenticsql_export_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with exp_c2:
                json_data = dataframe_to_json(df)
                st.download_button(
                    "📥 Download JSON",
                    data=json_data,
                    file_name=f"agenticsql_export_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True,
                )

        with tab_chart:
            render_chart_studio(df)

    # Chat Input Box
    user_prompt = st.chat_input("Ask a question about your database (e.g. 'Show monthly total orders grouped by status')...")
    if user_prompt:
        if not agent:
            st.error("Cannot query: Database agent is not initialized. Please verify your credentials.")
        else:
            handle_user_query(user_prompt, agent)
            st.rerun()


def render_schema_tab(db: Optional[Any]) -> None:
    """Render Tab 2: Live Database Schema Browser."""
    st.markdown("### 🗄️ Database Schema Explorer")
    st.caption("Inspect live tables, data types, foreign keys, and preview data.")

    if not db or not st.session_state.schema_info:
        st.info("Database not connected. Schema information unavailable.")
        return

    schema_dict = st.session_state.schema_info
    tables = list(schema_dict.keys())

    if not tables:
        st.warning("No tables discovered in the connected database.")
        return

    col_left, col_right = st.columns([1, 2])

    with col_left:
        selected_table = st.selectbox("Select Table", options=tables)

        if selected_table:
            st.markdown("#### Table Definition (DDL)")
            st.code(schema_dict[selected_table], language="sql")

    with col_right:
        if selected_table and st.session_state.agent:
            st.markdown(f"#### 🔍 Sample Data: `{selected_table}` (Top 10 Rows)")
            preview_sql = f"SELECT * FROM {selected_table} LIMIT 10;"
            dialect = getattr(db, "dialect", "")
            if "mssql" in str(dialect).lower():
                preview_sql = f"SELECT TOP (10) * FROM [{selected_table}];"

            preview_df = st.session_state.agent.execute_sql(preview_sql)
            if preview_df is not None and not preview_df.empty:
                st.dataframe(preview_df, use_container_width=True, height=350)
            else:
                st.info(f"No sample rows found in `{selected_table}`.")

    # Few-Shot Exemplars Explorer (Phase 4c)
    if st.session_state.agent and hasattr(st.session_state.agent, "fewshot_store"):
        st.markdown("---")
        st.markdown("#### ⭐ Few-Shot Golden Exemplars Repository (Phase 4c)")
        st.caption("Active exemplars used by semantic retrieval to ground LLM query generation.")
        exemplars = st.session_state.agent.fewshot_store.examples

        with st.expander(f"📚 View Active Exemplars ({len(exemplars)} Golden Pairs)", expanded=False):
            for ex_i, ex in enumerate(exemplars, 1):
                st.markdown(f"**#{ex_i} • [{ex.dialect.upper()}]** {ex.question}")
                st.code(ex.sql, language="sql")


def render_user_history_tab() -> None:
    """Render Tab 3: User's Isolated Query History."""
    user = st.session_state.current_user
    st.markdown("### 🕒 My Query History")
    st.caption("Historical queries executed in your tenant workspace.")

    if not user:
        st.info("Please sign in to view query history.")
        return

    history = st.session_state.auth_db.get_user_query_history(user.id, limit=100)

    if not history:
        st.info("No queries executed yet. Run queries in the AI Chat tab to see history.")
        return

    history_records = []
    for h in history:
        history_records.append({
            "Timestamp": h.created_at[:19].replace("T", " "),
            "Question": h.question,
            "Executed SQL": h.sql,
            "Latency (s)": h.latency,
            "Status": h.status,
        })

    hist_df = pd.DataFrame(history_records)
    st.dataframe(hist_df, use_container_width=True, height=450)


def render_guardrails_tab() -> None:
    """Render Tab 4: Safety Guardrails & Real-Time Audit Log Monitor."""
    st.markdown("### 🛡️ SQL Safety Guardrails & Audit Stream")
    st.caption("All queries pass through SQLGlot AST validation before execution. Destructive DDL/DML is blocked.")

    audit_log_path = Path("logs/query_audit.log")

    if not audit_log_path.exists():
        st.info("Audit log `logs/query_audit.log` has not been created yet.")
        return

    try:
        with open(audit_log_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        total_queries = len(lines)
        blocked_count = sum(1 for line in lines if "BLOCKED" in line)
        allowed_count = sum(1 for line in lines if "ALLOWED" in line)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val">{total_queries}</div><div class="metric-lbl">Total Queries Inspected</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val" style="color: #34d399;">{allowed_count}</div><div class="metric-lbl">Safe Queries Executed</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val" style="color: #f87171;">{blocked_count}</div><div class="metric-lbl">Dangerous Statements Blocked</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### 📜 Live Query Audit Stream (Recent 30 Events)")

        recent_lines = lines[-30:][::-1]
        log_records = []
        for line in recent_lines:
            status = "BLOCKED" if "BLOCKED" in line else "ALLOWED"
            log_records.append({
                "Event": line,
                "Status": status,
            })

        log_df = pd.DataFrame(log_records)
        st.dataframe(log_df, use_container_width=True, height=350)

    except Exception as e:
        st.error(f"Error reading audit log: {e}")


def render_admin_tab() -> None:
    """Render Tab 5: Executive Admin Usage & Analytics Dashboard."""
    st.markdown("### 👑 Administrator Usage & Analytics Dashboard")
    st.caption("System-wide metrics across all users, tenants, registered databases, and security events.")

    stats = st.session_state.auth_db.get_admin_stats()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{stats["total_users"]}</div><div class="metric-lbl">Total Users</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{stats["total_connections"]}</div><div class="metric-lbl">Registered DBs</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val" style="color: #38bdf8;">{stats["total_queries"]}</div><div class="metric-lbl">Total Queries</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val" style="color: #f59e0b;">{stats["avg_latency_seconds"]}s</div><div class="metric-lbl">Avg Query Latency</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_u, col_d = st.columns(2)

    with col_u:
        st.markdown("#### 👥 Top User Activity")
        if stats["top_users"]:
            user_df = pd.DataFrame(stats["top_users"])
            fig_user = px.bar(
                user_df,
                x="username",
                y="queries",
                title="Queries Executed by User",
                template="plotly_dark",
                color_discrete_sequence=["#38bdf8"],
            )
            fig_user.update_layout(
                paper_bgcolor="#1e1f20",
                plot_bgcolor="#131314",
                font_family="Google Sans, Inter, sans-serif",
                font_color="#e3e3e3",
            )
            st.plotly_chart(fig_user, use_container_width=True)
        else:
            st.info("No user query activity recorded yet.")

    with col_d:
        st.markdown("#### 🗄️ Database Dialect Distribution")
        if stats["database_dialects"]:
            dialect_df = pd.DataFrame(stats["database_dialects"])
            fig_d = px.pie(
                dialect_df,
                names="dialect",
                values="count",
                title="Registered Database Types",
                template="plotly_dark",
                color_discrete_sequence=["#f59e0b", "#38bdf8", "#34d399", "#a855f7"],
            )
            fig_d.update_layout(
                paper_bgcolor="#1e1f20",
                plot_bgcolor="#131314",
                font_family="Google Sans, Inter, sans-serif",
                font_color="#e3e3e3",
            )
            st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.info("No custom databases registered yet.")


def main() -> None:
    """Streamlit Application Main Entry Point."""
    st.set_page_config(
        page_title="AgenticSQL Studio",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()

    # If user is not logged in, show login/register view
    if st.session_state.current_user is None:
        render_login_screen()
        return

    # User is authenticated — load tenant backend
    agent, db, err = load_backend_for_tenant()

    render_sidebar(st.session_state.config, db)
    render_header(st.session_state.config, is_connected=(agent is not None))

    if err:
        st.error(f"⚠️ **Database Connection Notice**: {err}")

    # Build tab titles dynamically based on user role
    tabs_list = [
        "💬 AI Chat & Analytics",
        "🗄️ Database Schema Explorer",
        "🕒 My Query History",
        "🛡️ Safety Guardrails & Audit",
    ]

    is_admin = st.session_state.current_user.role == "admin"
    if is_admin:
        tabs_list.append("👑 Admin Analytics")

    rendered_tabs = st.tabs(tabs_list)

    with rendered_tabs[0]:
        render_chat_tab(agent)

    with rendered_tabs[1]:
        render_schema_tab(db)

    with rendered_tabs[2]:
        render_user_history_tab()

    with rendered_tabs[3]:
        render_guardrails_tab()

    if is_admin:
        with rendered_tabs[4]:
            render_admin_tab()


if __name__ == "__main__":
    main()
