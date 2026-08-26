"""
SQL Agent with conversation memory and safety guardrails.

Wraps LangChain's create_sql_agent with:
- A SafeSQLDatabase proxy that validates queries before execution
- Conversation memory (sliding window of recent turns)
- SQL extraction from intermediate agent steps
- Custom system prompt enforcing read-only behavior
"""

import logging
from typing import Optional
import pandas as pd

# pyrefly: ignore [missing-import]
from langchain_community.utilities import SQLDatabase
# pyrefly: ignore [missing-import]
from langchain_community.agent_toolkits import create_sql_agent
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI

from . import guardrails

logger = logging.getLogger(__name__)

def get_dialect_prompt(dialect: Optional[str] = None) -> str:
    """
    Generate dialect-aware system instructions for the agent LLM.

    Customizes syntax guidance (row limits, identifier quoting, date functions)
    based on whether the database is PostgreSQL, MySQL, MSSQL, SQLite, or CockroachDB.
    """
    dialect_clean = (dialect or "mssql").lower().strip()

    if "postgres" in dialect_clean or "cockroach" in dialect_clean:
        dialect_rules = """3. POSTGRESQL / COCKROACHDB DIALECT:
   - Use LIMIT (N) for limiting returned rows (never TOP).
   - Use double quotes "table_name"."column_name" when identifiers contain mixed case or reserved words.
   - Use ILIKE for case-insensitive pattern matching.
   - Use standard PostgreSQL date/time functions (e.g., NOW(), DATE_TRUNC('month', created_at), CURRENT_DATE).
   - Qualify tables with their schema (e.g., public.users) when appropriate."""
    elif "mysql" in dialect_clean or "mariadb" in dialect_clean:
        dialect_rules = """3. MYSQL / MARIADB DIALECT:
   - Use LIMIT (N) for limiting returned rows (never TOP).
   - Use backticks `table_name`.`column_name` when identifiers contain spaces or reserved words.
   - Use standard MySQL date/time functions (e.g., NOW(), DATE_FORMAT(created_at, '%Y-%m-%d'), CURDATE())."""
    elif "sqlite" in dialect_clean:
        dialect_rules = """3. SQLITE DIALECT:
   - Use LIMIT (N) for limiting returned rows.
   - Use standard SQLite functions (e.g., datetime('now'), strftime('%Y-%m', date_col))."""
    else:  # Default / MSSQL / Azure SQL
        dialect_rules = """3. T-SQL / MSSQL DIALECT:
   - Use TOP (N) instead of LIMIT.
   - Use square brackets [table_name].[column_name] when identifiers contain spaces or reserved words.
   - Use standard T-SQL date functions (e.g., DATEDIFF, DATEADD, GETDATE())."""

    return f"""You are AgenticSQL, an expert AI assistant that helps users query and understand their database using natural language.

CRITICAL RULES:
1. READ-ONLY ACCESS: You can ONLY execute SELECT queries. Never execute DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, EXEC, or any data-modifying statements.
2. SCHEMA INSPECTION: Always check table schemas before constructing queries to ensure column names and types exist.
{dialect_rules}
4. FORMATTING: Present query results clearly in clean Markdown tables accompanied by concise, helpful explanations.
5. EDUCATIONAL: Always present the exact SQL query executed so users can learn and verify.
6. CONVERSATION CONTEXT: Use prior conversation history to resolve follow-up questions and references."""


# Default system prompt prefix
AGENT_PREFIX = get_dialect_prompt("mssql")


def _apply_guardrails(db: SQLDatabase) -> SQLDatabase:
    """
    Apply safety guardrails to a SQLDatabase by monkey-patching its run() method.

    This preserves the original SQLDatabase instance (passing isinstance checks
    required by LangChain's Pydantic validation) while routing all query
    execution through the AST guardrails validator.
    """
    original_run = db.run
    dialect = getattr(db, "dialect", None)

    def guarded_run(command: str, fetch: str = "all", **kwargs) -> str:
        is_safe, reason = guardrails.validate_sql(command, dialect=str(dialect) if dialect else None)
        if not is_safe:
            return guardrails.format_blocked_message(reason)
        return original_run(command, fetch=fetch, **kwargs)

    db.run = guarded_run  # type: ignore[method-assign]
    return db


class AgenticSQLAgent:
    """
    Modern SQL Agent with conversation memory, guardrails, and tool-calling execution.

    Usage:
        agent = AgenticSQLAgent(llm=llm, db=db)
        response = agent.chat("How many customers are there?")
        print(response["output"])
        print(response["sql"])  # List of SQL queries executed
    """

    def __init__(
        self,
        llm: ChatGoogleGenerativeAI,
        db: SQLDatabase,
        verbose: bool = False,
        memory_window: int = 10,
        agent_type: str = "tool-calling",
    ):
        """
        Args:
            llm: The initialized LLM instance.
            db: The database connection.
            verbose: If True, print agent's intermediate reasoning steps.
            memory_window: Number of conversation turns to retain for context.
            agent_type: 'tool-calling' (default) or 'zero-shot-react-description'.
        """
        self.llm = llm
        self.db = db
        self.verbose = verbose
        self.memory_window = memory_window
        self.agent_type = agent_type

        self.chat_history: list[dict] = []
        self.last_sql: Optional[str] = None
        self.last_df: Optional[pd.DataFrame] = None

        # Detect dialect and generate dialect-tailored system prompt
        self.dialect = getattr(db, "dialect", None) or "mssql"
        prefix = get_dialect_prompt(str(self.dialect))

        # Apply safety guardrails directly to the db instance
        _apply_guardrails(db)

        # Initialize the LangChain SQL agent with tool-calling and intermediate step capture
        try:
            self._agent = create_sql_agent(
                llm=self.llm,
                db=self.db,
                agent_type=self.agent_type,
                verbose=self.verbose,
                prefix=prefix,
                agent_executor_kwargs={"return_intermediate_steps": True},
            )
        except Exception as e:
            logger.warning(
                "Could not initialize agent with agent_type='%s' (%s). Falling back to 'zero-shot-react-description'.",
                self.agent_type,
                e,
            )
            self._agent = create_sql_agent(
                llm=self.llm,
                db=self.db,
                agent_type="zero-shot-react-description",
                verbose=self.verbose,
                handle_parsing_errors=True,
                prefix=prefix,
                agent_executor_kwargs={"return_intermediate_steps": True},
            )

    def execute_sql(self, sql: str) -> Optional[pd.DataFrame]:
        """
        Execute a safe SQL query directly against the database engine and return a DataFrame.

        Enforces read-only safety guardrails before execution.

        Args:
            sql: The SQL query statement.

        Returns:
            A pandas DataFrame with the query results, or None if invalid/blocked.
        """
        if not sql or not sql.strip():
            return None

        is_safe, reason = guardrails.validate_sql(sql, dialect=str(self.dialect) if self.dialect else None)
        if not is_safe:
            logger.warning("Direct SQL execution blocked: %s", reason)
            return None

        try:
            # pyrefly: ignore [missing-import]
            from sqlalchemy import text

            engine = getattr(self.db, "_engine", None)
            if engine is None:
                return None

            with engine.connect() as conn:
                df = pd.read_sql_query(text(sql), conn)
                return df

        except Exception as e:
            logger.error("Direct SQL execution error: %s", e)
            return None

    def chat(self, user_input: str) -> dict:
        """
        Send a message to the agent and get a response.

        Automatically includes conversation history for multi-turn context
        and captures executed tabular data as a structured DataFrame.

        Args:
            user_input: The user's natural language question.

        Returns:
            A dict with keys:
                - 'output': The agent's text response.
                - 'sql': List of SQL queries generated (may be empty).
                - 'data': Dict with 'columns' and 'rows' (or None).
        """
        # Build context from conversation history
        history_context = self._build_history_context()

        # Construct the full prompt with history for multi-turn support
        if history_context:
            full_input = (
                f"Previous conversation context:\n{history_context}\n\n"
                f"Current user question: {user_input}"
            )
        else:
            full_input = user_input

        try:
            response = self._agent.invoke({"input": full_input})
            output = response.get("output", "No response generated.")

            # Extract SQL queries executed during intermediate steps
            sql_queries = self._extract_sql(response)

            # Direct DataFrame capture from the last executed SQL query
            structured_data = None
            if sql_queries:
                self.last_sql = sql_queries[-1]
                self.last_df = self.execute_sql(self.last_sql)
                if self.last_df is not None and not self.last_df.empty:
                    from .visualization import dataframe_to_dict
                    structured_data = dataframe_to_dict(self.last_df)
            else:
                self.last_sql = None
                self.last_df = None

            # Update conversation history
            self.chat_history.append({
                "role": "user",
                "content": user_input,
            })
            self.chat_history.append({
                "role": "assistant",
                "content": output,
                "sql": sql_queries,
            })

            # Trim history to the configured window size
            max_entries = self.memory_window * 2  # user + assistant pairs
            if len(self.chat_history) > max_entries:
                self.chat_history = self.chat_history[-max_entries:]

            return {
                "output": output,
                "sql": sql_queries,
                "data": structured_data,
            }

        except Exception as e:
            logger.error("Agent error: %s", e, exc_info=True)
            return {
                "output": (
                    f"[ERROR] An error occurred: {e}\n\n"
                    "Please try rephrasing your question, or check the logs for details."
                ),
                "sql": [],
                "data": None,
            }

    def _build_history_context(self) -> str:
        """Build a conversation history string for the agent's context window."""
        if not self.chat_history:
            return ""

        lines: list[str] = []
        for entry in self.chat_history:
            role = "User" if entry["role"] == "user" else "Assistant"
            lines.append(f"{role}: {entry['content']}")
            if entry.get("sql"):
                lines.append(f"  [Executed SQL: {'; '.join(entry['sql'])}]")
        return "\n".join(lines)

    @staticmethod
    def _extract_sql(response: dict) -> list[str]:
        """
        Extract executed SQL queries from the agent's intermediate steps.

        Supports ToolAgentAction, ReAct string actions, and dictionary inputs.
        """
        sql_queries: list[str] = []
        intermediate_steps = response.get("intermediate_steps", [])

        for step in intermediate_steps:
            if not (hasattr(step, "__len__") and len(step) >= 1):
                continue

            action = step[0]

            # Tool calling tool name check
            tool_name = getattr(action, "tool", "")

            # Check for tool_input as a string or dict
            tool_input = getattr(action, "tool_input", None)

            query_candidate = None
            if isinstance(tool_input, str):
                query_candidate = tool_input.strip()
            elif isinstance(tool_input, dict):
                query_candidate = tool_input.get("query") or tool_input.get("command") or tool_input.get("sql")

            if query_candidate and isinstance(query_candidate, str):
                cleaned = query_candidate.strip().strip("`").strip()
                if cleaned.startswith("sql"):
                    cleaned = cleaned[3:].strip()

                upper = cleaned.upper()
                if any(kw in upper for kw in ["SELECT", "SHOW", "DESCRIBE", "SP_HELP", "INFORMATION_SCHEMA", "WITH"]):
                    if cleaned not in sql_queries:
                        sql_queries.append(cleaned)
                elif tool_name in ["sql_db_query", "sql_db_query_checker"] and cleaned not in sql_queries:
                    sql_queries.append(cleaned)

        return sql_queries

    def clear_history(self) -> None:
        """Clear conversation history, last SQL cache, and cached DataFrame."""
        self.chat_history.clear()
        self.last_sql = None
        self.last_df = None
        logger.info("Conversation history cleared.")

    def get_history(self) -> list[dict]:
        """Get a copy of the conversation history."""
        return self.chat_history.copy()

    def get_schema(self) -> str:
        """Get the database schema information as a string."""
        try:
            return self.db.get_table_info()
        except Exception as e:
            logger.error("Error retrieving schema: %s", e)
            return f"Error retrieving schema: {e}"
