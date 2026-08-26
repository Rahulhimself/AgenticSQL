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

# pyrefly: ignore [missing-import]
from langchain_community.utilities import SQLDatabase
# pyrefly: ignore [missing-import]
from langchain_community.agent_toolkits import create_sql_agent
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI

from . import guardrails

logger = logging.getLogger(__name__)

# System prompt prefix injected into the agent's instructions
AGENT_PREFIX = """You are AgenticSQL, an AI assistant that helps users query and understand their database using natural language.

RULES:
1. You can ONLY execute SELECT queries. Never execute DROP, DELETE, UPDATE, INSERT, ALTER, or any data-modifying statements.
2. Always inspect the database schema before writing queries.
3. When presenting results, format them as clean, readable tables and add a brief explanation.
4. If a user asks you to modify data, politely decline and explain you are read-only.
5. If a query returns many rows, summarize the key findings instead of dumping everything.
6. Always show the SQL query you generated so the user can learn from it.
7. Use proper T-SQL syntax for Microsoft SQL Server.
8. For follow-up questions, use context from the conversation history provided.

You have access to the following database. Use the tools to inspect schema and run queries."""


def _apply_guardrails(db: SQLDatabase) -> SQLDatabase:
    """
    Apply safety guardrails to a SQLDatabase by monkey-patching its run() method.

    This preserves the original SQLDatabase instance (passing isinstance checks
    required by LangChain's Pydantic validation) while routing all query
    execution through the guardrails validator.
    """
    original_run = db.run

    def guarded_run(command: str, fetch: str = "all", **kwargs) -> str:
        is_safe, reason = guardrails.validate_sql(command)
        if not is_safe:
            return guardrails.format_blocked_message(reason)
        return original_run(command, fetch=fetch, **kwargs)

    db.run = guarded_run  # type: ignore[method-assign]
    return db


class AgenticSQLAgent:
    """
    SQL Agent with conversation memory, guardrails, and query tracking.

    Usage:
        agent = AgenticSQLAgent(llm=llm, db=db)
        response = agent.chat("How many customers are there?")
        print(response["output"])
        print(response["sql"])  # List of SQL queries generated
    """

    def __init__(
        self,
        llm: ChatGoogleGenerativeAI,
        db: SQLDatabase,
        verbose: bool = False,
        memory_window: int = 10,
    ):
        """
        Args:
            llm: The initialized LLM instance.
            db: The database connection.
            verbose: If True, print agent's intermediate reasoning steps.
            memory_window: Number of conversation turns to retain for context.
        """
        self.llm = llm
        self.db = db
        self.verbose = verbose
        self.memory_window = memory_window

        self.chat_history: list[dict] = []
        self.last_sql: Optional[str] = None

        # Apply safety guardrails directly to the db instance
        _apply_guardrails(db)

        # Create the LangChain SQL agent
        self._agent = create_sql_agent(
            llm=self.llm,
            db=self.db,
            agent_type="zero-shot-react-description",
            verbose=self.verbose,
            handle_parsing_errors=True,
            prefix=AGENT_PREFIX,
        )

    def chat(self, user_input: str) -> dict:
        """
        Send a message to the agent and get a response.

        Automatically includes conversation history for multi-turn context.

        Args:
            user_input: The user's natural language question.

        Returns:
            A dict with keys:
                - 'output': The agent's text response.
                - 'sql': List of SQL queries generated (may be empty).
        """
        # Build context from conversation history
        history_context = self._build_history_context()

        # Construct the full prompt with history for multi-turn support
        if history_context:
            full_input = (
                f"Previous conversation:\n{history_context}\n\n"
                f"Current question: {user_input}"
            )
        else:
            full_input = user_input

        try:
            response = self._agent.invoke({"input": full_input})
            output = response.get("output", "No response generated.")

            # Extract SQL from intermediate steps if available
            sql_queries = self._extract_sql(response)

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

            self.last_sql = sql_queries[-1] if sql_queries else None

            return {
                "output": output,
                "sql": sql_queries,
            }

        except Exception as e:
            logger.error("Agent error: %s", e, exc_info=True)
            return {
                "output": (
                    f"[ERROR] An error occurred: {e}\n\n"
                    "Please try rephrasing your question, or check the logs for details."
                ),
                "sql": [],
            }

    def _build_history_context(self) -> str:
        """Build a conversation history string for the agent's context window."""
        if not self.chat_history:
            return ""

        lines: list[str] = []
        for entry in self.chat_history:
            role = "User" if entry["role"] == "user" else "Assistant"
            lines.append(f"{role}: {entry['content']}")
        return "\n".join(lines)

    @staticmethod
    def _extract_sql(response: dict) -> list[str]:
        """
        Extract SQL queries from the agent's intermediate steps.

        The agent's response may contain intermediate_steps with the
        tool actions it took. We look for SQL-like strings in tool inputs.
        """
        sql_queries: list[str] = []
        intermediate_steps = response.get("intermediate_steps", [])

        for step in intermediate_steps:
            if not (hasattr(step, "__len__") and len(step) >= 2):
                continue

            action = step[0]

            # Check for tool_input as a string (common with ReAct agents)
            tool_input = getattr(action, "tool_input", None)
            if isinstance(tool_input, str):
                upper = tool_input.upper().strip()
                if any(kw in upper for kw in ["SELECT", "SHOW", "DESCRIBE", "SP_HELP", "INFORMATION_SCHEMA"]):
                    sql_queries.append(tool_input)
            elif isinstance(tool_input, dict):
                query = tool_input.get("query", "")
                if query:
                    sql_queries.append(query)

        return sql_queries

    def clear_history(self) -> None:
        """Clear conversation history and last SQL cache."""
        self.chat_history.clear()
        self.last_sql = None
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
