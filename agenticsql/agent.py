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
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from . import guardrails
from .fewshot import FewShotStore
from .profiler import QueryProfiler

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
        llm: BaseChatModel,
        db: SQLDatabase,
        verbose: bool = False,
        memory_window: int = 10,
        agent_type: str = "tool-calling",
        max_retries: int = 3,
        enable_self_healing: bool = True,
        enable_schema_pruning: bool = True,
    ):
        """
        Args:
            llm: The initialized LLM instance.
            db: The database connection.
            verbose: If True, print agent's intermediate reasoning steps.
            memory_window: Number of conversation turns to retain for context.
            agent_type: 'tool-calling' (default) or 'zero-shot-react-description'.
            max_retries: Max number of self-healing attempts on query errors.
            enable_self_healing: Enable automated error reflection and query repair.
            enable_schema_pruning: Dynamically prune irrelevant tables for large schemas.
        """
        self.llm = llm
        self.db = db
        self.verbose = verbose
        self.memory_window = memory_window
        self.agent_type = agent_type
        self.max_retries = max_retries
        self.enable_self_healing = enable_self_healing
        self.enable_schema_pruning = enable_schema_pruning

        self.chat_history: list[dict] = []
        self.last_sql: Optional[str] = None
        self.last_df: Optional[pd.DataFrame] = None

        # Detect dialect and generate dialect-tailored system prompt
        self.dialect = getattr(db, "dialect", None) or "mssql"
        prefix = get_dialect_prompt(str(self.dialect))

        # Phase 4c: Few-Shot RAG Exemplar Store & Query Profiler
        self.fewshot_store = FewShotStore()
        self.profiler = QueryProfiler(dialect=str(self.dialect))

        # Apply safety guardrails directly to the db instance
        _apply_guardrails(db)

        # Construct clean Gemini-compatible prompt without model prefilling
        custom_prompt = ChatPromptTemplate.from_messages([
            ("system", prefix),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Initialize the LangChain SQL agent with tool-calling and intermediate step capture
        agent_type_target = "openai-tools" if self.agent_type in ("openai-tools", "tool-calling") else self.agent_type
        try:
            self._agent = create_sql_agent(
                llm=self.llm,
                db=self.db,
                prompt=custom_prompt,
                agent_type=agent_type_target,
                verbose=self.verbose,
                agent_executor_kwargs={"return_intermediate_steps": True},
            )
        except Exception as e:
            logger.warning(
                "Could not initialize agent with prompt and agent_type='%s' (%s). Retrying with standard executor.",
                agent_type_target,
                e,
            )
            self._agent = create_sql_agent(
                llm=self.llm,
                db=self.db,
                prompt=custom_prompt,
                verbose=self.verbose,
                agent_executor_kwargs={"return_intermediate_steps": True},
            )

    def execute_sql_with_error(self, sql: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Execute a safe SQL query and return a tuple of (DataFrame, error_message).

        Enforces read-only safety guardrails before execution.
        """
        if not sql or not sql.strip():
            return None, "Empty SQL query provided."

        is_safe, reason = guardrails.validate_sql(sql, dialect=str(self.dialect) if self.dialect else None)
        if not is_safe:
            logger.warning("Direct SQL execution blocked: %s", reason)
            return None, f"Query blocked by safety guardrails: {reason}"

        try:
            # pyrefly: ignore [missing-import]
            from sqlalchemy import text

            engine = getattr(self.db, "_engine", None)
            if engine is None:
                return None, "Database engine unavailable."

            with engine.connect() as conn:
                df = pd.read_sql_query(text(sql), conn)
                return df, None

        except Exception as e:
            error_str = str(e)
            logger.warning("SQL execution failed: %s", error_str)
            return None, error_str

    def execute_sql(self, sql: str) -> Optional[pd.DataFrame]:
        """
        Execute a safe SQL query directly against the database engine and return a DataFrame.
        """
        df, _ = self.execute_sql_with_error(sql)
        return df

    def self_heal_sql(
        self,
        failed_sql: str,
        error_msg: str,
        user_question: str,
    ) -> Optional[str]:
        """
        Autonomous Error Reflection (Phase 4b):
        Diagnose a failed query + database error and generate a repaired, safe SQL query.
        """
        schema_summary = self.get_schema()
        if len(schema_summary) > 2500:
            schema_summary = schema_summary[:2500] + "\n... [schema truncated]"

        reflection_prompt = (
            f"You are an expert SQL engineer debugging a query execution failure.\n"
            f"Target Database Dialect: {self.dialect}\n\n"
            f"User Question: {user_question}\n"
            f"Failed SQL Query:\n```sql\n{failed_sql}\n```\n"
            f"Database Error Message:\n{error_msg}\n\n"
            f"Database Schema:\n{schema_summary}\n\n"
            f"Task:\n"
            f"1. Analyze the exact error (e.g. invalid table/column name, date function syntax, missing brackets, or row limit keyword).\n"
            f"2. Write a corrected, syntactically valid SELECT query for {self.dialect}.\n"
            f"3. Return ONLY the raw SQL query string with no markdown formatting, commentary, or conversational text."
        )

        try:
            res = self.llm.invoke(reflection_prompt)
            raw_text = res.content if hasattr(res, "content") else str(res)
            cleaned = raw_text.strip()
            # Strip markdown fences if present
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if len(lines) >= 2 and lines[0].startswith("```"):
                    cleaned = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
            cleaned = cleaned.strip()
            return cleaned if cleaned else None
        except Exception as e:
            logger.warning("Self-healing LLM reflection failed: %s", e)
            return None

    def explain_sql(self, sql: str) -> str:
        """
        Generate a structured, educational plain-English breakdown of an executed SQL query (Phase 4b).
        """
        if not sql or not sql.strip():
            return "No SQL query available to explain."

        try:
            prompt = (
                f"Explain this {self.dialect} SQL query in clean, structured bullet points for a business user:\n"
                f"```sql\n{sql}\n```\n\n"
                f"Provide concise points covering:\n"
                f"- 📊 **Target Tables & Columns**\n"
                f"- 🔍 **Filtering Conditions (WHERE / HAVING)**\n"
                f"- 🧮 **Calculations & Aggregations (GROUP BY / Functions)**\n"
                f"- ⚡ **Sorting & Limits (ORDER BY / TOP / LIMIT)**\n"
                f"Keep explanation under 120 words total."
            )
            res = self.llm.invoke(prompt)
            return res.content if hasattr(res, "content") else str(res)
        except Exception as e:
            logger.warning("SQL explanation generation failed: %s", e)
            return "Detailed SQL explanation unavailable."

    def prune_schema_tables(self, user_input: str) -> list[str]:
        """
        Intelligent Schema Pruner (Phase 4b):
        Identify relevant tables for the user's intent to avoid massive prompt token bloat.
        """
        try:
            all_tables = self.db.get_usable_table_names()
            if len(all_tables) <= 4 or not self.enable_schema_pruning:
                return all_tables

            input_lower = user_input.lower()
            tokens = set(input_lower.replace(",", " ").replace(".", " ").replace("_", " ").split())

            relevant = []
            for tbl in all_tables:
                tbl_lower = tbl.lower()
                tbl_tokens = set(tbl_lower.replace("_", " ").split())
                if tbl_lower in input_lower or tbl_tokens & tokens:
                    relevant.append(tbl)

            # If heuristic found matching tables, return them; otherwise keep top 5
            return relevant if relevant else all_tables[:5]
        except Exception:
            return []

    def chat(self, user_input: str) -> dict:
        """
        Send a message to the agent and get a response.

        Automatically includes conversation history for multi-turn context,
        performs self-healing retry reflection on execution errors,
        and generates structured query explanations.

        Args:
            user_input: The user's natural language question.

        Returns:
            A dict with keys:
                - 'output': The agent's text response.
                - 'sql': List of SQL queries generated.
                - 'data': Dict with 'columns' and 'rows' (or None).
                - 'healed': Boolean indicating if query required self-healing.
                - 'attempts': Total execution attempts made.
                - 'explanation': Educational breakdown of the executed SQL.
        """
        # Build context from conversation history
        history_context = self._build_history_context()

        # Retrieve relevant few-shot exemplars (Phase 4c)
        fewshot_block = self.fewshot_store.format_examples_for_prompt(user_input, dialect=str(self.dialect), top_k=2)

        # Construct the full prompt with history and few-shot exemplars
        prompt_sections = []
        if fewshot_block:
            prompt_sections.append(fewshot_block)
        if history_context:
            prompt_sections.append(f"Previous conversation context:\n{history_context}")
        prompt_sections.append(f"Current user question: {user_input}")

        full_input = "\n\n".join(prompt_sections)

        healed = False
        attempts = 1
        explanation = ""
        cost = "LOW"
        profiling_tips: list[str] = []
        profiling_warnings: list[str] = []

        try:
            response = self._agent.invoke({"input": full_input})
            raw_output = response.get("output", "No response generated.")

            # Unpack structured message blocks from Gemini tool calling if returned as list/dict
            if isinstance(raw_output, list):
                text_parts = []
                for item in raw_output:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                    elif isinstance(item, str):
                        text_parts.append(item)
                    else:
                        text_parts.append(str(item))
                output = "\n".join(text_parts) if text_parts else str(raw_output)
            elif isinstance(raw_output, dict) and "text" in raw_output:
                output = raw_output["text"]
            else:
                output = str(raw_output)

            # Extract SQL queries executed during intermediate steps
            sql_queries = self._extract_sql(response)

            # Direct DataFrame capture and Self-Healing Validation
            structured_data = None
            if sql_queries:
                self.last_sql = sql_queries[-1]
                df, err_msg = self.execute_sql_with_error(self.last_sql)

                # Self-Healing Reflection Loop (Phase 4b)
                if err_msg and self.enable_self_healing and self.max_retries > 1:
                    logger.info("Initiating Self-Healing Reflection Loop for query: %s (Error: %s)", self.last_sql, err_msg)
                    current_sql = self.last_sql
                    current_err = err_msg

                    for attempt_idx in range(1, self.max_retries):
                        attempts += 1
                        healed_sql = self.self_heal_sql(current_sql, current_err, user_input)
                        if healed_sql:
                            df_healed, heal_err = self.execute_sql_with_error(healed_sql)
                            if df_healed is not None and not df_healed.empty:
                                df = df_healed
                                self.last_sql = healed_sql
                                if healed_sql not in sql_queries:
                                    sql_queries.append(healed_sql)
                                healed = True
                                logger.info("Query successfully self-healed on attempt %d: %s", attempts, healed_sql)
                                break
                            else:
                                current_sql = healed_sql
                                current_err = heal_err or "Empty result or syntax error"

                self.last_df = df
                if self.last_df is not None and not self.last_df.empty:
                    from .visualization import dataframe_to_dict
                    structured_data = dataframe_to_dict(self.last_df)

                # Generate structured explanation and performance profile (Phase 4c)
                if self.last_sql:
                    explanation = self.explain_sql(self.last_sql)
                    report = self.profiler.profile(self.last_sql)
                    cost = report.cost_rating
                    profiling_tips = report.tips
                    profiling_warnings = report.warnings
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
                "healed": healed,
                "attempts": attempts,
                "explanation": explanation,
                "cost": cost,
                "profiling_tips": profiling_tips,
                "profiling_warnings": profiling_warnings,
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
                "healed": False,
                "attempts": 1,
                "explanation": "",
                "cost": "LOW",
                "profiling_tips": [],
                "profiling_warnings": [],
            }

    def add_golden_example(self, question: str, sql: str, category: str = "general") -> None:
        """
        Interactive Exemplar Learning (Phase 4c):
        Save a verified, high-quality (question, SQL) pair to the few-shot store.
        """
        self.fewshot_store.add_example(
            question=question,
            sql=sql,
            dialect=str(self.dialect),
            category=category,
        )
        logger.info("Saved golden exemplar to store: '%s'", question)

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
