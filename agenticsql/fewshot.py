"""
Few-Shot Exemplar Store and Semantic Retrieval for AgenticSQL (Phase 4c).

Manages a repository of curated golden (Question, SQL, Dialect) pairs.
Retrieves the most relevant exemplars based on keyword/semantic similarity
and dialect compatibility to provide high-accuracy context to the agent LLM.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_FEWSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "fewshot_bank.json"


@dataclass
class FewShotExample:
    """A curated natural language question to SQL exemplar."""
    question: str
    sql: str
    dialect: str = "mssql"
    category: str = "general"
    tables: list[str] = None

    def __post_init__(self):
        if self.tables is None:
            self.tables = []


# Seed golden exemplar bank covering various SQL dialect nuances
SEED_EXEMPLARS = [
    # MSSQL / T-SQL
    FewShotExample(
        question="Show top 5 customers with highest total order amounts",
        sql="SELECT TOP (5) c.customer_id, c.name, SUM(o.total) AS total_spent FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id, c.name ORDER BY total_spent DESC;",
        dialect="mssql",
        category="aggregation",
        tables=["customers", "orders"],
    ),
    FewShotExample(
        question="Find transactions made in the last 30 days",
        sql="SELECT transaction_id, amount, transaction_date FROM transactions WHERE transaction_date >= DATEADD(day, -30, GETDATE()) ORDER BY transaction_date DESC;",
        dialect="mssql",
        category="time_series",
        tables=["transactions"],
    ),
    FewShotExample(
        question="List products with inventory lower than 10 units",
        sql="SELECT product_id, product_name, stock_quantity FROM [products] WHERE stock_quantity < 10 ORDER BY stock_quantity ASC;",
        dialect="mssql",
        category="filtering",
        tables=["products"],
    ),
    FewShotExample(
        question="Find average stock closing price per month for Citigroup",
        sql="SELECT YEAR([Date]) AS yr, MONTH([Date]) AS mo, AVG([Close]) AS avg_close, SUM([Volume]) AS total_vol FROM [Citigroup_historical_data] GROUP BY YEAR([Date]), MONTH([Date]) ORDER BY yr DESC, mo DESC;",
        dialect="mssql",
        category="aggregation",
        tables=["Citigroup_historical_data"],
    ),
    # PostgreSQL
    FewShotExample(
        question="Show top 5 customers with highest total order amounts",
        sql='SELECT c.customer_id, c.name, SUM(o.total) AS total_spent FROM "customers" c JOIN "orders" o ON c.customer_id = o.customer_id GROUP BY c.customer_id, c.name ORDER BY total_spent DESC LIMIT 5;',
        dialect="postgresql",
        category="aggregation",
        tables=["customers", "orders"],
    ),
    FewShotExample(
        question="Find transactions made in the last 30 days",
        sql="SELECT transaction_id, amount, transaction_date FROM transactions WHERE transaction_date >= NOW() - INTERVAL '30 days' ORDER BY transaction_date DESC;",
        dialect="postgresql",
        category="time_series",
        tables=["transactions"],
    ),
    FewShotExample(
        question="Case-insensitive search for users with email domain gmail.com",
        sql="SELECT id, username, email FROM users WHERE email ILIKE '%@gmail.com' LIMIT 50;",
        dialect="postgresql",
        category="pattern_matching",
        tables=["users"],
    ),
    # MySQL
    FewShotExample(
        question="Show monthly revenue summary for completed orders",
        sql="SELECT DATE_FORMAT(order_date, '%Y-%m') AS order_month, COUNT(order_id) AS order_count, SUM(total) AS total_revenue FROM `orders` WHERE `status` = 'COMPLETED' GROUP BY DATE_FORMAT(order_date, '%Y-%m') ORDER BY order_month DESC;",
        dialect="mysql",
        category="aggregation",
        tables=["orders"],
    ),
    # SQLite
    FewShotExample(
        question="Count active users registered this year",
        sql="SELECT COUNT(*) AS active_user_count FROM users WHERE is_active = 1 AND strftime('%Y', created_at) = strftime('%Y', 'now');",
        dialect="sqlite",
        category="filtering",
        tables=["users"],
    ),
]


class FewShotStore:
    """
    In-memory and file-backed Few-Shot Exemplar Store with semantic & keyword retrieval.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or DEFAULT_FEWSHOT_PATH
        self.examples: list[FewShotExample] = []
        self._load()

    def _load(self) -> None:
        """Load exemplars from disk or initialize with seed bank."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.examples = [FewShotExample(**item) for item in data]
                    logger.info("Loaded %d few-shot exemplars from %s", len(self.examples), self.storage_path)
                    return
            except Exception as e:
                logger.warning("Could not read few-shot file %s: %s. Using seed exemplars.", self.storage_path, e)

        # Initialize with seed bank
        self.examples = list(SEED_EXEMPLARS)
        self._save()

    def _save(self) -> None:
        """Persist exemplars to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump([asdict(e) for e in self.examples], f, indent=2)
            logger.info("Saved %d few-shot exemplars to %s", len(self.examples), self.storage_path)
        except Exception as e:
            logger.warning("Failed to save few-shot exemplars to disk: %s", e)

    def add_example(
        self,
        question: str,
        sql: str,
        dialect: str = "mssql",
        category: str = "general",
        tables: Optional[list[str]] = None,
    ) -> FewShotExample:
        """Add a new golden exemplar and persist to disk."""
        example = FewShotExample(
            question=question.strip(),
            sql=sql.strip(),
            dialect=dialect.lower().strip(),
            category=category,
            tables=tables or [],
        )
        # Avoid exact duplicates
        for existing in self.examples:
            if existing.question.lower() == example.question.lower() and existing.dialect == example.dialect:
                existing.sql = example.sql
                self._save()
                return existing

        self.examples.append(example)
        self._save()
        return example

    def retrieve_relevant(
        self,
        user_query: str,
        dialect: Optional[str] = "mssql",
        top_k: int = 3,
    ) -> list[FewShotExample]:
        """
        Retrieve the top-K most relevant exemplars matching the query and target dialect.
        """
        if not self.examples:
            return []

        clean_dialect = (dialect or "mssql").lower()
        # Normalize dialect family
        if "postgres" in clean_dialect or "cockroach" in clean_dialect:
            target_dialect = "postgresql"
        elif "mysql" in clean_dialect or "mariadb" in clean_dialect:
            target_dialect = "mysql"
        else:
            target_dialect = "mssql"

        candidates = [e for e in self.examples if e.dialect == target_dialect]
        if not candidates:
            candidates = self.examples

        query_tokens = set(user_query.lower().replace("?", "").replace(",", " ").split())

        scored_examples: list[tuple[float, FewShotExample]] = []

        for ex in candidates:
            # Token overlap score
            ex_tokens = set(ex.question.lower().split())
            overlap = len(query_tokens & ex_tokens)
            jaccard = overlap / max(len(query_tokens | ex_tokens), 1)

            # Table name presence in query
            table_bonus = 0.0
            for t in ex.tables:
                if t.lower() in user_query.lower():
                    table_bonus += 0.5

            score = jaccard * 2.0 + table_bonus

            # Prefer dialect matches
            if ex.dialect == target_dialect:
                score += 0.05

            scored_examples.append((score, ex))

        # Sort descending by relevance score
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored_examples[:top_k]]

    def format_examples_for_prompt(
        self,
        user_query: str,
        dialect: Optional[str] = "mssql",
        top_k: int = 2,
    ) -> str:
        """Format retrieved exemplars into a prompt block."""
        relevant = self.retrieve_relevant(user_query, dialect=dialect, top_k=top_k)
        if not relevant:
            return ""

        blocks = ["\n### RELEVANT GOLDEN QUERY EXAMPLES:"]
        for idx, ex in enumerate(relevant, 1):
            blocks.append(
                f"Example #{idx}:\n"
                f"- Question: {ex.question}\n"
                f"- Correct SQL ({ex.dialect}):\n```sql\n{ex.sql}\n```"
            )

        return "\n".join(blocks)
