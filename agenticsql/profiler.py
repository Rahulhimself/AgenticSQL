"""
AST-based Query Performance Profiler and Index Suggester for AgenticSQL (Phase 4c).

Leverages SQLGlot AST inspection to detect performance anti-patterns:
- Unbounded SELECT scans (missing TOP / LIMIT)
- Leading wildcard pattern searches (LIKE '%...') that bypass B-Tree indexes
- Unconstrained Cartesian products (JOIN without ON/USING)
- Heavy `SELECT *` projection on large tables
- Suggests targeted indexing and query optimizations.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

# pyrefly: ignore [missing-import]
import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)


@dataclass
class ProfileReport:
    """
    Detailed performance profiling report for an executed SQL query.
    Contains cost rating (LOW/MEDIUM/HIGH), anti-pattern warnings, and optimization tips.
    """
    sql: str
    cost_rating: str  # "LOW", "MEDIUM", "HIGH"
    warnings: list[str] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)
    complexity_score: int = 1  # 1 to 10 scale


class QueryProfiler:
    """
    AST-based Query Performance Profiler and Index Suggester.
    Detects unbounded table scans, SELECT * usage, wildcard searches, and unconstrained JOINs.
    """

    def __init__(self, dialect: Optional[str] = "mssql"):
        self.dialect = (dialect or "mssql").lower()

    def _normalize_dialect(self) -> str:
        if "postgres" in self.dialect or "cockroach" in self.dialect:
            return "postgres"
        elif "mysql" in self.dialect or "mariadb" in self.dialect:
            return "mysql"
        elif "sqlite" in self.dialect:
            return "sqlite"
        return "tsql"

    def profile(self, sql: str) -> ProfileReport:
        """
        Profile a SQL query and generate performance diagnostics.
        """
        if not sql or not sql.strip():
            return ProfileReport(
                sql="",
                cost_rating="LOW",
                warnings=["Empty SQL query."],
                tips=[],
                complexity_score=0,
            )

        sql_clean = sql.strip().rstrip(";")
        norm_dialect = self._normalize_dialect()

        warnings: list[str] = []
        tips: list[str] = []
        score = 1

        try:
            parsed = sqlglot.parse_one(sql_clean, read=norm_dialect)
        except Exception as e:
            logger.debug("SQLGlot failed to parse for profiling: %s. Using regex heuristics.", e)
            return self._heuristic_profile(sql_clean)

        # 1. Check for Unbounded SELECT (missing LIMIT / TOP / FETCH)
        has_limit = parsed.find(exp.Limit) is not None or parsed.find(exp.Fetch) is not None
        has_top = "TOP" in sql_clean.upper()
        has_aggregate = parsed.find(exp.Count) is not None or parsed.find(exp.Sum) is not None or parsed.find(exp.Avg) is not None

        if not has_limit and not has_top and not has_aggregate:
            warnings.append("Unbounded Query: Result set is not limited by TOP or LIMIT.")
            tips.append("Add `TOP (N)` or `LIMIT N` to prevent accidental high-volume memory buffer consumption.")
            score += 2

        # 2. Check for SELECT * projection
        if parsed.find(exp.Star) is not None:
            warnings.append("`SELECT *` used: Retrieves all table columns including unnecessary wide payloads.")
            tips.append("Specify only the necessary column names instead of `SELECT *` to reduce I/O overhead.")
            score += 1

        # 3. Check for Leading Wildcard LIKE '%pattern'
        for like_expr in parsed.find_all((exp.Like, exp.ILike)):
            pattern_str = ""
            if hasattr(like_expr, "expression") and like_expr.expression is not None:
                pattern_str = getattr(like_expr.expression, "name", "") or str(like_expr.expression)
            pattern_clean = pattern_str.strip("'\"")
            if pattern_clean.startswith("%"):
                warnings.append(f"Leading Wildcard Pattern: `{pattern_clean}` causes full table scan.")
                tips.append("Avoid leading `%` in LIKE conditions if an index is present on the search column.")
                score += 3

        # 4. Check for Cartesian JOINs (JOIN without ON/USING)
        for join in parsed.find_all(exp.Join):
            kind = getattr(join, "kind", "").upper() if hasattr(join, "kind") else ""
            if "CROSS" in kind or (not join.args.get("on") and not join.args.get("using")):
                warnings.append("Cartesian Product: Query contains an unconstrained CROSS JOIN.")
                tips.append("Ensure every JOIN includes explicit `ON` matching foreign/primary key columns.")
                score += 4

        # 5. Check for Multiple JOINs & Nested Subqueries
        joins_count = len(list(parsed.find_all(exp.Join)))
        if joins_count >= 3:
            score += 2
            tips.append(f"Complex Query with {joins_count} JOINs: Verify indexes on all join predicate keys.")

        # Rate Cost Level
        if score <= 2:
            cost_rating = "LOW"
        elif score <= 5:
            cost_rating = "MEDIUM"
        else:
            cost_rating = "HIGH"

        return ProfileReport(
            sql=sql_clean,
            cost_rating=cost_rating,
            warnings=warnings,
            tips=tips,
            complexity_score=min(score, 10),
        )

    def _heuristic_profile(self, sql: str) -> ProfileReport:
        """Fallback regex profiler when AST parsing is incomplete."""
        upper = sql.upper()
        warnings = []
        tips = []
        score = 1

        if "LIMIT" not in upper and "TOP" not in upper and "COUNT(" not in upper:
            warnings.append("Unbounded Query: No LIMIT or TOP detected.")
            tips.append("Add a row limit to guard against large result transfers.")
            score += 2

        if "SELECT *" in upper or "SELECT  *" in upper:
            warnings.append("`SELECT *` detected.")
            tips.append("Specify explicit columns.")
            score += 1

        if "LIKE '%" in upper:
            warnings.append("Leading wildcard pattern in LIKE.")
            tips.append("Leading `%` disables index lookups.")
            score += 3

        cost = "LOW" if score <= 2 else ("MEDIUM" if score <= 5 else "HIGH")
        return ProfileReport(
            sql=sql,
            cost_rating=cost,
            warnings=warnings,
            tips=tips,
            complexity_score=score,
        )
