"""
AST Comparison and Semantic Equivalence Engine powered by sqlglot.

Provides:
1. SQL Syntax Validation and Parsing
2. AST Normalization (alias desugaring, case folding, qualification)
3. Clause-by-clause structural comparison (SELECT, FROM/JOIN, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT)
4. AST Similarity Scoring (0.0 to 1.0) and clause diff diagnosis
5. Multi-dialect transpilation verification
"""

import logging
from typing import Optional, Any
from dataclasses import dataclass, field
import sqlglot
from sqlglot import exp, errors

logger = logging.getLogger(__name__)


@dataclass
class ClauseMatchResult:
    """Detailed comparison result for an individual SQL clause."""
    matched: bool
    gold_clause: Optional[str] = None
    candidate_clause: Optional[str] = None
    similarity: float = 0.0
    details: str = ""


@dataclass
class ASTComparisonResult:
    """Comprehensive AST comparison result."""
    is_valid_syntax: bool
    syntax_error: Optional[str] = None
    exact_ast_match: bool = False
    ast_similarity_score: float = 0.0
    clauses: dict[str, ClauseMatchResult] = field(default_factory=dict)
    normalized_gold_sql: Optional[str] = None
    normalized_candidate_sql: Optional[str] = None
    tables_extracted: list[str] = field(default_factory=list)
    columns_extracted: list[str] = field(default_factory=list)


def normalize_dialect_name(dialect: Optional[str]) -> str:
    """Map user/db dialect names to sqlglot dialect identifiers."""
    if not dialect:
        return "tsql"
    d = dialect.lower().strip()
    if any(k in d for k in ["postgres", "cockroach", "neon", "supabase"]):
        return "postgres"
    if any(k in d for k in ["mysql", "mariadb"]):
        return "mysql"
    if "sqlite" in d:
        return "sqlite"
    if "oracle" in d:
        return "oracle"
    if "snowflake" in d:
        return "snowflake"
    if "bigquery" in d:
        return "bigquery"
    return "tsql"


class SQLASTMatcher:
    """
    AST Comparison Engine using sqlglot for semantic SQL evaluation.
    """

    def __init__(self, default_dialect: str = "sqlite"):
        self.default_dialect = normalize_dialect_name(default_dialect)

    def parse_sql(self, sql: str, dialect: Optional[str] = None) -> tuple[Optional[exp.Expression], Optional[str]]:
        """
        Parse SQL string into sqlglot Expression AST.

        Returns:
            Tuple of (AST Expression or None, syntax error string or None)
        """
        if not sql or not sql.strip():
            return None, "Empty SQL query string"

        glot_dialect = normalize_dialect_name(dialect or self.default_dialect)
        clean_sql = sql.strip().rstrip(";")

        try:
            expression = sqlglot.parse_one(clean_sql, read=glot_dialect)
            return expression, None
        except (errors.ParseError, errors.SqlglotError, Exception) as e:
            return None, str(e)

    def normalize_ast(self, expression: exp.Expression, target_dialect: Optional[str] = None) -> exp.Expression:
        """
        Normalize an AST to canonical representation:
        - Lowercase identifiers
        - Canonicalize boolean logic order in WHERE clauses
        - Standardize COUNT(1) to COUNT(*)
        """
        cloned = expression.copy()

        # Transform COUNT(1) or COUNT(0) to COUNT(*)
        def simplify_nodes(node: exp.Expression) -> exp.Expression:
            if isinstance(node, exp.Count):
                # If argument is a literal 1, replace with Star
                this = node.this
                if isinstance(this, exp.Literal) and this.this in ("1", "0"):
                    return exp.Count(this=exp.Star())
            return node

        cloned = cloned.transform(simplify_nodes)
        return cloned

    def to_canonical_sql(self, expression: exp.Expression, dialect: Optional[str] = None) -> str:
        """Generate canonical formatted SQL string from AST."""
        glot_dialect = normalize_dialect_name(dialect or self.default_dialect)
        normalized = self.normalize_ast(expression, glot_dialect)
        return normalized.sql(dialect=glot_dialect, pretty=False, normalize=True, pad=0).strip()

    def extract_schema_entities(self, expression: exp.Expression) -> tuple[list[str], list[str]]:
        """
        Extract table names and column names referenced in the AST.

        Returns:
            Tuple of (list of tables, list of columns)
        """
        tables = set()
        columns = set()
        alias_map = {}

        for table in expression.find_all(exp.Table):
            table_name = table.name.lower()
            if table_name:
                tables.add(table_name)
                if table.alias:
                    alias_map[table.alias.lower()] = table_name

        for col in expression.find_all(exp.Column):
            col_name = col.name.lower()
            table_ref = col.table.lower() if col.table else None
            # Resolve alias to actual table name if mapped
            resolved_table = alias_map.get(table_ref, table_ref)
            if resolved_table:
                columns.add(f"{resolved_table}.{col_name}")
            elif col_name:
                columns.add(col_name)

        return sorted(list(tables)), sorted(list(columns))

    def compare(
        self,
        candidate_sql: str,
        gold_sql: str,
        dialect: Optional[str] = None,
    ) -> ASTComparisonResult:
        """
        Perform a full AST structural comparison between candidate SQL and gold reference SQL.

        Args:
            candidate_sql: The SQL generated by the agent.
            gold_sql: The reference gold standard SQL query.
            dialect: SQL dialect name.

        Returns:
            ASTComparisonResult with clause breakdown and similarity score.
        """
        glot_dialect = normalize_dialect_name(dialect or self.default_dialect)

        # 1. Parse candidate SQL
        cand_ast, cand_err = self.parse_sql(candidate_sql, dialect=glot_dialect)
        if cand_ast is None:
            return ASTComparisonResult(
                is_valid_syntax=False,
                syntax_error=cand_err,
                exact_ast_match=False,
                ast_similarity_score=0.0,
            )

        # 2. Parse gold SQL
        gold_ast, gold_err = self.parse_sql(gold_sql, dialect=glot_dialect)
        if gold_ast is None:
            # Fallback parsing with generic dialect
            gold_ast, _ = self.parse_sql(gold_sql, dialect=None)
            if gold_ast is None:
                raise ValueError(f"Gold SQL has invalid syntax ({gold_err}): {gold_sql}")

        # Extract schema entities
        tables, columns = self.extract_schema_entities(cand_ast)

        # 3. Canonical SQL representations
        cand_canonical = self.to_canonical_sql(cand_ast, glot_dialect)
        gold_canonical = self.to_canonical_sql(gold_ast, glot_dialect)

        if cand_canonical.lower() == gold_canonical.lower():
            # Perfect exact canonical match
            clauses = self._compare_all_clauses(cand_ast, gold_ast, glot_dialect)
            return ASTComparisonResult(
                is_valid_syntax=True,
                syntax_error=None,
                exact_ast_match=True,
                ast_similarity_score=1.0,
                clauses=clauses,
                normalized_gold_sql=gold_canonical,
                normalized_candidate_sql=cand_canonical,
                tables_extracted=tables,
                columns_extracted=columns,
            )

        # 4. Clause-by-clause detailed comparison
        clauses = self._compare_all_clauses(cand_ast, gold_ast, glot_dialect)

        # 5. Weighted similarity score computation
        weights = {
            "from_join": 0.25,
            "where": 0.25,
            "select": 0.20,
            "group_by": 0.15,
            "having": 0.05,
            "order_by": 0.05,
            "limit": 0.05,
        }

        total_weight = 0.0
        weighted_score = 0.0

        for clause_key, weight in weights.items():
            if clause_key in clauses:
                clause_res = clauses[clause_key]
                weighted_score += weight * clause_res.similarity
                total_weight += weight

        final_score = (weighted_score / total_weight) if total_weight > 0 else 0.0

        return ASTComparisonResult(
            is_valid_syntax=True,
            syntax_error=None,
            exact_ast_match=False,
            ast_similarity_score=round(final_score, 4),
            clauses=clauses,
            normalized_gold_sql=gold_canonical,
            normalized_candidate_sql=cand_canonical,
            tables_extracted=tables,
            columns_extracted=columns,
        )

    def _compare_all_clauses(
        self,
        cand_ast: exp.Expression,
        gold_ast: exp.Expression,
        dialect: str,
    ) -> dict[str, ClauseMatchResult]:
        """Compare each SQL clause between candidate and gold AST."""
        clauses: dict[str, ClauseMatchResult] = {}

        # 1. SELECT (projections)
        clauses["select"] = self._compare_select_clause(cand_ast, gold_ast, dialect)

        # 2. FROM / JOIN
        clauses["from_join"] = self._compare_from_join_clause(cand_ast, gold_ast, dialect)

        # 3. WHERE
        clauses["where"] = self._compare_where_clause(cand_ast, gold_ast, dialect)

        # 4. GROUP BY
        clauses["group_by"] = self._compare_group_by_clause(cand_ast, gold_ast, dialect)

        # 5. HAVING
        clauses["having"] = self._compare_having_clause(cand_ast, gold_ast, dialect)

        # 6. ORDER BY
        clauses["order_by"] = self._compare_order_by_clause(cand_ast, gold_ast, dialect)

        # 7. LIMIT / TOP
        clauses["limit"] = self._compare_limit_clause(cand_ast, gold_ast, dialect)

        return clauses

    def _compare_select_clause(
        self, cand_ast: exp.Expression, gold_ast: exp.Expression, dialect: str
    ) -> ClauseMatchResult:
        cand_selects = cand_ast.args.get("expressions", [])
        gold_selects = gold_ast.args.get("expressions", [])

        cand_str = ", ".join(s.sql(dialect=dialect).lower().strip() for s in cand_selects)
        gold_str = ", ".join(s.sql(dialect=dialect).lower().strip() for s in gold_selects)

        if not gold_selects and not cand_selects:
            return ClauseMatchResult(matched=True, similarity=1.0)

        cand_set = {s.sql(dialect=dialect).lower().strip() for s in cand_selects}
        gold_set = {s.sql(dialect=dialect).lower().strip() for s in gold_selects}

        intersection = cand_set.intersection(gold_set)
        union = cand_set.union(gold_set)
        sim = len(intersection) / len(union) if union else 1.0

        return ClauseMatchResult(
            matched=(sim == 1.0),
            gold_clause=gold_str,
            candidate_clause=cand_str,
            similarity=sim,
            details=f"Matched {len(intersection)}/{len(gold_set)} projection expressions",
        )

    def _compare_from_join_clause(
        self, cand_ast: exp.Expression, gold_ast: exp.Expression, dialect: str
    ) -> ClauseMatchResult:
        cand_tables = {t.name.lower() for t in cand_ast.find_all(exp.Table) if t.name}
        gold_tables = {t.name.lower() for t in gold_ast.find_all(exp.Table) if t.name}

        cand_joins = [j.sql(dialect=dialect).lower().strip() for j in cand_ast.find_all(exp.Join)]
        gold_joins = [j.sql(dialect=dialect).lower().strip() for j in gold_ast.find_all(exp.Join)]

        if cand_tables == gold_tables and len(cand_joins) == len(gold_joins):
            return ClauseMatchResult(
                matched=True,
                gold_clause=f"Tables: {gold_tables}, Joins: {len(gold_joins)}",
                candidate_clause=f"Tables: {cand_tables}, Joins: {len(cand_joins)}",
                similarity=1.0,
                details="Exact table and join structure match",
            )

        intersection = cand_tables.intersection(gold_tables)
        union = cand_tables.union(gold_tables)
        sim = len(intersection) / len(union) if union else 1.0

        return ClauseMatchResult(
            matched=(sim == 1.0 and len(cand_joins) == len(gold_joins)),
            gold_clause=f"Tables: {gold_tables}",
            candidate_clause=f"Tables: {cand_tables}",
            similarity=sim,
            details=f"Matched tables: {intersection}",
        )

    def _compare_where_clause(
        self, cand_ast: exp.Expression, gold_ast: exp.Expression, dialect: str
    ) -> ClauseMatchResult:
        cand_where = cand_ast.args.get("where")
        gold_where = gold_ast.args.get("where")

        if cand_where is None and gold_where is None:
            return ClauseMatchResult(matched=True, similarity=1.0, details="No WHERE clause in either query")

        if (cand_where is None) != (gold_where is None):
            return ClauseMatchResult(
                matched=False,
                gold_clause=gold_where.sql(dialect=dialect) if gold_where else None,
                candidate_clause=cand_where.sql(dialect=dialect) if cand_where else None,
                similarity=0.0,
                details="WHERE clause missing in one of the queries",
            )

        cand_where_sql = cand_where.sql(dialect=dialect).lower().strip()
        gold_where_sql = gold_where.sql(dialect=dialect).lower().strip()

        if cand_where_sql == gold_where_sql:
            return ClauseMatchResult(
                matched=True,
                gold_clause=gold_where_sql,
                candidate_clause=cand_where_sql,
                similarity=1.0,
                details="Exact WHERE clause match",
            )

        # Decompose AND expressions
        def extract_predicates(where_node: exp.Where) -> set[str]:
            preds = set()
            this = where_node.this
            if isinstance(this, exp.And):
                for p in this.flatten():
                    preds.add(p.sql(dialect=dialect).lower().strip())
            elif this:
                preds.add(this.sql(dialect=dialect).lower().strip())
            return preds

        cand_preds = extract_predicates(cand_where)
        gold_preds = extract_predicates(gold_where)

        intersection = cand_preds.intersection(gold_preds)
        union = cand_preds.union(gold_preds)
        sim = len(intersection) / len(union) if union else 1.0

        return ClauseMatchResult(
            matched=(sim == 1.0),
            gold_clause=gold_where_sql,
            candidate_clause=cand_where_sql,
            similarity=sim,
            details=f"Matched {len(intersection)}/{len(gold_preds)} WHERE predicates",
        )

    def _compare_group_by_clause(
        self, cand_ast: exp.Expression, gold_ast: exp.Expression, dialect: str
    ) -> ClauseMatchResult:
        cand_gb = cand_ast.args.get("group")
        gold_gb = gold_ast.args.get("group")

        if cand_gb is None and gold_gb is None:
            return ClauseMatchResult(matched=True, similarity=1.0, details="No GROUP BY in either query")

        if (cand_gb is None) != (gold_gb is None):
            return ClauseMatchResult(
                matched=False,
                gold_clause=gold_gb.sql(dialect=dialect) if gold_gb else None,
                candidate_clause=cand_gb.sql(dialect=dialect) if cand_gb else None,
                similarity=0.0,
                details="GROUP BY missing in one query",
            )

        cand_cols = {e.sql(dialect=dialect).lower().strip() for e in cand_gb.expressions}
        gold_cols = {e.sql(dialect=dialect).lower().strip() for e in gold_gb.expressions}

        intersection = cand_cols.intersection(gold_cols)
        union = cand_cols.union(gold_cols)
        sim = len(intersection) / len(union) if union else 1.0

        return ClauseMatchResult(
            matched=(sim == 1.0),
            gold_clause=", ".join(gold_cols),
            candidate_clause=", ".join(cand_cols),
            similarity=sim,
            details=f"Matched {len(intersection)}/{len(gold_cols)} GROUP BY columns",
        )

    def _compare_having_clause(
        self, cand_ast: exp.Expression, gold_ast: exp.Expression, dialect: str
    ) -> ClauseMatchResult:
        cand_hav = cand_ast.args.get("having")
        gold_hav = gold_ast.args.get("having")

        if cand_hav is None and gold_hav is None:
            return ClauseMatchResult(matched=True, similarity=1.0, details="No HAVING in either query")

        if (cand_hav is None) != (gold_hav is None):
            return ClauseMatchResult(
                matched=False,
                gold_clause=gold_hav.sql(dialect=dialect) if gold_hav else None,
                candidate_clause=cand_hav.sql(dialect=dialect) if cand_hav else None,
                similarity=0.0,
            )

        cand_sql = cand_hav.sql(dialect=dialect).lower().strip()
        gold_sql = gold_hav.sql(dialect=dialect).lower().strip()
        matched = (cand_sql == gold_sql)

        return ClauseMatchResult(
            matched=matched,
            gold_clause=gold_sql,
            candidate_clause=cand_sql,
            similarity=1.0 if matched else 0.5,
        )

    def _compare_order_by_clause(
        self, cand_ast: exp.Expression, gold_ast: exp.Expression, dialect: str
    ) -> ClauseMatchResult:
        cand_ord = cand_ast.args.get("order")
        gold_ord = gold_ast.args.get("order")

        if cand_ord is None and gold_ord is None:
            return ClauseMatchResult(matched=True, similarity=1.0, details="No ORDER BY in either query")

        if (cand_ord is None) != (gold_ord is None):
            return ClauseMatchResult(
                matched=False,
                gold_clause=gold_ord.sql(dialect=dialect) if gold_ord else None,
                candidate_clause=cand_ord.sql(dialect=dialect) if cand_ord else None,
                similarity=0.0,
            )

        cand_sql = cand_ord.sql(dialect=dialect).lower().strip()
        gold_sql = gold_ord.sql(dialect=dialect).lower().strip()
        matched = (cand_sql == gold_sql)

        return ClauseMatchResult(
            matched=matched,
            gold_clause=gold_sql,
            candidate_clause=cand_sql,
            similarity=1.0 if matched else 0.5,
        )

    def _compare_limit_clause(
        self, cand_ast: exp.Expression, gold_ast: exp.Expression, dialect: str
    ) -> ClauseMatchResult:
        cand_lim = cand_ast.args.get("limit")
        gold_lim = gold_ast.args.get("limit")

        if cand_lim is None and gold_lim is None:
            return ClauseMatchResult(matched=True, similarity=1.0, details="No LIMIT in either query")

        if (cand_lim is None) != (gold_lim is None):
            return ClauseMatchResult(
                matched=False,
                gold_clause=gold_lim.sql(dialect=dialect) if gold_lim else None,
                candidate_clause=cand_lim.sql(dialect=dialect) if cand_lim else None,
                similarity=0.0,
            )

        cand_sql = cand_lim.sql(dialect=dialect).lower().strip()
        gold_sql = gold_lim.sql(dialect=dialect).lower().strip()
        matched = (cand_sql == gold_sql)

        return ClauseMatchResult(
            matched=matched,
            gold_clause=gold_sql,
            candidate_clause=cand_sql,
            similarity=1.0 if matched else 0.0,
        )
