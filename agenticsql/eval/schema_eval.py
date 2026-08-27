"""
Schema Linking and Entity Extraction Evaluator.

Evaluates Stage 1 (Schema Linking / RAG) of the Text-to-SQL pipeline:
- Calculates Table Precision, Recall, and F1
- Calculates Column Precision, Recall, and F1
- Diagnoses missed entity links and hallucinated schema elements
"""

import logging
from typing import Optional
from dataclasses import dataclass, field
import sqlglot
from sqlglot import exp

from .dataset import SchemaLinkingTarget
from .ast_matcher import SQLASTMatcher

logger = logging.getLogger(__name__)


@dataclass
class SchemaMetricResult:
    """Precision, Recall, and F1 for a specific entity type (tables or columns)."""
    precision: float
    recall: float
    f1: float
    true_positives: list[str] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)  # Hallucinated or extraneous entities
    false_negatives: list[str] = field(default_factory=list)  # Missed ground-truth entities


@dataclass
class SchemaLinkingResult:
    """Overall schema linking evaluation result."""
    overall_f1: float
    tables_metric: SchemaMetricResult
    columns_metric: SchemaMetricResult
    predicted_tables: list[str] = field(default_factory=list)
    predicted_columns: list[str] = field(default_factory=list)


class SchemaLinkingEvaluator:
    """
    Evaluator for checking accuracy of schema linking from user intent.
    """

    def __init__(self, ast_matcher: Optional[SQLASTMatcher] = None):
        self.ast_matcher = ast_matcher or SQLASTMatcher()

    def evaluate_from_sql(
        self,
        generated_sql: str,
        ground_truth: SchemaLinkingTarget,
        dialect: Optional[str] = None,
    ) -> SchemaLinkingResult:
        """
        Extract schema entities from generated SQL and evaluate against ground truth.
        """
        ast, err = self.ast_matcher.parse_sql(generated_sql, dialect=dialect)
        if ast is None:
            # If SQL is unparseable, schema linking score is 0
            empty_metric = SchemaMetricResult(precision=0.0, recall=0.0, f1=0.0, false_negatives=ground_truth.tables)
            return SchemaLinkingResult(
                overall_f1=0.0,
                tables_metric=empty_metric,
                columns_metric=SchemaMetricResult(precision=0.0, recall=0.0, f1=0.0, false_negatives=ground_truth.columns),
            )

        pred_tables, pred_cols = self.ast_matcher.extract_schema_entities(ast)
        return self.evaluate_entities(pred_tables, pred_cols, ground_truth)

    def evaluate_entities(
        self,
        predicted_tables: list[str],
        predicted_columns: list[str],
        ground_truth: SchemaLinkingTarget,
    ) -> SchemaLinkingResult:
        """
        Calculate precision, recall, and F1 for predicted schema entities.
        """
        # Normalize to lowercase stripped strings
        p_tables = {t.lower().strip() for t in predicted_tables if t}
        g_tables = {t.lower().strip() for t in ground_truth.tables if t}

        # Normalize columns: handle qualified 'table.col' and unqualified 'col'
        p_cols = {c.lower().strip() for c in predicted_columns if c}
        g_cols = {c.lower().strip() for c in ground_truth.columns if c}

        # Evaluate tables
        t_metric = self._compute_prf1(p_tables, g_tables)

        # Evaluate columns (allow matching unqualified if unambiguous)
        c_metric = self._compute_column_prf1(p_cols, g_cols)

        # Overall composite F1 score
        if len(g_cols) > 0 and len(g_tables) > 0:
            overall = 0.4 * t_metric.f1 + 0.6 * c_metric.f1
        elif len(g_tables) > 0:
            overall = t_metric.f1
        else:
            overall = 1.0

        return SchemaLinkingResult(
            overall_f1=round(overall, 4),
            tables_metric=t_metric,
            columns_metric=c_metric,
            predicted_tables=sorted(list(p_tables)),
            predicted_columns=sorted(list(p_cols)),
        )

    def _compute_prf1(self, predicted: set[str], ground_truth: set[str]) -> SchemaMetricResult:
        """Compute precision, recall, and F1 for two sets of strings."""
        if not ground_truth and not predicted:
            return SchemaMetricResult(precision=1.0, recall=1.0, f1=1.0)

        tp = predicted.intersection(ground_truth)
        fp = predicted.difference(ground_truth)
        fn = ground_truth.difference(predicted)

        precision = len(tp) / len(predicted) if predicted else 0.0
        recall = len(tp) / len(ground_truth) if ground_truth else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return SchemaMetricResult(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            true_positives=sorted(list(tp)),
            false_positives=sorted(list(fp)),
            false_negatives=sorted(list(fn)),
        )

    def _compute_column_prf1(self, predicted: set[str], ground_truth: set[str]) -> SchemaMetricResult:
        """Compute column metric taking into account table-qualification aliases."""
        if not ground_truth and not predicted:
            return SchemaMetricResult(precision=1.0, recall=1.0, f1=1.0)

        # Helper to extract basename of column
        def col_base(name: str) -> str:
            return name.split(".")[-1].strip()

        tp = set()
        fp = set()
        matched_gt = set()

        for p_col in predicted:
            p_base = col_base(p_col)
            p_table = p_col.split(".")[0] if "." in p_col else None
            found_match = False
            for g_col in ground_truth:
                g_base = col_base(g_col)
                g_table = g_col.split(".")[0] if "." in g_col else None
                if p_col == g_col or (p_base == g_base and (p_table == g_table or not p_table or not g_table)):
                    tp.add(p_col)
                    matched_gt.add(g_col)
                    found_match = True
                    break
            if not found_match:
                fp.add(p_col)

        fn = ground_truth.difference(matched_gt)

        precision = len(tp) / len(predicted) if predicted else 0.0
        recall = len(matched_gt) / len(ground_truth) if ground_truth else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return SchemaMetricResult(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            true_positives=sorted(list(tp)),
            false_positives=sorted(list(fp)),
            false_negatives=sorted(list(fn)),
        )
