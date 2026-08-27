"""
Safety and Execution Guardrail Evaluation Harness.

Evaluates Stage 3 (Safety & Execution Guardrails) of the Text-to-SQL pipeline:
- Measures True Positive Rate (TPR) for blocking destructive DDL/DML, injections, and forbidden procs
- Measures False Positive Rate (FPR) on valid, complex SELECT queries
- Verifies audit log capture and violation reason classification
"""

import logging
from typing import Optional
from dataclasses import dataclass, field
from agenticsql.guardrails import validate_sql
from .dataset import SafetyExpectation

logger = logging.getLogger(__name__)


@dataclass
class SafetyEvaluationResult:
    """Detailed evaluation result for safety guardrails."""
    is_safe_verdict: bool  # What the guardrail decided (True = allowed, False = blocked)
    expected_block: bool   # What the test case expected
    correct_verdict: bool  # True if guardrail verdict matches expectation
    reason: str
    is_destructive_detected: bool = False
    is_injection_detected: bool = False
    details: str = ""


@dataclass
class GuardrailBenchmarkMetrics:
    """Aggregated safety benchmark metrics."""
    total_test_cases: int
    malicious_queries_count: int
    safe_queries_count: int
    true_positives: int    # Correctly blocked malicious query
    false_positives: int   # Incorrectly blocked safe query
    true_negatives: int    # Correctly allowed safe query
    false_negatives: int   # Failed to block malicious query (CRITICAL FAILURE)
    tpr: float             # True Positive Rate (Recall on malicious): TP / (TP + FN)
    fpr: float             # False Positive Rate: FP / (FP + TN)
    accuracy: float        # (TP + TN) / Total


class GuardrailsEvaluator:
    """
    Evaluator for checking safety guardrails against malicious and safe queries.
    """

    def evaluate_query(
        self,
        sql: str,
        expectation: SafetyExpectation,
        dialect: Optional[str] = None,
    ) -> SafetyEvaluationResult:
        """
        Evaluate a single SQL query against guardrails.
        """
        is_safe, reason = validate_sql(sql, dialect=dialect)
        blocked = not is_safe
        correct = (blocked == expectation.should_block)

        is_destructive = any(
            kw in reason.upper() for kw in ["DROP", "DELETE", "TRUNCATE", "ALTER", "INSERT", "UPDATE", "CREATE"]
        )
        is_injection = "INJECTION" in reason.upper() or "DROP" in reason.upper() or ";" in sql

        return SafetyEvaluationResult(
            is_safe_verdict=is_safe,
            expected_block=expectation.should_block,
            correct_verdict=correct,
            reason=reason,
            is_destructive_detected=is_destructive,
            is_injection_detected=is_injection,
            details="Guardrail verdict aligned with safety expectation" if correct else "Safety violation mismatch",
        )

    def benchmark_batch(
        self,
        test_items: list[tuple[str, SafetyExpectation]],
        dialect: Optional[str] = None,
    ) -> GuardrailBenchmarkMetrics:
        """
        Run a batch of SQL queries through the guardrails and compute TPR/FPR.
        """
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        malicious_count = 0
        safe_count = 0

        for sql, exp in test_items:
            res = self.evaluate_query(sql, exp, dialect=dialect)
            if exp.should_block:
                malicious_count += 1
                if not res.is_safe_verdict:
                    tp += 1
                else:
                    fn += 1
                    logger.error("CRITICAL SAFETY LEAK: Malicious query allowed: %s", sql)
            else:
                safe_count += 1
                if res.is_safe_verdict:
                    tn += 1
                else:
                    fp += 1
                    logger.warning("False positive guardrail block on safe query: %s", sql)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        total = len(test_items)
        accuracy = (tp + tn) / total if total > 0 else 1.0

        return GuardrailBenchmarkMetrics(
            total_test_cases=total,
            malicious_queries_count=malicious_count,
            safe_queries_count=safe_count,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            tpr=round(tpr, 4),
            fpr=round(fpr, 4),
            accuracy=round(accuracy, 4),
        )
