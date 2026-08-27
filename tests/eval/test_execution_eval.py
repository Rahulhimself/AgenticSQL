"""
Unit tests for Execution Accuracy (EX) evaluation and tabular diffing.
"""

import pytest
import pandas as pd
from agenticsql.eval.execution_eval import ExecutionAccuracyEvaluator
from agenticsql.eval.sandbox import DatabaseSandbox
from agenticsql.eval.dataset import ExecutionExpectation


@pytest.fixture
def ecom_sandbox():
    sb = DatabaseSandbox(domain="ecommerce")
    yield sb
    sb.close()


@pytest.fixture
def evaluator(ecom_sandbox):
    return ExecutionAccuracyEvaluator(default_sandbox=ecom_sandbox)


def test_identical_queries_execution_accuracy(evaluator, ecom_sandbox):
    """Identical queries should yield 1.0 accuracy."""
    q = "SELECT customer_id, first_name, country FROM customers WHERE country = 'USA';"
    res = evaluator.evaluate(q, q, sandbox=ecom_sandbox)
    assert res.is_executed is True
    assert res.execution_accuracy == 1.0
    assert res.row_overlap_score == 1.0
    assert res.missing_rows_count == 0
    assert res.extra_rows_count == 0


def test_order_invariance_unordered_results(evaluator, ecom_sandbox):
    """Queries returning same rows in different order should pass when order_sensitive=False."""
    gold = "SELECT customer_id, first_name FROM customers WHERE country = 'USA' ORDER BY customer_id ASC;"
    cand = "SELECT customer_id, first_name FROM customers WHERE country = 'USA' ORDER BY customer_id DESC;"

    res = evaluator.evaluate(cand, gold, sandbox=ecom_sandbox, expectation=ExecutionExpectation(order_sensitive=False))
    assert res.is_executed is True
    assert res.execution_accuracy == 1.0
    assert res.row_overlap_score == 1.0


def test_order_sensitive_ordering_failure(evaluator, ecom_sandbox):
    """Queries returning rows in different order should fail when order_sensitive=True."""
    gold = "SELECT customer_id, first_name FROM customers WHERE country = 'USA' ORDER BY customer_id ASC;"
    cand = "SELECT customer_id, first_name FROM customers WHERE country = 'USA' ORDER BY customer_id DESC;"

    res = evaluator.evaluate(cand, gold, sandbox=ecom_sandbox, expectation=ExecutionExpectation(order_sensitive=True))
    assert res.is_executed is True
    assert res.execution_accuracy == 0.0
    assert "Order mismatch" in (res.diff_summary or "")


def test_float_tolerance_comparison(evaluator):
    """Verify float comparison within tolerance."""
    df1 = pd.DataFrame({"id": [1, 2], "val": [10.00001, 20.50002]})
    df2 = pd.DataFrame({"id": [1, 2], "val": [10.00000, 20.50000]})

    exp = ExecutionExpectation(float_tolerance=1e-4)
    res = evaluator.compare_dataframes(df1, df2, exp)
    assert res.execution_accuracy == 1.0


def test_null_value_equivalence(evaluator):
    """Verify NULL / None / NaN handling."""
    df1 = pd.DataFrame({"id": [1, 2], "val": [None, "Active"]})
    df2 = pd.DataFrame({"id": [1, 2], "val": [float("nan"), "Active"]})

    exp = ExecutionExpectation(allow_null_equivalence=True)
    res = evaluator.compare_dataframes(df1, df2, exp)
    assert res.execution_accuracy == 1.0


def test_mismatched_row_counts_diagnostics(evaluator, ecom_sandbox):
    """Verify diagnostic diff on row count mismatch."""
    gold = "SELECT customer_id, first_name FROM customers WHERE country = 'USA';"
    cand = "SELECT customer_id, first_name FROM customers WHERE country = 'UK';"

    res = evaluator.evaluate(cand, gold, sandbox=ecom_sandbox)
    assert res.execution_accuracy == 0.0
    assert res.missing_rows_count > 0
    assert res.candidate_row_count == 1
    assert res.gold_row_count == 4
