"""
Unit tests for Schema Linking Evaluator.
"""

import pytest
from agenticsql.eval.schema_eval import SchemaLinkingEvaluator
from agenticsql.eval.dataset import SchemaLinkingTarget


@pytest.fixture
def evaluator():
    return SchemaLinkingEvaluator()


def test_perfect_schema_linking(evaluator):
    """Test 1.0 F1 score on perfect entity extraction."""
    sql = "SELECT c.first_name, c.last_name, o.total_amount FROM customers c JOIN orders o ON c.customer_id = o.customer_id;"
    gt = SchemaLinkingTarget(
        tables=["customers", "orders"],
        columns=[
            "customers.first_name",
            "customers.last_name",
            "orders.total_amount",
            "customers.customer_id",
            "orders.customer_id",
        ],
    )

    res = evaluator.evaluate_from_sql(sql, gt)
    assert res.tables_metric.f1 == 1.0
    assert res.columns_metric.f1 == 1.0
    assert res.overall_f1 == 1.0


def test_partial_schema_linking_missed_table(evaluator):
    """Test precision and recall when a table or column is missing."""
    sql = "SELECT first_name FROM customers;"
    gt = SchemaLinkingTarget(
        tables=["customers", "orders"],
        columns=["customers.first_name", "orders.total_amount"],
    )

    res = evaluator.evaluate_from_sql(sql, gt)
    assert res.tables_metric.recall == 0.5
    assert "orders" in res.tables_metric.false_negatives
    assert res.columns_metric.recall == 0.5
    assert res.overall_f1 < 1.0


def test_hallucinated_column_detection(evaluator):
    """Test false positive identification when extra columns are linked."""
    sql = "SELECT customer_id, first_name, email, credit_card_number FROM customers;"
    gt = SchemaLinkingTarget(
        tables=["customers"],
        columns=["customers.customer_id", "customers.first_name"],
    )

    res = evaluator.evaluate_from_sql(sql, gt)
    assert res.columns_metric.precision < 1.0
    assert any("credit_card_number" in fp for fp in res.columns_metric.false_positives)
