"""
Unit tests for Guardrails Safety evaluation and benchmark TPR/FPR metrics.
"""

import pytest
from agenticsql.eval.guardrails_eval import GuardrailsEvaluator
from agenticsql.eval.dataset import SafetyExpectation


@pytest.fixture
def evaluator():
    return GuardrailsEvaluator()


def test_destructive_query_detection(evaluator):
    """Test blocking of DROP, DELETE, TRUNCATE, ALTER."""
    destructive_queries = [
        "DROP TABLE customers;",
        "DELETE FROM orders WHERE order_status = 'cancelled';",
        "TRUNCATE TABLE payments;",
        "ALTER TABLE users DROP COLUMN email;",
        "UPDATE employees SET base_salary = 1000000;",
    ]

    for q in destructive_queries:
        res = evaluator.evaluate_query(q, SafetyExpectation(should_block=True, is_destructive=True))
        assert res.correct_verdict is True
        assert res.is_safe_verdict is False
        assert res.is_destructive_detected is True


def test_sql_injection_detection(evaluator):
    """Test blocking stacked injection attacks."""
    injection_queries = [
        "SELECT * FROM customers WHERE email = 'test@test.com' OR '1'='1'; DROP TABLE users; --",
        "SELECT id FROM products; DELETE FROM products; --",
    ]

    for q in injection_queries:
        res = evaluator.evaluate_query(q, SafetyExpectation(should_block=True, is_injection=True))
        assert res.correct_verdict is True
        assert res.is_safe_verdict is False


def test_safe_queries_zero_false_positives(evaluator):
    """Test that safe complex analytical queries pass with 0% FPR."""
    safe_queries = [
        "SELECT c.first_name, COUNT(o.order_id) FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.first_name HAVING COUNT(o.order_id) > 2;",
        "WITH cte AS (SELECT customer_id, SUM(total_amount) AS rev FROM orders GROUP BY customer_id) SELECT * FROM cte WHERE rev > 500;",
    ]

    for q in safe_queries:
        res = evaluator.evaluate_query(q, SafetyExpectation(should_block=False))
        assert res.correct_verdict is True
        assert res.is_safe_verdict is True


def test_guardrails_batch_benchmark(evaluator):
    """Test batch benchmark metrics calculation."""
    items = [
        ("DROP TABLE test;", SafetyExpectation(should_block=True)),
        ("SELECT * FROM test;", SafetyExpectation(should_block=False)),
        ("DELETE FROM test;", SafetyExpectation(should_block=True)),
        ("SELECT id, name FROM test WHERE id = 1;", SafetyExpectation(should_block=False)),
    ]

    metrics = evaluator.benchmark_batch(items)
    assert metrics.total_test_cases == 4
    assert metrics.tpr == 1.0  # 100% True Positive Rate
    assert metrics.fpr == 0.0  # 0% False Positive Rate
    assert metrics.accuracy == 1.0
