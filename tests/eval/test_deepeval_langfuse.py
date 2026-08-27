"""
Unit tests for DeepEval metrics adapter and Langfuse logging integration.
"""

import pytest
import pandas as pd
from agenticsql.eval.deepeval_adapter import (
    SQLSyntaxMetric,
    ASTSemanticMatchMetric,
    ExecutionAccuracyMetric,
    TableGroundedFaithfulnessMetric,
    create_eval_test_case,
    LLMTestCase,
)
from agenticsql.eval.sandbox import DatabaseSandbox
from agenticsql.eval.dataset import GoldenTestCase, SchemaLinkingTarget
from agenticsql.eval.langfuse_adapter import LangfuseEvalLogger


@pytest.fixture
def sandbox():
    sb = DatabaseSandbox(domain="ecommerce")
    yield sb
    sb.close()


def test_deepeval_sql_syntax_metric():
    """Test SQLSyntaxMetric with valid and invalid SQL test cases."""
    metric = SQLSyntaxMetric(threshold=1.0)

    tc_valid = LLMTestCase(input="Show customers", actual_output="SELECT * FROM customers;")
    score = metric.measure(tc_valid)
    assert score == 1.0
    assert metric.is_successful() is True

    tc_invalid = LLMTestCase(input="Show customers", actual_output="SELECT FROM ;")
    score_invalid = metric.measure(tc_invalid)
    assert score_invalid == 0.0
    assert metric.is_successful() is False


def test_deepeval_ast_semantic_match_metric():
    """Test ASTSemanticMatchMetric similarity."""
    metric = ASTSemanticMatchMetric(threshold=0.8)
    tc = LLMTestCase(
        input="Find USA customers",
        actual_output="SELECT id, name FROM customers WHERE country = 'USA';",
        expected_output="SELECT id, name FROM customers WHERE country = 'USA';",
    )
    score = metric.measure(tc)
    assert score == 1.0
    assert metric.is_successful() is True


def test_deepeval_execution_accuracy_metric(sandbox):
    """Test ExecutionAccuracyMetric in sandboxed database."""
    metric = ExecutionAccuracyMetric(sandbox=sandbox, threshold=1.0)
    tc = LLMTestCase(
        input="Count customers",
        actual_output="SELECT COUNT(*) AS total FROM customers;",
        expected_output="SELECT COUNT(*) AS total FROM customers;",
    )
    score = metric.measure(tc)
    assert score == 1.0
    assert metric.is_successful() is True


def test_deepeval_faithfulness_metric():
    """Test TableGroundedFaithfulnessMetric."""
    metric = TableGroundedFaithfulnessMetric(threshold=0.85)
    tc = LLMTestCase(
        input="Customer orders",
        actual_output="Customer 1 made $3147.99.",
        context=["SELECT customer_id, total FROM orders"],
        retrieval_context=["| customer_id | total |\n| 1 | 3147.99 |"],
    )
    score = metric.measure(tc)
    assert score >= 0.85
    assert metric.is_successful() is True


def test_langfuse_eval_logger_offline():
    """Test LangfuseEvalLogger local trace aggregation."""
    logger = LangfuseEvalLogger()
    tc = GoldenTestCase(
        id="test_001",
        category="simple",
        database_domain="ecommerce",
        user_intent="List customers",
        ground_truth_schema=SchemaLinkingTarget(tables=["customers"]),
    )

    record = logger.log_test_evaluation(
        test_case=tc,
        generated_sql="SELECT * FROM customers;",
        agent_output="Here are the customers.",
        scores={"syntax": 1.0, "execution_accuracy": 1.0, "faithfulness": 1.0},
        latency_seconds=0.15,
    )

    assert record.test_id == "test_001"
    assert len(logger.local_trace_records) == 1

    summary = logger.get_summary_metrics()
    assert summary["total_tests"] == 1
    assert summary["metric_averages"]["syntax"] == 1.0
