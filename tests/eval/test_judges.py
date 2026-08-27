"""
Unit tests for LLM-as-a-Judge Rubrics (Faithfulness, Semantic Parity, Ambiguity).
"""

import pytest
import pandas as pd
from agenticsql.eval.judges import LLMJudgeEvaluator


@pytest.fixture
def judge():
    return LLMJudgeEvaluator(use_mock=True)


def test_faithfulness_grounded_response(judge):
    """Grounded responses should achieve 1.0 faithfulness."""
    df = pd.DataFrame({"customer_id": [1, 2], "revenue": [3147.99, 1944.50]})
    resp = "Customer 1 generated $3147.99 in revenue, while Customer 2 generated $1944.50."

    verdict = judge.evaluate_faithfulness(
        user_question="What is the revenue for top customers?",
        executed_sql="SELECT customer_id, revenue FROM ...",
        table_data=df,
        agent_response=resp,
    )
    assert verdict.is_faithful is True
    assert verdict.score >= 0.85
    assert len(verdict.hallucinations_detected) == 0


def test_faithfulness_hallucinated_response(judge):
    """Fabricated numbers not in table should be flagged as hallucinations."""
    df = pd.DataFrame({"customer_id": [1, 2], "revenue": [3147.99, 1944.50]})
    resp = "Customer 1 made $9999.00 and Customer 99 generated $5555.00."

    verdict = judge.evaluate_faithfulness(
        user_question="What is the revenue for top customers?",
        executed_sql="SELECT customer_id, revenue FROM ...",
        table_data=df,
        agent_response=resp,
    )
    assert verdict.is_faithful is False
    assert verdict.score < 0.85
    assert len(verdict.hallucinations_detected) > 0


def test_semantic_parity_identical_queries(judge):
    """Identical SQL queries should pass semantic parity with score 1.0."""
    sql = "SELECT id, name FROM customers WHERE country = 'USA';"
    verdict = judge.evaluate_semantic_parity(
        user_intent="List US customers",
        gold_sql=sql,
        candidate_sql=sql,
    )
    assert verdict.is_semantically_equivalent is True
    assert verdict.score == 1.0


def test_ambiguity_clarification_handling(judge):
    """Test that clarification keywords or assumption disclosures pass ambiguity judge."""
    resp_good = "Assuming you mean total revenue in USD, here are the top selling products: ..."
    verdict_good = judge.evaluate_ambiguity(
        user_intent="Show me top products",
        agent_response=resp_good,
    )
    assert verdict_good.handled_appropriately is True
    assert verdict_good.assumptions_stated is True

    resp_poor = "Here are the products."
    verdict_poor = judge.evaluate_ambiguity(
        user_intent="Show me top products",
        agent_response=resp_poor,
    )
    assert verdict_poor.handled_appropriately is False
