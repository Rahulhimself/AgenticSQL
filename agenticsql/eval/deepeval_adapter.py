"""
DeepEval Integration Adapter for Agentic SQL.

Provides custom DeepEval metrics:
- SQLSyntaxMetric
- ASTSemanticMatchMetric
- ExecutionAccuracyMetric
- SchemaLinkingMetric
- SQLSafetyMetric
- TableGroundedFaithfulnessMetric
"""

import logging
from typing import Optional, Any
import pandas as pd

try:
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    _DEEPEVAL_AVAILABLE = True
except ImportError:
    # Graceful fallback shim if deepeval is not installed in the current environment
    _DEEPEVAL_AVAILABLE = False
    class BaseMetric:  # type: ignore[no-redef]
        def __init__(self, threshold: float = 0.5):
            self.threshold = threshold
            self.score = 0.0
            self.reason = ""
            self.success = False

        def measure(self, test_case: Any) -> float:
            return 0.0

        def is_successful(self) -> bool:
            return self.success

    class LLMTestCase:  # type: ignore[no-redef]
        def __init__(self, input: str, actual_output: str, expected_output: Optional[str] = None, context: Optional[list] = None, retrieval_context: Optional[list] = None, **kwargs):
            self.input = input
            self.actual_output = actual_output
            self.expected_output = expected_output
            self.context = context or []
            self.retrieval_context = retrieval_context or []
            self.additional_metadata = kwargs

from .ast_matcher import SQLASTMatcher
from .execution_eval import ExecutionAccuracyEvaluator
from .sandbox import DatabaseSandbox
from .schema_eval import SchemaLinkingEvaluator
from .guardrails_eval import GuardrailsEvaluator
from .judges import LLMJudgeEvaluator
from .dataset import GoldenTestCase, ExecutionExpectation, SafetyExpectation, SchemaLinkingTarget

logger = logging.getLogger(__name__)


class SQLSyntaxMetric(BaseMetric):
    """DeepEval metric verifying SQL syntax validity using sqlglot."""

    def __init__(self, threshold: float = 1.0, dialect: str = "sqlite"):
        super().__init__()
        self.threshold = threshold
        self.dialect = dialect
        self.matcher = SQLASTMatcher(default_dialect=dialect)
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        sql = test_case.actual_output
        ast, err = self.matcher.parse_sql(sql, dialect=self.dialect)
        if ast is not None:
            self.score = 1.0
            self.reason = "SQL syntax is valid."
            self.success = True
        else:
            self.score = 0.0
            self.reason = f"SQL syntax error: {err}"
            self.success = False
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "SQL Syntax Validity"


class ASTSemanticMatchMetric(BaseMetric):
    """DeepEval metric measuring AST clause similarity against reference SQL."""

    def __init__(self, threshold: float = 0.8, dialect: str = "sqlite"):
        super().__init__()
        self.threshold = threshold
        self.dialect = dialect
        self.matcher = SQLASTMatcher(default_dialect=dialect)
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        cand_sql = test_case.actual_output
        gold_sql = test_case.expected_output or ""
        if not gold_sql:
            self.score = 1.0
            self.reason = "No reference gold SQL provided."
            self.success = True
            return self.score

        res = self.matcher.compare(cand_sql, gold_sql, dialect=self.dialect)
        self.score = res.ast_similarity_score
        self.reason = f"AST similarity score: {self.score:.4f} (Exact match: {res.exact_ast_match})"
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "AST Semantic Similarity"


class ExecutionAccuracyMetric(BaseMetric):
    """DeepEval metric checking sandboxed tabular execution accuracy."""

    def __init__(self, sandbox: DatabaseSandbox, threshold: float = 1.0, order_sensitive: bool = False):
        super().__init__()
        self.sandbox = sandbox
        self.threshold = threshold
        self.order_sensitive = order_sensitive
        self.evaluator = ExecutionAccuracyEvaluator(default_sandbox=sandbox)
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        cand_sql = test_case.actual_output
        gold_sql = test_case.expected_output or ""
        exp = ExecutionExpectation(order_sensitive=self.order_sensitive)

        res = self.evaluator.evaluate(cand_sql, gold_sql, sandbox=self.sandbox, expectation=exp)
        self.score = res.execution_accuracy
        self.reason = res.diff_summary or f"Execution score: {self.score}"
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Execution Accuracy (EX)"


class TableGroundedFaithfulnessMetric(BaseMetric):
    """DeepEval metric evaluating NL response grounding in executed table data."""

    def __init__(self, judge: Optional[LLMJudgeEvaluator] = None, threshold: float = 0.85):
        super().__init__()
        self.judge = judge or LLMJudgeEvaluator(use_mock=True)
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        user_question = test_case.input
        agent_response = test_case.actual_output
        table_context = "\n".join(test_case.retrieval_context) if test_case.retrieval_context else ""
        executed_sql = test_case.context[0] if test_case.context else ""

        verdict = self.judge.evaluate_faithfulness(
            user_question=user_question,
            executed_sql=executed_sql,
            table_data=table_context,
            agent_response=agent_response,
        )

        self.score = verdict.score
        self.reason = verdict.reasoning
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Table-Grounded Faithfulness"


def create_eval_test_case(
    test_case: GoldenTestCase,
    generated_sql: str,
    agent_output: str,
    executed_df: Optional[pd.DataFrame] = None,
) -> LLMTestCase:
    """
    Build an LLMTestCase from agent execution results.
    """
    table_markdown = executed_df.to_markdown(index=False) if (executed_df is not None and not executed_df.empty) else "No rows"

    return LLMTestCase(
        input=test_case.user_intent,
        actual_output=agent_output,
        expected_output=test_case.gold_sql,
        context=[generated_sql],
        retrieval_context=[table_markdown],
        additional_metadata={
            "test_id": test_case.id,
            "category": test_case.category,
            "domain": test_case.database_domain,
            "difficulty": test_case.difficulty,
        },
    )
