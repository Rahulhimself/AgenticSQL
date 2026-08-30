"""
AgenticSQL Evaluation Pipeline Package.
Provides comprehensive benchmarking: AST semantic matching, execution parity,
schema linking F1, safety guardrails testing, LLM-as-a-judge, DeepEval, and Langfuse logging.
"""

from .dataset import (
    GoldenDataset,
    GoldenTestCase,
    SchemaLinkingTarget,
    ExecutionExpectation,
    SafetyExpectation,
    NLResponseExpectation,
)
from .sandbox import DatabaseSandbox, create_all_sandboxes, DOMAIN_CONFIGS
from .ast_matcher import SQLASTMatcher, ASTComparisonResult, ClauseMatchResult
from .execution_eval import ExecutionAccuracyEvaluator, ExecutionResult
from .schema_eval import SchemaLinkingEvaluator, SchemaLinkingResult
from .guardrails_eval import GuardrailsEvaluator, GuardrailBenchmarkMetrics
from .judges import (
    LLMJudgeEvaluator,
    FaithfulnessVerdict,
    SemanticParityVerdict,
    AmbiguityVerdict,
)
from .deepeval_adapter import (
    SQLSyntaxMetric,
    ASTSemanticMatchMetric,
    ExecutionAccuracyMetric,
    TableGroundedFaithfulnessMetric,
    create_eval_test_case,
)
from .langfuse_adapter import LangfuseEvalLogger
from .runner import EvaluationRunner, TestCaseResult, BenchmarkSummary

__all__ = [
    "GoldenDataset",
    "GoldenTestCase",
    "SchemaLinkingTarget",
    "ExecutionExpectation",
    "SafetyExpectation",
    "NLResponseExpectation",
    "DatabaseSandbox",
    "create_all_sandboxes",
    "DOMAIN_CONFIGS",
    "SQLASTMatcher",
    "ASTComparisonResult",
    "ClauseMatchResult",
    "ExecutionAccuracyEvaluator",
    "ExecutionResult",
    "SchemaLinkingEvaluator",
    "SchemaLinkingResult",
    "GuardrailsEvaluator",
    "GuardrailBenchmarkMetrics",
    "LLMJudgeEvaluator",
    "FaithfulnessVerdict",
    "SemanticParityVerdict",
    "AmbiguityVerdict",
    "SQLSyntaxMetric",
    "ASTSemanticMatchMetric",
    "ExecutionAccuracyMetric",
    "TableGroundedFaithfulnessMetric",
    "create_eval_test_case",
    "LangfuseEvalLogger",
    "EvaluationRunner",
    "TestCaseResult",
    "BenchmarkSummary",
]
