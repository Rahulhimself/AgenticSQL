"""
Dataset definitions, Pydantic schemas, and loaders for Agentic SQL Evaluation.

Provides structured models for test cases covering all 5 pipeline stages:
1. Schema Linking targets
2. Gold SQL & Dialect variations
3. Safety expectations & injection vectors
4. Execution constraints and tolerances
5. NL response grounding facts & ambiguity markers
"""

import json
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class SchemaLinkingTarget(BaseModel):
    """
    Ground truth schema entities (tables, columns, foreign keys) expected for intent resolution.
    """
    tables: list[str] = Field(default_factory=list, description="Target table names")
    columns: list[str] = Field(default_factory=list, description="Qualified column names (e.g. table.col)")
    foreign_keys: list[str] = Field(default_factory=list, description="Expected join relationships")


class SafetyExpectation(BaseModel):
    """
    Safety expectations for AST inspection, injection detection, and blocked statements.
    """
    should_block: bool = Field(default=False, description="True if query must be blocked")
    is_destructive: bool = Field(default=False, description="Contains DROP, TRUNCATE, DELETE, ALTER, etc.")
    is_injection: bool = Field(default=False, description="Contains SQL injection payloads")
    is_forbidden_proc: bool = Field(default=False, description="Calls xp_cmdshell, sp_configure, etc.")
    expected_category: Optional[str] = Field(default=None, description="Expected violation category")


class ExecutionExpectation(BaseModel):
    """
    Constraints and tolerances (float epsilon, order sensitivity) for sandboxed result comparison.
    """
    order_sensitive: bool = Field(default=False, description="True if ORDER BY is strictly required")
    float_tolerance: float = Field(default=1e-4, description="Absolute/relative float equality tolerance")
    allow_null_equivalence: bool = Field(default=True, description="Treat None, NaN, and NULL as equivalent")
    case_sensitive_strings: bool = Field(default=False, description="Case-sensitive string matching")
    expected_row_count: Optional[int] = Field(default=None, description="Expected number of result rows")
    expected_columns: Optional[list[str]] = Field(default=None, description="Expected result column names")


class NLResponseExpectation(BaseModel):
    """Expectations for natural language synthesis, grounding, and ambiguity."""
    grounded_facts: list[str] = Field(
        default_factory=list,
        description="Key numerical or categorical facts that MUST be present in the response",
    )
    forbidden_hallucinations: list[str] = Field(
        default_factory=list,
        description="Plausible hallucinations that must NOT be present",
    )
    is_ambiguous: bool = Field(default=False, description="True if user query is ambiguous")
    clarification_points: list[str] = Field(
        default_factory=list,
        description="Ambiguity points the agent should ask to clarify or state assumptions about",
    )


class GoldenTestCase(BaseModel):
    """Comprehensive test case for end-to-end evaluation."""
    id: str = Field(..., description="Unique test case identifier (e.g., eval_ecom_001)")
    category: Literal[
        "simple",
        "aggregation_group_by",
        "multi_table_join",
        "null_handling",
        "subquery_cte",
        "destructive_ddl_dml",
        "sql_injection",
        "ambiguous_intent",
        "dialect_specific",
    ] = Field(..., description="Evaluation category")
    database_domain: Literal["ecommerce", "hr_payroll", "financial_ledger"] = Field(
        ..., description="Test database domain schema to run against"
    )
    user_intent: str = Field(..., description="User's natural language input query")
    conversation_context: list[dict] = Field(
        default_factory=list, description="Prior conversation turns for multi-turn evaluation"
    )
    ground_truth_schema: SchemaLinkingTarget = Field(
        default_factory=SchemaLinkingTarget, description="Expected schema entities"
    )
    gold_sql: Optional[str] = Field(default=None, description="Canonical reference SQL query")
    dialect_variations: dict[str, str] = Field(
        default_factory=dict, description="Dialect-specific queries (sqlite, postgres, tsql, mysql)"
    )
    safety_expectation: SafetyExpectation = Field(
        default_factory=SafetyExpectation, description="Safety guardrail expectations"
    )
    execution_expectation: ExecutionExpectation = Field(
        default_factory=ExecutionExpectation, description="Tabular execution expectations"
    )
    nl_response_expectation: NLResponseExpectation = Field(
        default_factory=NLResponseExpectation, description="Grounding and synthesis expectations"
    )
    difficulty: Literal["easy", "medium", "hard", "adversarial"] = Field(
        default="medium", description="Difficulty level"
    )
    description: str = Field(default="", description="Brief explanation of test case purpose")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Test case ID cannot be empty.")
        return v.strip()


class GoldenDataset(BaseModel):
    """Collection of golden evaluation test cases."""
    version: str = Field(default="1.0.0", description="Dataset schema version")
    name: str = Field(default="AgenticSQL Golden Evaluation Benchmark")
    description: str = Field(default="Benchmark dataset for multi-step text-to-SQL agent evaluation")
    test_cases: list[GoldenTestCase] = Field(default_factory=list)

    def get_by_id(self, test_id: str) -> Optional[GoldenTestCase]:
        """Find a test case by unique identifier."""
        for tc in self.test_cases:
            if tc.id == test_id:
                return tc
        return None

    def filter_by_category(self, category: str) -> list[GoldenTestCase]:
        """Filter test cases by category."""
        return [tc for tc in self.test_cases if tc.category == category]

    def filter_by_domain(self, domain: str) -> list[GoldenTestCase]:
        """Filter test cases by database domain."""
        return [tc for tc in self.test_cases if tc.database_domain == domain]

    def filter_by_difficulty(self, difficulty: str) -> list[GoldenTestCase]:
        """Filter test cases by difficulty level."""
        return [tc for tc in self.test_cases if tc.difficulty == difficulty]

    def save_to_file(self, file_path: str | Path) -> None:
        """Serialize dataset to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str | Path) -> "GoldenDataset":
        """Load and validate dataset from a JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Golden dataset file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)
