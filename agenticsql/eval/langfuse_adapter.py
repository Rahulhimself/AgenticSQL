"""
Langfuse Trace Scoring and Dataset Synchronization Adapter.

Provides:
1. Multi-step trace and span creation for Agentic Text-to-SQL pipeline runs
2. Step-level score logging (Schema Linking, AST, EX, Guardrails, Faithfulness)
3. Dataset synchronization with Langfuse Datasets
4. Benchmark experiment logging and aggregate metrics computation
"""

import os
import logging
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from .dataset import GoldenDataset, GoldenTestCase

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False
    Langfuse = None  # type: ignore[assignment, misc]


@dataclass
class EvalScoreRecord:
    """A metric score record logged to Langfuse."""
    name: str
    value: float
    comment: Optional[str] = None
    data_type: str = "NUMERIC"


@dataclass
class EvalTraceRecord:
    """Trace and evaluation scores for an individual test case run."""
    test_id: str
    user_intent: str
    generated_sql: Optional[str]
    agent_output: str
    scores: list[EvalScoreRecord] = field(default_factory=list)
    latency_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class LangfuseEvalLogger:
    """
    Adapter for logging evaluation metrics and dataset runs to Langfuse.
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
    ):
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        self.client: Optional[Any] = None
        self.is_connected = False
        self.local_trace_records: list[EvalTraceRecord] = []

        if _LANGFUSE_AVAILABLE and self.public_key and self.secret_key:
            try:
                self.client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                )
                self.is_connected = True
                logger.info("Connected to Langfuse at %s", self.host)
            except Exception as e:
                logger.warning("Could not connect to Langfuse (%s). Falling back to offline logging.", e)
        else:
            logger.info("Langfuse credentials not configured. Running in offline/mock trace logging mode.")

    def log_test_evaluation(
        self,
        test_case: GoldenTestCase,
        generated_sql: Optional[str],
        agent_output: str,
        scores: dict[str, float],
        latency_seconds: float = 0.0,
        comments: Optional[dict[str, str]] = None,
    ) -> EvalTraceRecord:
        """
        Log an evaluation trace with step-level scores.

        Args:
            test_case: GoldenTestCase instance
            generated_sql: Executed SQL string
            agent_output: Natural language agent response
            scores: Dictionary of metric names and numerical values
            latency_seconds: Execution latency
            comments: Optional dictionary of diagnostic notes per metric
        """
        score_records = []
        for metric_name, val in scores.items():
            comment_str = (comments or {}).get(metric_name, "")
            score_records.append(
                EvalScoreRecord(name=metric_name, value=val, comment=comment_str)
            )

        record = EvalTraceRecord(
            test_id=test_case.id,
            user_intent=test_case.user_intent,
            generated_sql=generated_sql,
            agent_output=agent_output,
            scores=score_records,
            latency_seconds=latency_seconds,
        )
        self.local_trace_records.append(record)

        # Upload to live Langfuse instance if connected
        if self.is_connected and self.client is not None:
            try:
                trace = self.client.trace(
                    name=f"eval_run_{test_case.id}",
                    input={"query": test_case.user_intent, "domain": test_case.database_domain},
                    output={"sql": generated_sql, "response": agent_output},
                    metadata={
                        "category": test_case.category,
                        "difficulty": test_case.difficulty,
                        "latency_s": latency_seconds,
                    },
                )

                for s in score_records:
                    trace.score(
                        name=s.name,
                        value=s.value,
                        comment=s.comment,
                    )

                self.client.flush()
            except Exception as e:
                logger.warning("Failed to upload trace to Langfuse: %s", e)

        return record

    def sync_dataset_to_langfuse(self, dataset: GoldenDataset, dataset_name: str = "AgenticSQL-Golden-Eval") -> int:
        """
        Synchronize Golden Dataset test cases to a Langfuse Dataset.
        """
        if not self.is_connected or self.client is None:
            logger.info("Langfuse not connected. Skipping remote dataset sync (%d items cached locally).", len(dataset.test_cases))
            return 0

        synced_count = 0
        try:
            for tc in dataset.test_cases:
                self.client.create_dataset_item(
                    dataset_name=dataset_name,
                    input={"query": tc.user_intent, "domain": tc.database_domain},
                    expected_output={"gold_sql": tc.gold_sql, "facts": tc.nl_response_expectation.grounded_facts},
                    metadata={
                        "test_id": tc.id,
                        "category": tc.category,
                        "difficulty": tc.difficulty,
                    },
                )
                synced_count += 1

            self.client.flush()
            logger.info("Successfully synced %d test cases to Langfuse dataset '%s'", synced_count, dataset_name)
        except Exception as e:
            logger.warning("Error syncing dataset to Langfuse: %s", e)

        return synced_count

    def get_summary_metrics(self) -> dict[str, Any]:
        """
        Compute aggregated summary statistics across all logged trace records.
        """
        if not self.local_trace_records:
            return {"total_tests": 0}

        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        total_latency = 0.0

        for rec in self.local_trace_records:
            total_latency += rec.latency_seconds
            for s in rec.scores:
                metric_sums[s.name] = metric_sums.get(s.name, 0.0) + s.value
                metric_counts[s.name] = metric_counts.get(s.name, 0) + 1

        averages = {
            m: round(metric_sums[m] / metric_counts[m], 4)
            for m in metric_sums
        }

        total_tests = len(self.local_trace_records)
        return {
            "total_tests": total_tests,
            "average_latency_s": round(total_latency / total_tests, 3) if total_tests > 0 else 0.0,
            "metric_averages": averages,
        }
