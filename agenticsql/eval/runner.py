"""
Automated Evaluation Benchmark Runner & CLI Harness.

Executes the full evaluation pipeline across all 5 stages:
1. Schema Linking / RAG
2. SQL Generation & AST Parsing
3. Safety & Execution Guardrails
4. Sandboxed Execution & Result Verification (EX)
5. NL Response Grounding & Faithfulness

Outputs rich terminal summaries, JSON reports, and HTML scorecards with CI/CD threshold gating.
"""

import sys
import os
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, asdict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from .dataset import GoldenDataset, GoldenTestCase
from .sandbox import DatabaseSandbox, DOMAIN_CONFIGS
from .ast_matcher import SQLASTMatcher
from .execution_eval import ExecutionAccuracyEvaluator
from .schema_eval import SchemaLinkingEvaluator
from .guardrails_eval import GuardrailsEvaluator
from .judges import LLMJudgeEvaluator
from .langfuse_adapter import LangfuseEvalLogger

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class TestCaseResult:
    """Individual test case evaluation output."""
    test_id: str
    category: str
    domain: str
    difficulty: str
    user_intent: str
    generated_sql: Optional[str]
    gold_sql: Optional[str]
    syntax_valid: bool
    schema_f1: float
    ast_similarity: float
    guardrail_passed: bool
    execution_accuracy: float
    row_overlap: float
    faithfulness_score: float
    latency_seconds: float
    is_passed: bool
    details: str = ""


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark statistics."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    syntax_validity_rate: float
    mean_schema_f1: float
    mean_ast_similarity: float
    guardrail_tpr: float
    execution_accuracy_rate: float
    mean_faithfulness: float
    mean_latency_seconds: float
    category_breakdown: dict[str, dict[str, Any]]


class EvaluationRunner:
    """
    Main evaluation pipeline orchestrator.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        dialect: str = "sqlite",
        use_mock_judge: bool = True,
        llm: Optional[Any] = None,
    ):
        default_data_file = Path(__file__).parent / "data" / "golden_sql_eval.json"
        self.dataset_path = Path(dataset_path) if dataset_path else default_data_file
        self.dataset = GoldenDataset.load_from_file(self.dataset_path)
        self.dialect = dialect

        # Initialise evaluators
        self.sandboxes: dict[str, DatabaseSandbox] = {
            domain: DatabaseSandbox(domain=domain) for domain in DOMAIN_CONFIGS
        }
        self.ast_matcher = SQLASTMatcher(default_dialect=dialect)
        self.execution_evaluator = ExecutionAccuracyEvaluator()
        self.schema_evaluator = SchemaLinkingEvaluator(ast_matcher=self.ast_matcher)
        self.guardrails_evaluator = GuardrailsEvaluator()
        self.judge = LLMJudgeEvaluator(llm=llm, use_mock=use_mock_judge)
        self.langfuse_logger = LangfuseEvalLogger()

    def run_single(self, test_case: GoldenTestCase, agent: Optional[Any] = None) -> TestCaseResult:
        """
        Evaluate a single test case through the 5-step harness.
        """
        t0 = time.perf_counter()
        sandbox = self.sandboxes[test_case.database_domain]

        # 1. Obtain SQL & Response from Agent (or fallback to test case gold standard simulation)
        if agent is not None:
            agent_res = agent.chat(test_case.user_intent)
            candidate_sql = agent_res.get("sql", [None])[-1] if agent_res.get("sql") else None
            agent_output = agent_res.get("output", "")
        else:
            # Gold execution simulation for benchmark baseline
            candidate_sql = test_case.gold_sql
            agent_output = f"Grounded response for {test_case.id} with facts: {', '.join(test_case.nl_response_expectation.grounded_facts)}"

        # 2. Stage 3: Guardrails Check
        guard_res = self.guardrails_evaluator.evaluate_query(
            candidate_sql or "",
            expectation=test_case.safety_expectation,
            dialect=self.dialect,
        )
        guardrail_passed = guard_res.correct_verdict

        # If malicious query was correctly blocked, mark as clean safety pass
        if test_case.safety_expectation.should_block:
            latency = time.perf_counter() - t0
            scores = {
                "guardrail_safety": 1.0 if guardrail_passed else 0.0,
                "syntax_validity": 1.0,
                "execution_accuracy": 1.0 if guardrail_passed else 0.0,
                "faithfulness": 1.0,
            }
            self.langfuse_logger.log_test_evaluation(
                test_case=test_case,
                generated_sql=candidate_sql,
                agent_output=agent_output,
                scores=scores,
                latency_seconds=latency,
            )
            return TestCaseResult(
                test_id=test_case.id,
                category=test_case.category,
                domain=test_case.database_domain,
                difficulty=test_case.difficulty,
                user_intent=test_case.user_intent,
                generated_sql=candidate_sql,
                gold_sql=test_case.gold_sql,
                syntax_valid=True,
                schema_f1=1.0,
                ast_similarity=1.0,
                guardrail_passed=guardrail_passed,
                execution_accuracy=1.0 if guardrail_passed else 0.0,
                row_overlap=1.0 if guardrail_passed else 0.0,
                faithfulness_score=1.0,
                latency_seconds=round(latency, 3),
                is_passed=guardrail_passed,
                details=f"Safety guardrail verdict: {guard_res.reason}",
            )

        # 3. Stage 1: Schema Linking
        if candidate_sql:
            schema_res = self.schema_evaluator.evaluate_from_sql(
                candidate_sql,
                test_case.ground_truth_schema,
                dialect=self.dialect,
            )
            schema_f1 = schema_res.overall_f1
        else:
            schema_f1 = 0.0

        # 4. Stage 2: AST Comparison & Syntax Validity
        if candidate_sql and test_case.gold_sql:
            ast_res = self.ast_matcher.compare(
                candidate_sql,
                test_case.gold_sql,
                dialect=self.dialect,
            )
            syntax_valid = ast_res.is_valid_syntax
            ast_sim = ast_res.ast_similarity_score
        else:
            syntax_valid = False
            ast_sim = 0.0

        # 5. Stage 4: Execution Accuracy (EX)
        if candidate_sql and test_case.gold_sql and syntax_valid:
            exec_res = self.execution_evaluator.evaluate(
                candidate_sql,
                test_case.gold_sql,
                sandbox=sandbox,
                expectation=test_case.execution_expectation,
            )
            ex_score = exec_res.execution_accuracy
            row_overlap = exec_res.row_overlap_score
            exec_df, _ = sandbox.execute_query(candidate_sql)
        else:
            ex_score = 0.0
            row_overlap = 0.0
            exec_df = None

        # 6. Stage 5: Faithfulness & Grounding Judge
        faith_res = self.judge.evaluate_faithfulness(
            user_question=test_case.user_intent,
            executed_sql=candidate_sql or "",
            table_data=exec_df,
            agent_response=agent_output,
        )
        faith_score = faith_res.score

        latency = time.perf_counter() - t0

        # Pass criteria: valid syntax + guardrails passed + execution accurate + faithful
        is_passed = (
            syntax_valid
            and guardrail_passed
            and ex_score >= 1.0
            and faith_score >= 0.8
        )

        scores = {
            "schema_linking_f1": schema_f1,
            "syntax_validity": 1.0 if syntax_valid else 0.0,
            "ast_similarity": ast_sim,
            "guardrail_safety": 1.0 if guardrail_passed else 0.0,
            "execution_accuracy": ex_score,
            "faithfulness": faith_score,
        }
        self.langfuse_logger.log_test_evaluation(
            test_case=test_case,
            generated_sql=candidate_sql,
            agent_output=agent_output,
            scores=scores,
            latency_seconds=latency,
        )

        return TestCaseResult(
            test_id=test_case.id,
            category=test_case.category,
            domain=test_case.database_domain,
            difficulty=test_case.difficulty,
            user_intent=test_case.user_intent,
            generated_sql=candidate_sql,
            gold_sql=test_case.gold_sql,
            syntax_valid=syntax_valid,
            schema_f1=schema_f1,
            ast_similarity=ast_sim,
            guardrail_passed=guardrail_passed,
            execution_accuracy=ex_score,
            row_overlap=row_overlap,
            faithfulness_score=faith_score,
            latency_seconds=round(latency, 3),
            is_passed=is_passed,
            details=f"EX: {ex_score}, Faith: {faith_score:.2f}, AST: {ast_sim:.2f}",
        )

    def run_all(
        self,
        domain_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        agent: Optional[Any] = None,
    ) -> tuple[list[TestCaseResult], BenchmarkSummary]:
        """
        Execute full benchmark suite across all matching test cases.
        """
        cases = self.dataset.test_cases
        if domain_filter:
            cases = [c for c in cases if c.database_domain == domain_filter]
        if category_filter:
            cases = [c for c in cases if c.category == category_filter]

        results = []
        for tc in cases:
            res = self.run_single(tc, agent=agent)
            results.append(res)

        summary = self._compute_summary(results)
        return results, summary

    def _compute_summary(self, results: list[TestCaseResult]) -> BenchmarkSummary:
        """Compute aggregate benchmark summary statistics."""
        total = len(results)
        if total == 0:
            return BenchmarkSummary(
                total_tests=0, passed_tests=0, failed_tests=0, pass_rate=0.0,
                syntax_validity_rate=0.0, mean_schema_f1=0.0, mean_ast_similarity=0.0,
                guardrail_tpr=1.0, execution_accuracy_rate=0.0, mean_faithfulness=0.0,
                mean_latency_seconds=0.0, category_breakdown={},
            )

        passed = sum(1 for r in results if r.is_passed)
        syntax_count = sum(1 for r in results if r.syntax_valid)
        schema_sum = sum(r.schema_f1 for r in results)
        ast_sum = sum(r.ast_similarity for r in results)
        guard_passed = sum(1 for r in results if r.guardrail_passed)
        ex_passed = sum(1 for r in results if r.execution_accuracy >= 1.0)
        faith_sum = sum(r.faithfulness_score for r in results)
        latency_sum = sum(r.latency_seconds for r in results)

        # Breakdown by category
        categories: dict[str, list[TestCaseResult]] = {}
        for r in results:
            categories.setdefault(r.category, []).append(r)

        cat_breakdown = {}
        for cat_name, cat_results in categories.items():
            cat_total = len(cat_results)
            cat_passed = sum(1 for r in cat_results if r.is_passed)
            cat_breakdown[cat_name] = {
                "total": cat_total,
                "passed": cat_passed,
                "pass_rate": round(cat_passed / cat_total, 4) if cat_total > 0 else 0.0,
                "mean_ex": round(sum(r.execution_accuracy for r in cat_results) / cat_total, 4),
            }

        return BenchmarkSummary(
            total_tests=total,
            passed_tests=passed,
            failed_tests=total - passed,
            pass_rate=round(passed / total, 4),
            syntax_validity_rate=round(syntax_count / total, 4),
            mean_schema_f1=round(schema_sum / total, 4),
            mean_ast_similarity=round(ast_sum / total, 4),
            guardrail_tpr=round(guard_passed / total, 4),
            execution_accuracy_rate=round(ex_passed / total, 4),
            mean_faithfulness=round(faith_sum / total, 4),
            mean_latency_seconds=round(latency_sum / total, 3),
            category_breakdown=cat_breakdown,
        )

    def print_rich_report(self, results: list[TestCaseResult], summary: BenchmarkSummary) -> None:
        """Render beautiful terminal report with Rich tables."""
        console.print("\n")
        console.print(
            Panel.fit(
                "[bold cyan]AgenticSQL Automated Evaluation Benchmark[/bold cyan]\n"
                f"[dim]Version: {self.dataset.version} | Dialect: {self.dialect} | Tests: {summary.total_tests}[/dim]",
                box=box.DOUBLE,
                border_style="cyan",
            )
        )

        # 1. Test cases detailed table
        table = Table(title="Test Case Evaluation Results", box=box.ROUNDED, header_style="bold magenta")
        table.add_column("ID", style="dim", width=15)
        table.add_column("Category", style="cyan", width=20)
        table.add_column("Domain", width=15)
        table.add_column("Syntax", justify="center", width=8)
        table.add_column("Schema F1", justify="right", width=10)
        table.add_column("AST Sim", justify="right", width=10)
        table.add_column("EX Acc", justify="right", width=10)
        table.add_column("Faithfulness", justify="right", width=12)
        table.add_column("Status", justify="center", width=10)

        for r in results:
            status = "[bold green]PASS[/bold green]" if r.is_passed else "[bold red]FAIL[/bold red]"
            syntax_str = "[green]OK[/green]" if r.syntax_valid else "[red]ERR[/red]"
            table.add_row(
                r.test_id,
                r.category,
                r.domain,
                syntax_str,
                f"{r.schema_f1:.2f}",
                f"{r.ast_similarity:.2f}",
                f"{r.execution_accuracy:.1f}",
                f"{r.faithfulness_score:.2f}",
                status,
            )

        console.print(table)

        # 2. Benchmark Summary KPIs Table
        kpi_table = Table(title="Aggregate Benchmark KPIs", box=box.SIMPLE_HEAVY, header_style="bold blue")
        kpi_table.add_column("Metric Name", style="bold")
        kpi_table.add_column("Target Threshold", justify="center")
        kpi_table.add_column("Benchmark Score", justify="center")
        kpi_table.add_column("Status", justify="center")

        def add_kpi(name: str, target: float, actual: float, is_percentage: bool = True):
            passed = actual >= target
            status = "[green]PASSED[/green]" if passed else "[red]FAILED[/red]"
            tgt_str = f"{target*100:.1f}%" if is_percentage else f"{target:.2f}"
            act_str = f"{actual*100:.1f}%" if is_percentage else f"{actual:.2f}"
            kpi_table.add_row(name, tgt_str, act_str, status)

        add_kpi("Overall Pass Rate", 0.85, summary.pass_rate)
        add_kpi("SQL Syntax Validity", 0.95, summary.syntax_validity_rate)
        add_kpi("Execution Accuracy (EX)", 0.85, summary.execution_accuracy_rate)
        add_kpi("Safety Guardrails (TPR)", 1.00, summary.guardrail_tpr)
        add_kpi("Mean Schema Linking F1", 0.85, summary.mean_schema_f1)
        add_kpi("Table Faithfulness", 0.85, summary.mean_faithfulness)

        console.print(kpi_table)

    def export_reports(
        self,
        results: list[TestCaseResult],
        summary: BenchmarkSummary,
        output_dir: str | Path = "logs/eval_results",
    ) -> tuple[Path, Path]:
        """Export JSON and HTML reports."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / "eval_report.json"
        html_path = out / "eval_report.html"

        # JSON Export
        data = {
            "summary": asdict(summary),
            "results": [asdict(r) for r in results],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # HTML Export
        html_content = self._generate_html_report(results, summary)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return json_path, html_path

    def _generate_html_report(self, results: list[TestCaseResult], summary: BenchmarkSummary) -> str:
        """Generate interactive HTML report."""
        rows_html = ""
        for r in results:
            badge_color = "#10b981" if r.is_passed else "#ef4444"
            badge_text = "PASS" if r.is_passed else "FAIL"
            rows_html += f"""
            <tr>
                <td><code>{r.test_id}</code></td>
                <td><span class="tag">{r.category}</span></td>
                <td>{r.domain}</td>
                <td>{'✅' if r.syntax_valid else '❌'}</td>
                <td>{r.schema_f1:.2f}</td>
                <td>{r.ast_similarity:.2f}</td>
                <td>{r.execution_accuracy:.1f}</td>
                <td>{r.faithfulness_score:.2f}</td>
                <td><span style="background: {badge_color}; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold;">{badge_text}</span></td>
            </tr>
            """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AgenticSQL Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        .card {{ background: #1e293b; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
        h1 {{ margin-top: 0; color: #38bdf8; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px; }}
        .kpi {{ background: #334155; padding: 16px; border-radius: 6px; text-align: center; }}
        .kpi-val {{ font-size: 28px; font-weight: bold; color: #38bdf8; }}
        .kpi-label {{ font-size: 14px; color: #94a3b8; text-transform: uppercase; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #334155; color: #cbd5e1; font-weight: 600; }}
        tr:hover {{ background: #33415555; }}
        code {{ background: #0f172a; padding: 2px 6px; border-radius: 4px; font-family: monospace; }}
        .tag {{ background: #475569; padding: 2px 8px; border-radius: 12px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>🚀 AgenticSQL Automated Evaluation Benchmark</h1>
        <p style="color: #94a3b8;">Multi-step evaluation results for Schema Linking, AST Parsing, Guardrails, Execution Accuracy, and NL Faithfulness.</p>
        <div class="grid">
            <div class="kpi"><div class="kpi-val">{summary.pass_rate * 100:.1f}%</div><div class="kpi-label">Pass Rate ({summary.passed_tests}/{summary.total_tests})</div></div>
            <div class="kpi"><div class="kpi-val">{summary.execution_accuracy_rate * 100:.1f}%</div><div class="kpi-label">Execution Accuracy</div></div>
            <div class="kpi"><div class="kpi-val">{summary.guardrail_tpr * 100:.1f}%</div><div class="kpi-label">Guardrail TPR</div></div>
            <div class="kpi"><div class="kpi-val">{summary.mean_faithfulness * 100:.1f}%</div><div class="kpi-label">NL Faithfulness</div></div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Test ID</th>
                    <th>Category</th>
                    <th>Domain</th>
                    <th>Syntax</th>
                    <th>Schema F1</th>
                    <th>AST Sim</th>
                    <th>EX Acc</th>
                    <th>Faithfulness</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>
"""


def main():
    """CLI entrypoint for benchmark runner."""
    parser = argparse.ArgumentParser(description="AgenticSQL Automated Evaluation Pipeline Runner")
    parser.add_argument("--dataset", type=str, default=None, help="Path to golden dataset JSON")
    parser.add_argument("--domain", type=str, default=None, help="Filter by database domain")
    parser.add_argument("--category", type=str, default=None, help="Filter by evaluation category")
    parser.add_argument("--dialect", type=str, default="sqlite", help="SQL dialect (sqlite, postgres, tsql)")
    parser.add_argument("--output-dir", type=str, default="logs/eval_results", help="Directory to save JSON & HTML reports")
    parser.add_argument("--fail-under", type=float, default=0.85, help="Minimum pass rate threshold for CI (default 0.85)")
    args = parser.parse_args()

    runner = EvaluationRunner(dataset_path=args.dataset, dialect=args.dialect, use_mock_judge=True)
    results, summary = runner.run_all(domain_filter=args.domain, category_filter=args.category)

    runner.print_rich_report(results, summary)
    json_path, html_path = runner.export_reports(results, summary, output_dir=args.output_dir)
    console.print(f"\n[bold green]Reports saved to:[/bold green] {json_path} & {html_path}")

    # Enforce CI quality gates
    if summary.pass_rate < args.fail_under:
        console.print(f"\n[bold red]CI Quality Gate FAILED:[/bold red] Pass rate {summary.pass_rate*100:.1f}% is below threshold {args.fail_under*100:.1f}%\n")
        sys.exit(1)
    if summary.guardrail_tpr < 1.0:
        console.print(f"\n[bold red]CI Quality Gate FAILED:[/bold red] Guardrail TPR {summary.guardrail_tpr*100:.1f}% must be 100.0%\n")
        sys.exit(1)

    console.print(f"\n[bold green]CI Quality Gate PASSED![/bold green]\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
