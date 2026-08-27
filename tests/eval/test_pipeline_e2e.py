"""
End-to-End Evaluation Pipeline and Benchmark Runner test suite.
"""

from pathlib import Path
import pytest
from agenticsql.eval.runner import EvaluationRunner


@pytest.fixture
def runner():
    return EvaluationRunner(dialect="sqlite", use_mock_judge=True)


def test_runner_executes_all_golden_test_cases(runner):
    """Verify runner evaluates all golden test cases and produces summary KPIs."""
    results, summary = runner.run_all()
    assert summary.total_tests >= 10
    assert summary.passed_tests > 0
    assert summary.pass_rate >= 0.85
    assert summary.syntax_validity_rate >= 0.95
    assert summary.guardrail_tpr == 1.0  # 100% TPR for safety
    assert summary.execution_accuracy_rate >= 0.85
    assert summary.mean_faithfulness >= 0.85


def test_runner_category_breakdown(runner):
    """Verify category-level aggregation."""
    results, summary = runner.run_all()
    breakdown = summary.category_breakdown
    assert "simple" in breakdown
    assert "aggregation_group_by" in breakdown
    assert "multi_table_join" in breakdown
    assert "destructive_ddl_dml" in breakdown


def test_runner_exports_json_and_html_reports(tmp_path, runner):
    """Verify report export functionality."""
    results, summary = runner.run_all()
    json_p, html_p = runner.export_reports(results, summary, output_dir=tmp_path)

    assert json_p.exists()
    assert html_p.exists()

    with open(json_p, "r", encoding="utf-8") as f:
        data = json_p.read_text(encoding="utf-8")
        assert "summary" in data
        assert "results" in data

    html_content = html_p.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_content
    assert "AgenticSQL Evaluation Report" in html_content
