"""
Unit tests for evaluation dataset loading, schemas, and filtering.
"""

from pathlib import Path
import pytest
from agenticsql.eval.dataset import GoldenDataset, GoldenTestCase, SchemaLinkingTarget, SafetyExpectation


@pytest.fixture
def dataset_path():
    p = Path(__file__).parent.parent.parent / "agenticsql" / "eval" / "data" / "golden_sql_eval.json"
    assert p.exists(), f"Golden dataset file does not exist at {p}"
    return p


def test_load_golden_dataset(dataset_path):
    """Test loading and validating golden dataset from JSON."""
    dataset = GoldenDataset.load_from_file(dataset_path)
    assert dataset.name == "AgenticSQL Golden Evaluation Benchmark"
    assert len(dataset.test_cases) >= 10


def test_dataset_contains_all_domains(dataset_path):
    """Verify test cases cover ecommerce, hr_payroll, and financial_ledger."""
    dataset = GoldenDataset.load_from_file(dataset_path)
    domains = {tc.database_domain for tc in dataset.test_cases}
    assert "ecommerce" in domains
    assert "hr_payroll" in domains
    assert "financial_ledger" in domains


def test_dataset_contains_all_categories(dataset_path):
    """Verify presence of key evaluation categories."""
    dataset = GoldenDataset.load_from_file(dataset_path)
    categories = {tc.category for tc in dataset.test_cases}
    assert "simple" in categories
    assert "aggregation_group_by" in categories
    assert "multi_table_join" in categories
    assert "null_handling" in categories
    assert "subquery_cte" in categories
    assert "destructive_ddl_dml" in categories
    assert "sql_injection" in categories
    assert "ambiguous_intent" in categories


def test_dataset_filtering_helpers(dataset_path):
    """Test filtering by category, domain, and ID."""
    dataset = GoldenDataset.load_from_file(dataset_path)
    
    sec_cases = dataset.filter_by_category("destructive_ddl_dml")
    assert len(sec_cases) >= 2
    for tc in sec_cases:
        assert tc.safety_expectation.should_block is True

    ecom_cases = dataset.filter_by_domain("ecommerce")
    assert len(ecom_cases) >= 4

    tc_001 = dataset.get_by_id("eval_ecom_001")
    assert tc_001 is not None
    assert tc_001.database_domain == "ecommerce"
    assert len(tc_001.ground_truth_schema.tables) >= 1


def test_save_and_reload_dataset(tmp_path, dataset_path):
    """Test serializing and reloading dataset."""
    dataset = GoldenDataset.load_from_file(dataset_path)
    target_file = tmp_path / "saved_dataset.json"
    dataset.save_to_file(target_file)
    
    reloaded = GoldenDataset.load_from_file(target_file)
    assert len(reloaded.test_cases) == len(dataset.test_cases)
    assert reloaded.test_cases[0].id == dataset.test_cases[0].id
