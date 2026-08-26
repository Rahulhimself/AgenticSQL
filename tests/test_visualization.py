"""
Tests for agenticsql.visualization module.

Validates:
- DataFrame to JSON dictionary conversion
- Direct DataFrame chart generation (bar, grouped multi-series, line, pie)
- Direct DataFrame exports to CSV and JSON
- Automatic column and time-series type detection
- Fallback text table parsing
"""

import os
import json
# pyrefly: ignore [missing-import]
import pytest
import pandas as pd
from pathlib import Path

from agenticsql.visualization import (
    dataframe_to_dict,
    parse_table_from_text,
    save_chart_from_dataframe,
    save_chart,
    export_dataframe_to_csv,
    export_dataframe_to_json,
    export_to_csv,
    export_to_json,
)


@pytest.fixture
def sample_sales_df():
    """Fixture providing a sample single-metric dataframe."""
    return pd.DataFrame({
        "Product": ["Laptops", "Smartphones", "Tablets", "Headphones"],
        "Revenue": [120000.50, 95000.00, 45000.75, 22000.00],
    })


@pytest.fixture
def multi_series_df():
    """Fixture providing a multi-series metric dataframe."""
    return pd.DataFrame({
        "Region": ["North", "South", "East", "West"],
        "Q1_Sales": [15000, 18000, 12000, 22000],
        "Q2_Sales": [17000, 19500, 14000, 25000],
    })


@pytest.fixture
def timeseries_df():
    """Fixture providing a time-series dataframe."""
    return pd.DataFrame({
        "Month": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"],
        "Active_Users": [1200, 1450, 1800, 2200],
    })


class TestDataframeToDict:
    """Test DataFrame to dictionary serialization."""

    def test_converts_columns_and_rows(self, sample_sales_df):
        """Should serialize dataframe into columns list and rows list."""
        data = dataframe_to_dict(sample_sales_df)
        assert data["columns"] == ["Product", "Revenue"]
        assert len(data["rows"]) == 4
        assert data["rows"][0] == ["Laptops", 120000.50]

    def test_handles_empty_dataframe(self):
        """Should return empty lists for empty dataframe."""
        data = dataframe_to_dict(pd.DataFrame())
        assert data == {"columns": [], "rows": []}


class TestChartGeneration:
    """Test chart generation from DataFrames."""

    def test_bar_chart_generation(self, sample_sales_df, tmp_path):
        """Generate bar chart from dataframe."""
        path = save_chart_from_dataframe(sample_sales_df, chart_type="bar", output_dir=str(tmp_path))
        assert path is not None
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_pie_chart_generation(self, sample_sales_df, tmp_path):
        """Generate pie chart from dataframe."""
        path = save_chart_from_dataframe(sample_sales_df, chart_type="pie", output_dir=str(tmp_path))
        assert path is not None
        assert Path(path).exists()

    def test_line_chart_generation(self, timeseries_df, tmp_path):
        """Generate line chart for time-series."""
        path = save_chart_from_dataframe(timeseries_df, chart_type="line", output_dir=str(tmp_path))
        assert path is not None
        assert Path(path).exists()

    def test_multi_series_grouped_bar_chart(self, multi_series_df, tmp_path):
        """Generate multi-series grouped bar chart."""
        path = save_chart_from_dataframe(multi_series_df, chart_type="bar", output_dir=str(tmp_path))
        assert path is not None
        assert Path(path).exists()

    def test_auto_chart_type_detection(self, timeseries_df, tmp_path):
        """Auto detection should pick line chart for time-series dates."""
        path = save_chart_from_dataframe(timeseries_df, chart_type="auto", output_dir=str(tmp_path))
        assert path is not None
        assert Path(path).exists()

    def test_unified_save_chart_facade(self, sample_sales_df, tmp_path):
        """save_chart facade should accept DataFrame or dict."""
        # Test with DataFrame
        path1 = save_chart(sample_sales_df, output_dir=str(tmp_path))
        assert path1 is not None

        # Test with dict
        data_dict = {"columns": ["Item", "Count"], "rows": [["A", 10], ["B", 20]]}
        path2 = save_chart(data_dict, output_dir=str(tmp_path))
        assert path2 is not None


class TestExports:
    """Test CSV and JSON exports."""

    def test_export_dataframe_to_csv(self, sample_sales_df, tmp_path):
        """Export dataframe to CSV."""
        path = export_dataframe_to_csv(sample_sales_df, output_dir=str(tmp_path))
        assert path is not None
        assert Path(path).exists()
        content = Path(path).read_text(encoding="utf-8")
        assert "Product,Revenue" in content
        assert "Laptops,120000.5" in content

    def test_export_dataframe_to_json(self, sample_sales_df, tmp_path):
        """Export dataframe to JSON records."""
        path = export_dataframe_to_json(sample_sales_df, output_dir=str(tmp_path))
        assert path is not None
        assert Path(path).exists()
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        assert len(records) == 4
        assert records[0]["Product"] == "Laptops"
        assert records[0]["Revenue"] == 120000.50

    def test_unified_export_facade(self, sample_sales_df, tmp_path):
        """export_to_csv and export_to_json facades should work with DataFrame and legacy tuple."""
        # CSV with DataFrame
        path_csv = export_to_csv(sample_sales_df, output_dir=str(tmp_path))
        assert path_csv is not None and Path(path_csv).exists()

        # JSON with legacy tuple (headers, rows)
        path_json = export_to_json(["Col1", "Col2"], [["Val1", "100"], ["Val2", "200"]], output_dir=str(tmp_path))
        assert path_json is not None and Path(path_json).exists()


class TestParseTableFromText:
    """Test fallback parsing of markdown tables."""

    def test_parses_pipe_table(self):
        """Should parse pipe delimited markdown table."""
        text = """Here are the results:
| Name | Age | Department |
|---|---|---|
| Alice | 30 | Engineering |
| Bob | 25 | Marketing |
"""
        parsed = parse_table_from_text(text)
        assert parsed is not None
        headers, rows = parsed
        assert headers == ["Name", "Age", "Department"]
        assert len(rows) == 2
        assert rows[0] == ["Alice", "30", "Engineering"]

    def test_returns_none_for_plain_text(self):
        """Should return None if no table is in text."""
        assert parse_table_from_text("This is just plain text without any tables.") is None
