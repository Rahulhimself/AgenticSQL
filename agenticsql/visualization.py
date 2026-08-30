"""
Data visualization and structured data export engine.

Provides:
- DataFrame-native chart generation (bar, grouped bar, line, multi-series, pie/donut)
- Intelligent column type inference and auto chart detection
- Direct CSV and JSON export from pandas DataFrames
- Fallback parsing from Markdown text tables for backwards compatibility
"""

import re
import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Any

import pandas as pd

logger = logging.getLogger(__name__)


def dataframe_to_dict(df: pd.DataFrame) -> dict:
    """
    Convert a pandas DataFrame into a JSON-serializable dictionary with columns and rows.
    Safely coerces timestamps, NaN values, and non-primitive dtypes into standard JSON formats.
    """
    if df is None or df.empty:
        return {"columns": [], "rows": []}

    # Convert timestamps and non-standard types to JSON-safe primitives
    safe_df = df.copy()
    for col in safe_df.columns:
        if pd.api.types.is_datetime64_any_dtype(safe_df[col]):
            safe_df[col] = safe_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        elif pd.api.types.is_numeric_dtype(safe_df[col]):
            safe_df[col] = safe_df[col].apply(lambda x: None if pd.isna(x) else float(x) if isinstance(x, (float, int)) else x)
        else:
            safe_df[col] = safe_df[col].astype(str).replace({"nan": None, "None": None, "<NA>": None})

    return {
        "columns": [str(c) for c in safe_df.columns],
        "rows": safe_df.values.tolist(),
    }


def dataframe_to_csv(df: pd.DataFrame) -> str:
    """
    Convert a pandas DataFrame into a CSV string without index columns.
    """
    if df is None or df.empty:
        return ""
    return df.to_csv(index=False)


def dataframe_to_json(df: pd.DataFrame) -> str:
    """
    Convert a pandas DataFrame into a formatted JSON records string.
    """
    if df is None or df.empty:
        return "[]"
    data = dataframe_to_dict(df)
    records = [dict(zip(data["columns"], row)) for row in data["rows"]]
    return json.dumps(records, indent=2, ensure_ascii=False)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """
    Convert a pandas DataFrame into a clean Markdown table string.
    """
    if df is None or df.empty:
        return ""
    return df.to_markdown(index=False)


def is_chartable(df: pd.DataFrame) -> bool:
    """
    Determine if a DataFrame has suitable dimensions and numeric data for charting.

    Requires at least 2 columns with at least one numeric column.
    """
    if df is None or df.empty or len(df.columns) < 2 or len(df) == 0:
        return False
    numeric_cols = df.select_dtypes(include=["number"]).columns
    return len(numeric_cols) > 0


def parse_table_from_text(text: str) -> Optional[tuple[list[str], list[list[str]]]]:
    """
    Attempt to parse tabular data from agent output text (pipe-delimited tables).

    Args:
        text: Agent response text that may contain a Markdown table.

    Returns:
        A tuple of (headers, rows) if a table is found, or None.
    """
    if not text:
        return None

    lines = text.strip().split("\n")
    # Extract tables using regex with support for varying whitespace and optional markdown formatting
    
    table_lines: list[list[str]] = []

    for line in lines:
        stripped = line.strip()
        if "|" in stripped:
            cells = [c.strip() for c in stripped.split("|") if c.strip()]
            if len(cells) >= 2:
                table_lines.append(cells)

    if len(table_lines) < 2:
        return None

    # Filter out markdown separator lines (e.g., |---|---|)
    data_lines = [
        row for row in table_lines
        if not all(re.match(r"^[-:=]+$", cell) for cell in row)
    ]

    if len(data_lines) >= 2:
        return data_lines[0], data_lines[1:]

    return None


def _to_dataframe(
    data: Union[pd.DataFrame, tuple[list[str], list[list[str]]], dict, str, None]
) -> Optional[pd.DataFrame]:
    """Coerce various tabular data representations into a clean pandas DataFrame."""
    if data is None:
        return None

    if isinstance(data, pd.DataFrame):
        return data if not data.empty else None

    if isinstance(data, dict):
        cols = data.get("columns", [])
        rows = data.get("rows", [])
        if cols and rows:
            return pd.DataFrame(rows, columns=cols)
        return None

    if isinstance(data, (tuple, list)) and len(data) == 2:
        headers, rows = data
        if headers and rows:
            return pd.DataFrame(rows, columns=headers)
        return None

    if isinstance(data, str):
        parsed = parse_table_from_text(data)
        if parsed:
            headers, rows = parsed
            return pd.DataFrame(rows, columns=headers)

    return None


def save_chart_from_dataframe(
    df: pd.DataFrame,
    chart_type: str = "auto",
    title: str = "",
    output_dir: str = "exports",
) -> Optional[str]:
    """
    Generate and save a high-quality chart directly from a pandas DataFrame.

    Supports:
    - Multi-series grouped bar charts and line charts
    - Automatic column type detection (time series, categorical, numerical)
    - Auto-detection of optimal chart type based on data shape and types

    Args:
        df: The pandas DataFrame with query results.
        chart_type: 'auto', 'bar', 'line', 'pie', or 'scatter'.
        title: Custom chart title (optional).
        output_dir: Directory to save the PNG chart.

    Returns:
        Path to the saved chart image file, or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed — skipping chart generation.")
        return None

    if df is None or df.empty or len(df.columns) < 2:
        return None

    try:
        clean_df = df.copy()

        # Identify candidate label/index column and numeric columns
        # First check non-numeric columns
        non_numeric_cols = clean_df.select_dtypes(exclude=["number"]).columns.tolist()
        numeric_cols = clean_df.select_dtypes(include=["number"]).columns.tolist()

        # If numeric columns were read as strings, attempt conversion
        if not numeric_cols:
            for col in clean_df.columns[1:]:
                try:
                    converted = (
                        clean_df[col]
                        .astype(str)
                        .str.replace(",", "", regex=False)
                        .str.replace("$", "", regex=False)
                        .str.replace("₹", "", regex=False)
                        .str.strip()
                    )
                    clean_df[col] = pd.to_numeric(converted, errors="coerce")
                    if not clean_df[col].isna().all():
                        numeric_cols.append(col)
                except Exception:
                    pass

        if not numeric_cols:
            return None

        # Choose label column
        if non_numeric_cols:
            label_col = non_numeric_cols[0]
        else:
            label_col = clean_df.columns[0]
            if label_col in numeric_cols:
                numeric_cols.remove(label_col)

        if not numeric_cols:
            return None

        labels = clean_df[label_col].astype(str).tolist()

        # Check if label column represents dates/time-series
        is_timeseries = False
        try:
            pd.to_datetime(clean_df[label_col], errors="raise")
            is_timeseries = True
        except Exception:
            is_timeseries = False

        # Auto-detect chart type
        chosen_type = chart_type.lower()
        if chosen_type == "auto":
            if is_timeseries:
                chosen_type = "line"
            elif len(clean_df) <= 6 and len(numeric_cols) == 1:
                chosen_type = "pie"
            elif len(clean_df) > 20 and len(numeric_cols) == 1:
                chosen_type = "line"
            else:
                chosen_type = "bar"

        # Apply clean visual styling
        available_styles = plt.style.available
        if "seaborn-v0_8-whitegrid" in available_styles:
            plt.style.use("seaborn-v0_8-whitegrid")
        elif "seaborn-v0_8-darkgrid" in available_styles:
            plt.style.use("seaborn-v0_8-darkgrid")
        elif "ggplot" in available_styles:
            plt.style.use("ggplot")

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        chart_title = title or f"{', '.join(numeric_cols)} by {label_col}"

        # 1. Pie / Donut Chart
        if chosen_type == "pie" and len(numeric_cols) == 1:
            values = clean_df[numeric_cols[0]].fillna(0).tolist()
            if not any(v > 0 for v in values):
                plt.close(fig)
                return None
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 1)))
            wedges, texts, autotexts = ax.pie(
                values,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=140,
                wedgeprops=dict(width=0.6, edgecolor="white", linewidth=1.5),
            )
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_fontweight("bold")
            ax.set_aspect("equal")

        # 2. Line Chart (Multi-series supported)
        elif chosen_type == "line":
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(numeric_cols), 1)))
            for idx, col in enumerate(numeric_cols):
                vals = clean_df[col].fillna(0).tolist()
                ax.plot(labels, vals, marker="o", linewidth=2.2, label=col, color=colors[idx])
                if len(numeric_cols) == 1:
                    ax.fill_between(range(len(labels)), vals, alpha=0.15, color=colors[idx])

            ax.set_xlabel(label_col, fontsize=11, fontweight="bold")
            ax.set_ylabel("Values", fontsize=11, fontweight="bold")
            if len(numeric_cols) > 1:
                ax.legend(frameon=True)
            plt.xticks(rotation=45, ha="right")

        # 3. Bar Chart (Grouped multi-series supported)
        else:
            x = np.arange(len(labels))
            n_series = len(numeric_cols)
            width = 0.8 / max(n_series, 1)
            colors = plt.cm.viridis(np.linspace(0.2, 0.85, max(n_series, 1)))

            for i, col in enumerate(numeric_cols):
                vals = clean_df[col].fillna(0).tolist()
                offset = (i - n_series / 2) * width + width / 2
                rects = ax.bar(x + offset, vals, width, label=col, color=colors[i], edgecolor="white", linewidth=0.5)

                # Add value labels on bars if few rows
                if len(labels) <= 10 and n_series == 1:
                    ax.bar_label(rects, padding=3, fmt="%.1f" if any(isinstance(v, float) and not v.is_integer() for v in vals) else "%.0f")

            ax.set_xlabel(label_col, fontsize=11, fontweight="bold")
            ax.set_ylabel("Values", fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            if n_series > 1:
                ax.legend(frameon=True)

        ax.set_title(chart_title, fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()

        # Save to export folder
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"chart_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Chart successfully saved to %s", filename)
        return str(filename)

    except Exception as e:
        logger.error("Chart generation failed: %s", e, exc_info=True)
        return None


def export_dataframe_to_csv(df: pd.DataFrame, output_dir: str = "exports") -> Optional[str]:
    """
    Export a pandas DataFrame directly to a CSV file.

    Args:
        df: The pandas DataFrame.
        output_dir: Destination folder.

    Returns:
        Path to the saved CSV file, or None on failure.
    """
    if df is None or df.empty:
        return None

    try:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"export_{timestamp}.csv"

        df.to_csv(filename, index=False, encoding="utf-8")
        logger.info("CSV exported directly from DataFrame to %s", filename)
        return str(filename)

    except Exception as e:
        logger.error("CSV export failed: %s", e)
        return None


def export_dataframe_to_json(df: pd.DataFrame, output_dir: str = "exports") -> Optional[str]:
    """
    Export a pandas DataFrame directly to a JSON file (records format).

    Args:
        df: The pandas DataFrame.
        output_dir: Destination folder.

    Returns:
        Path to the saved JSON file, or None on failure.
    """
    if df is None or df.empty:
        return None

    try:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"export_{timestamp}.json"

        # Convert to records JSON
        data = dataframe_to_dict(df)
        records = [dict(zip(data["columns"], row)) for row in data["rows"]]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        logger.info("JSON exported directly from DataFrame to %s", filename)
        return str(filename)

    except Exception as e:
        logger.error("JSON export failed: %s", e)
        return None


# --- Backwards Compatible Facade Functions ---


def save_chart(
    data: Any,
    rows: Optional[list[list[str]]] = None,
    chart_type: str = "auto",
    output_dir: str = "exports",
) -> Optional[str]:
    """
    Generate and save a chart from either a DataFrame, structured dict, or (headers, rows) tuple.
    """
    if rows is not None and isinstance(data, list):
        # Legacy signature: save_chart(headers, rows)
        df = pd.DataFrame(rows, columns=data)
        return save_chart_from_dataframe(df, chart_type=chart_type, output_dir=output_dir)

    df = _to_dataframe(data)
    if df is not None:
        return save_chart_from_dataframe(df, chart_type=chart_type, output_dir=output_dir)

    return None


def export_to_csv(
    data: Any,
    rows: Optional[list[list[str]]] = None,
    output_dir: str = "exports",
) -> Optional[str]:
    """
    Export tabular data to CSV from DataFrame, dict, or (headers, rows).
    """
    if rows is not None and isinstance(data, list):
        df = pd.DataFrame(rows, columns=data)
        return export_dataframe_to_csv(df, output_dir=output_dir)

    df = _to_dataframe(data)
    if df is not None:
        return export_dataframe_to_csv(df, output_dir=output_dir)

    return None


def export_to_json(
    data: Any,
    rows: Optional[list[list[str]]] = None,
    output_dir: str = "exports",
) -> Optional[str]:
    """
    Export tabular data to JSON from DataFrame, dict, or (headers, rows).
    """
    if rows is not None and isinstance(data, list):
        df = pd.DataFrame(rows, columns=data)
        return export_dataframe_to_json(df, output_dir=output_dir)

    df = _to_dataframe(data)
    if df is not None:
        return export_dataframe_to_json(df, output_dir=output_dir)

    return None
