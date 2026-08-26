"""
Data visualization and export utilities.

Provides auto-chart generation from query results, table parsing
from agent output text, and CSV/JSON export functionality.
"""

import re
import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def parse_table_from_text(text: str) -> Optional[tuple[list[str], list[list[str]]]]:
    """
    Attempt to parse tabular data from agent output text.

    Looks for pipe-delimited (|) table formatting commonly produced
    by LLM responses.

    Args:
        text: Agent response text that may contain a table.

    Returns:
        A tuple of (headers, rows) if a table is found, or None.
    """
    lines = text.strip().split("\n")

    # Collect lines that look like pipe-delimited table rows
    table_lines: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if "|" in stripped:
            cells = [c.strip() for c in stripped.split("|") if c.strip()]
            if len(cells) >= 2:
                table_lines.append(cells)

    if len(table_lines) < 2:
        return None

    # Filter out separator lines (e.g., |---|---|)
    data_lines = [
        row for row in table_lines
        if not all(re.match(r"^[-:=]+$", cell) for cell in row)
    ]

    if len(data_lines) >= 2:
        return data_lines[0], data_lines[1:]

    return None


def save_chart(
    headers: list[str],
    rows: list[list[str]],
    chart_type: str = "auto",
    output_dir: str = "exports",
) -> Optional[str]:
    """
    Generate and save a chart from tabular data.

    Auto-detects the best chart type based on data shape:
    - ≤6 rows → pie chart
    - ≤20 rows → bar chart
    - >20 rows → line chart

    Args:
        headers: Column header names.
        rows: Data rows (list of lists of strings).
        chart_type: 'bar', 'line', 'pie', or 'auto'.
        output_dir: Directory to save the chart image.

    Returns:
        Path to the saved chart PNG, or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend for headless use
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping chart generation.")
        return None

    if len(headers) < 2 or len(rows) < 1:
        return None

    try:
        labels = [row[0] for row in rows]

        # Parse numeric values from the second column
        values: list[float] = []
        for row in rows:
            try:
                raw = row[1].replace(",", "").replace("$", "").replace("₹", "").strip()
                values.append(float(raw))
            except (ValueError, IndexError):
                values.append(0)

        # Skip if all values are zero
        if not any(v != 0 for v in values):
            return None

        # Auto-detect chart type
        if chart_type == "auto":
            if len(rows) <= 6:
                chart_type = "pie"
            elif len(rows) <= 20:
                chart_type = "bar"
            else:
                chart_type = "line"

        # Choose a style
        available_styles = plt.style.available
        if "seaborn-v0_8-darkgrid" in available_styles:
            plt.style.use("seaborn-v0_8-darkgrid")
        elif "ggplot" in available_styles:
            plt.style.use("ggplot")

        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "bar":
            colors = plt.cm.viridis([i / max(len(labels), 1) for i in range(len(labels))])
            ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_xlabel(headers[0], fontsize=12)
            ax.set_ylabel(headers[1], fontsize=12)
            plt.xticks(rotation=45, ha="right")

        elif chart_type == "line":
            ax.plot(labels, values, marker="o", linewidth=2, markersize=6, color="#4A90D9")
            ax.fill_between(range(len(labels)), values, alpha=0.15, color="#4A90D9")
            ax.set_xlabel(headers[0], fontsize=12)
            ax.set_ylabel(headers[1], fontsize=12)
            plt.xticks(rotation=45, ha="right")

        elif chart_type == "pie":
            colors = plt.cm.Set3([i / max(len(labels), 1) for i in range(len(labels))])
            ax.pie(values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
            ax.set_aspect("equal")

        ax.set_title(f"{headers[1]} by {headers[0]}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"chart_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Chart saved to %s", filename)
        return str(filename)

    except Exception as e:
        logger.error("Chart generation failed: %s", e)
        return None


def export_to_csv(
    headers: list[str],
    rows: list[list[str]],
    output_dir: str = "exports",
) -> Optional[str]:
    """
    Export tabular data to a CSV file.

    Args:
        headers: Column header names.
        rows: Data rows.
        output_dir: Directory to save the CSV file.

    Returns:
        Path to the saved CSV file, or None on failure.
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"export_{timestamp}.csv"

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        logger.info("CSV exported to %s", filename)
        return str(filename)

    except Exception as e:
        logger.error("CSV export failed: %s", e)
        return None


def export_to_json(
    headers: list[str],
    rows: list[list[str]],
    output_dir: str = "exports",
) -> Optional[str]:
    """
    Export tabular data to a JSON file.

    Each row becomes a dict keyed by header names.

    Args:
        headers: Column header names.
        rows: Data rows.
        output_dir: Directory to save the JSON file.

    Returns:
        Path to the saved JSON file, or None on failure.
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"export_{timestamp}.json"

        data = [dict(zip(headers, row)) for row in rows]
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("JSON exported to %s", filename)
        return str(filename)

    except Exception as e:
        logger.error("JSON export failed: %s", e)
        return None
