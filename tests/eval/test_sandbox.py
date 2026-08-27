"""
Unit tests for DatabaseSandbox isolated instances and seed data.
"""

import pytest
import pandas as pd
from agenticsql.eval.sandbox import DatabaseSandbox, create_all_sandboxes


def test_ecommerce_sandbox_initialization():
    """Test e-commerce domain sandbox tables and data."""
    sb = DatabaseSandbox(domain="ecommerce")
    schema = sb.get_schema()
    assert "customers" in schema
    assert "orders" in schema
    assert "products" in schema
    assert "categories" in schema
    assert "order_items" in schema

    df, err = sb.execute_query("SELECT COUNT(*) AS cnt FROM customers;")
    assert err is None
    assert df is not None
    assert df.iloc[0]["cnt"] == 7
    sb.close()


def test_hr_sandbox_initialization():
    """Test HR & Payroll domain sandbox tables and data."""
    sb = DatabaseSandbox(domain="hr_payroll")
    schema = sb.get_schema()
    assert "employees" in schema
    assert "departments" in schema
    assert "salaries" in schema

    df, err = sb.execute_query("SELECT COUNT(*) AS cnt FROM employees;")
    assert err is None
    assert df is not None
    assert df.iloc[0]["cnt"] == 9
    sb.close()


def test_finance_sandbox_initialization():
    """Test Financial Ledger domain sandbox tables and data."""
    sb = DatabaseSandbox(domain="financial_ledger")
    schema = sb.get_schema()
    assert "accounts" in schema
    assert "transactions" in schema
    assert "audit_logs" in schema

    df, err = sb.execute_query("SELECT COUNT(*) AS cnt FROM transactions;")
    assert err is None
    assert df is not None
    assert df.iloc[0]["cnt"] == 7
    sb.close()


def test_sandbox_query_error_handling():
    """Test invalid syntax handling."""
    sb = DatabaseSandbox(domain="ecommerce")
    df, err = sb.execute_query("SELECT invalid_column_xyz FROM non_existent_table;")
    assert df is None
    assert err is not None
    assert "no such table" in err.lower()
    sb.close()


def test_create_all_sandboxes():
    """Test dictionary generation of all domain sandboxes."""
    sandboxes = create_all_sandboxes()
    assert "ecommerce" in sandboxes
    assert "hr_payroll" in sandboxes
    assert "financial_ledger" in sandboxes
    for sb in sandboxes.values():
        sb.close()
