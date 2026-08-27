"""
Unit tests for sqlglot AST Comparison and Normalization.
"""

import pytest
from agenticsql.eval.ast_matcher import SQLASTMatcher


@pytest.fixture
def matcher():
    return SQLASTMatcher(default_dialect="sqlite")


def test_syntax_validity_check(matcher):
    """Test valid vs invalid SQL syntax parsing."""
    valid_ast, err1 = matcher.parse_sql("SELECT name, age FROM users WHERE age > 21;")
    assert valid_ast is not None
    assert err1 is None

    invalid_ast, err2 = matcher.parse_sql("SELECT FROM WHERE ;;;")
    assert invalid_ast is None
    assert err2 is not None


def test_exact_canonical_match(matcher):
    """Test that identical or trivially formatted queries match 1.0."""
    q1 = "SELECT id, name FROM customers WHERE country = 'USA' ORDER BY name ASC;"
    q2 = "select id, name from customers where country = 'USA' order by name asc"

    res = matcher.compare(q1, q2)
    assert res.is_valid_syntax is True
    assert res.exact_ast_match is True
    assert res.ast_similarity_score == 1.0


def test_ast_count_normalization(matcher):
    """Test that COUNT(1) and COUNT(*) are normalized."""
    q1 = "SELECT COUNT(*) FROM orders WHERE order_status = 'completed';"
    q2 = "SELECT COUNT(1) FROM orders WHERE order_status = 'completed';"

    res = matcher.compare(q1, q2)
    assert res.is_valid_syntax is True
    assert res.exact_ast_match is True
    assert res.ast_similarity_score == 1.0


def test_extract_schema_entities(matcher):
    """Test extracting tables and columns from AST."""
    sql = "SELECT c.first_name, o.total_amount FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.order_status = 'completed';"
    ast, _ = matcher.parse_sql(sql)
    assert ast is not None
    tables, cols = matcher.extract_schema_entities(ast)
    assert "customers" in tables
    assert "orders" in tables
    assert any("first_name" in c for c in cols)
    assert any("total_amount" in c for c in cols)


def test_clause_similarity_on_partial_match(matcher):
    """Test clause breakdown and similarity score when WHERE predicates differ."""
    gold = "SELECT customer_id, first_name FROM customers WHERE country = 'USA' AND city = 'New York';"
    cand = "SELECT customer_id, first_name FROM customers WHERE country = 'USA' AND city = 'Chicago';"

    res = matcher.compare(cand, gold)
    assert res.is_valid_syntax is True
    assert res.exact_ast_match is False
    assert 0.4 <= res.ast_similarity_score <= 0.9
    assert res.clauses["select"].matched is True
    assert res.clauses["from_join"].matched is True
    assert res.clauses["where"].matched is False
