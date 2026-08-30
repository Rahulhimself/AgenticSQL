"""
AgenticSQL — Chat with your database using natural language.
Provides secure multi-tenant SQL agent execution, AST guardrails, and dynamic database switching.
"""

from .database import connect, connect_from_uri, normalize_connection_uri, get_schema_info

__version__ = "1.0.0"

# Publicly exported database connection helpers
__all__ = [
    "connect",
    "connect_from_uri",
    "normalize_connection_uri",
    "get_schema_info",
]


