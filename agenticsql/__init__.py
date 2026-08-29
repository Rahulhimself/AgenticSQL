"""AgenticSQL — Chat with your database using natural language."""

from .database import connect, connect_from_uri, normalize_connection_uri, get_schema_info

__version__ = "1.0.0"

__all__ = [
    "connect",
    "connect_from_uri",
    "normalize_connection_uri",
    "get_schema_info",
]

