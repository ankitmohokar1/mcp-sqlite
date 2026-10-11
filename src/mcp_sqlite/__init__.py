"""A read-only SQLite explorer exposed over the Model Context Protocol.

Gives an AI assistant four tools -- list tables, describe a table, run a read-only query, sample
rows -- against a SQLite file, over MCP's stdio transport. No runtime dependencies.

See :mod:`mcp_sqlite.readonly` for why "read-only" here means three enforcement layers rather than
a promise, and :mod:`mcp_sqlite.protocol` for why the server speaks two protocol eras.
"""

from mcp_sqlite.protocol import MODERN_VERSION, SERVER_NAME, SERVER_VERSION, SUPPORTED_VERSIONS
from mcp_sqlite.readonly import ReadOnlyViolationError, ensure_read_only, open_readonly
from mcp_sqlite.server import Server, serve
from mcp_sqlite.tools import ToolRegistry, build_registry

__all__ = [
    "MODERN_VERSION",
    "SERVER_NAME",
    "SERVER_VERSION",
    "SUPPORTED_VERSIONS",
    "ReadOnlyViolationError",
    "Server",
    "ToolRegistry",
    "build_registry",
    "ensure_read_only",
    "open_readonly",
    "serve",
]
