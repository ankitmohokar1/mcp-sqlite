"""Command line entry point.

Usage is a single required argument -- the database file -- because an MCP server is launched as a
subprocess by a client whose configuration is a command line. Anything that needed interactive
setup could not be launched that way.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from mcp_sqlite import protocol
from mcp_sqlite.readonly import open_readonly
from mcp_sqlite.server import serve


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-sqlite",
        description="Expose a SQLite database to an AI assistant over MCP, read-only.",
        epilog=(
            "Speaks MCP over stdio. Register it with a client -- for Claude Code:\n"
            "  claude mcp add sqlite -- mcp-sqlite /path/to/database.db"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("database", type=Path, help="Path to the SQLite database file.")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {protocol.SERVER_VERSION}"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    args = _build_parser().parse_args(argv)

    try:
        connection = open_readonly(args.database)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except sqlite3.DatabaseError as error:
        print(f"error: {args.database} is not a SQLite database ({error})", file=sys.stderr)
        return 2

    # Startup logging goes to stderr. Anything on stdout that is not an MCP message corrupts the
    # stream, and the client's parse error would give no hint of where it came from.
    print(
        f"{protocol.SERVER_NAME} {protocol.SERVER_VERSION} serving {args.database} (read-only)",
        file=sys.stderr,
        flush=True,
    )

    try:
        serve(connection, sys.stdin, sys.stdout)
    except KeyboardInterrupt:
        return 130
    finally:
        connection.close()

    # Reaching here means stdin hit end of file, which is the protocol's graceful shutdown signal.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
