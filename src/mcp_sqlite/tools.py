"""The tool surface: what the model can actually ask the database.

Two things shape every tool here, and neither is about SQLite.

**Output is bounded, always.** A tool result goes into a language model's context window, so an
unbounded result is a denial of service against the thing calling you -- ``SELECT * FROM events``
on a real table does not return a large string, it returns a broken conversation. Every result here
caps rows, caps cell width, and *says so* when it truncates, because silently returning the first
100 of 4,000,000 rows invites the model to draw a conclusion from a sample it thinks is the whole
table.

**Errors are addressed to the model.** A failure that says what was wrong and what to try instead
lets the model self-correct on the next turn, which is the entire difference between a tool that
works and one that produces a loop. MCP draws this distinction in the protocol: a malformed request
is a JSON-RPC error, while a query that ran and failed is a *result* with ``isError`` set, and only
the second is expected to reach the model.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mcp_sqlite.readonly import ReadOnlyViolationError, ensure_read_only

DEFAULT_MAX_ROWS = 100
HARD_MAX_ROWS = 1000
"""Ceiling the caller cannot raise.

``max_rows`` is a tool argument so the model can ask for fewer, but not for more than this: a limit
the caller can lift is a default, not a limit, and the model has every incentive to lift it.
"""

MAX_CELL_CHARS = 500
"""Longest rendering of a single value. A BLOB column or a stored document would otherwise fill the
context from one row."""


@dataclass
class ToolResult:
    """What a tool hands back.

    ``structured_content`` is the machine-readable form and ``content`` the human- and
    model-readable one. The protocol asks for both when a tool has structured output -- the text
    block keeps older clients working, and in practice the model reads it more reliably than it
    reads a JSON blob.
    """

    text: str
    structured: Any | None = None
    is_error: bool = False

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "content": [{"type": "text", "text": self.text}],
            "isError": self.is_error,
        }
        if self.structured is not None:
            payload["structuredContent"] = self.structured
        return payload


@dataclass(frozen=True)
class Tool:
    """A tool definition plus the function that runs it."""

    name: str
    title: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[sqlite3.Connection, dict[str, Any]], ToolResult]
    output_schema: dict[str, Any] | None = None

    def to_definition(self) -> dict[str, Any]:
        """The wire form for ``tools/list``."""
        definition: dict[str, Any] = {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        if self.output_schema is not None:
            definition["outputSchema"] = self.output_schema
        return definition


@dataclass
class ToolRegistry:
    """The server's tools, in a fixed order.

    Order is deliberate: the protocol asks for a deterministic ordering so clients can cache the
    list and so the tool block stays byte-identical between requests, which is what lets a model
    provider's prompt cache hit.
    """

    tools: list[Tool] = field(default_factory=list)

    def add(self, tool: Tool) -> None:
        if any(existing.name == tool.name for existing in self.tools):
            raise ValueError(f"duplicate tool name: {tool.name}")
        self.tools.append(tool)

    def get(self, name: str) -> Tool | None:
        return next((tool for tool in self.tools if tool.name == name), None)

    def definitions(self) -> list[dict[str, Any]]:
        return [tool.to_definition() for tool in self.tools]


# -- rendering ---------------------------------------------------------------------------------


def _render_cell(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bytes):
        # Never inline binary. Its length is the only part a model can reason about.
        return f"<{len(value)} bytes of BLOB>"
    text = str(value)
    if len(text) > MAX_CELL_CHARS:
        return text[:MAX_CELL_CHARS] + f"... [truncated, {len(text)} chars total]"
    return text


def _render_table(columns: list[str], rows: list[tuple[Any, ...]]) -> str:
    """Formats rows as an aligned text table.

    Aligned rather than JSON because it is markedly more compact for tabular data -- JSON repeats
    every column name on every row, which for a 20-column result is most of the tokens spent on
    punctuation the model already knows.
    """
    if not rows:
        return "(no rows)"

    rendered = [[_render_cell(cell) for cell in row] for row in rows]
    widths = [len(column) for column in columns]
    for row in rendered:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], min(len(cell), MAX_CELL_CHARS))

    lines = [" | ".join(column.ljust(widths[i]) for i, column in enumerate(columns))]
    lines.append("-+-".join("-" * width for width in widths))
    lines.extend(
        " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rendered
    )
    return "\n".join(lines)


def _clamp_max_rows(arguments: dict[str, Any]) -> int:
    requested = arguments.get("max_rows", DEFAULT_MAX_ROWS)
    if not isinstance(requested, int) or isinstance(requested, bool):
        return DEFAULT_MAX_ROWS
    return max(1, min(requested, HARD_MAX_ROWS))


# -- handlers ----------------------------------------------------------------------------------


def _list_tables(connection: sqlite3.Connection, arguments: dict[str, Any]) -> ToolResult:
    rows = connection.execute(
        """
        SELECT name, type
        FROM sqlite_schema
        WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()

    if not rows:
        return ToolResult("The database contains no tables or views.", structured={"objects": []})

    objects = [{"name": row["name"], "type": row["type"]} for row in rows]
    listing = "\n".join(f"  {entry['type']:<5}  {entry['name']}" for entry in objects)
    return ToolResult(
        f"{len(objects)} object(s):\n{listing}\n\n"
        "Use describe_table for columns, indexes and foreign keys.",
        structured={"objects": objects},
    )


def _describe_table(connection: sqlite3.Connection, arguments: dict[str, Any]) -> ToolResult:
    name = arguments.get("table")
    if not isinstance(name, str) or not name:
        return ToolResult("The 'table' argument is required and must be a string.", is_error=True)

    # Resolved against sqlite_schema rather than interpolated into a PRAGMA, because a table name
    # cannot be a bound parameter. Looking it up first means the value reaching the PRAGMA is one
    # this database actually contains, which is the only safe way to build that statement.
    found = connection.execute(
        "SELECT name, type FROM sqlite_schema WHERE name = ? AND type IN ('table', 'view')",
        (name,),
    ).fetchone()
    if found is None:
        available = connection.execute(
            "SELECT name FROM sqlite_schema WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        names = ", ".join(row["name"] for row in available) or "(none)"
        return ToolResult(
            f"No table or view named {name!r}. Available: {names}",
            is_error=True,
        )

    real_name = found["name"]
    columns = connection.execute(f'PRAGMA table_info("{real_name}")').fetchall()
    indexes = connection.execute(f'PRAGMA index_list("{real_name}")').fetchall()
    foreign_keys = connection.execute(f'PRAGMA foreign_key_list("{real_name}")').fetchall()

    column_dicts = [
        {
            "name": column["name"],
            "type": column["type"] or "(untyped)",
            "notnull": bool(column["notnull"]),
            "primary_key": bool(column["pk"]),
            "default": column["dflt_value"],
        }
        for column in columns
    ]

    lines = [f"{found['type']} {real_name}", "", "Columns:"]
    for column in column_dicts:
        flags = []
        if column["primary_key"]:
            flags.append("PK")
        if column["notnull"]:
            flags.append("NOT NULL")
        if column["default"] is not None:
            flags.append(f"DEFAULT {column['default']}")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        lines.append(f"  {column['name']}: {column['type']}{suffix}")

    index_dicts = [{"name": index["name"], "unique": bool(index["unique"])} for index in indexes]
    if index_dicts:
        lines.extend(["", "Indexes:"])
        lines.extend(
            f"  {index['name']}{' (unique)' if index['unique'] else ''}" for index in index_dicts
        )

    fk_dicts = [
        {"column": fk["from"], "references_table": fk["table"], "references_column": fk["to"]}
        for fk in foreign_keys
    ]
    if fk_dicts:
        lines.extend(["", "Foreign keys:"])
        lines.extend(
            f"  {fk['column']} -> {fk['references_table']}.{fk['references_column']}"
            for fk in fk_dicts
        )

    return ToolResult(
        "\n".join(lines),
        structured={
            "name": real_name,
            "type": found["type"],
            "columns": column_dicts,
            "indexes": index_dicts,
            "foreign_keys": fk_dicts,
        },
    )


def _query(connection: sqlite3.Connection, arguments: dict[str, Any]) -> ToolResult:
    sql = arguments.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        return ToolResult("The 'sql' argument is required and must be a string.", is_error=True)

    parameters = arguments.get("parameters", [])
    if not isinstance(parameters, list):
        return ToolResult(
            "The 'parameters' argument must be a list of values bound to ? placeholders.",
            is_error=True,
        )

    try:
        ensure_read_only(sql)
    except ReadOnlyViolationError as violation:
        # A refusal, not a crash -- returned as a tool error so the model sees the reason and can
        # rewrite the query rather than retrying the same one.
        return ToolResult(f"Refused: {violation}", is_error=True)

    max_rows = _clamp_max_rows(arguments)

    try:
        # Fetch one more than the cap so truncation can be reported as a fact rather than guessed
        # at by comparing the row count to the limit -- those are indistinguishable when a query
        # happens to return exactly max_rows.
        cursor = connection.execute(sql, tuple(parameters))
        rows = cursor.fetchmany(max_rows + 1)
    except sqlite3.Error as error:
        return ToolResult(
            f"SQLite rejected the query: {error}\n\nSQL was:\n{sql}",
            is_error=True,
        )

    if cursor.description is None:
        return ToolResult("The statement returned no result set.", structured={"rows": []})

    columns = [description[0] for description in cursor.description]
    truncated = len(rows) > max_rows
    visible = rows[:max_rows]

    header = f"{len(visible)} row(s)"
    if truncated:
        header += (
            f" -- TRUNCATED at max_rows={max_rows}. More rows match; this is a sample, not the "
            "full result. Re-run with a narrower WHERE, an ORDER BY, or an aggregate if you need "
            "to reason about the whole table."
        )

    return ToolResult(
        f"{header}\n\n{_render_table(columns, [tuple(row) for row in visible])}",
        structured={
            "columns": columns,
            "rows": [[_structured_cell(cell) for cell in row] for row in visible],
            "truncated": truncated,
            "max_rows": max_rows,
        },
    )


def _structured_cell(value: Any) -> Any:
    """Converts a value to something JSON can carry.

    Only BLOBs need help; everything SQLite stores natively is already a JSON type.
    """
    if isinstance(value, bytes):
        return {"blob_bytes": len(value)}
    return value


def _sample_rows(connection: sqlite3.Connection, arguments: dict[str, Any]) -> ToolResult:
    name = arguments.get("table")
    if not isinstance(name, str) or not name:
        return ToolResult("The 'table' argument is required and must be a string.", is_error=True)

    found = connection.execute(
        "SELECT name FROM sqlite_schema WHERE name = ? AND type IN ('table', 'view')",
        (name,),
    ).fetchone()
    if found is None:
        return ToolResult(f"No table or view named {name!r}.", is_error=True)

    limit = _clamp_max_rows(arguments)
    # The identifier is quoted and came from sqlite_schema, so it is a name this database holds.
    # The limit is an integer the code produced, not a string the caller supplied.
    return _query(
        connection,
        {"sql": f'SELECT * FROM "{found["name"]}" LIMIT {limit}', "max_rows": limit},
    )


# -- registry ----------------------------------------------------------------------------------

_MAX_ROWS_SCHEMA = {
    "type": "integer",
    "minimum": 1,
    "maximum": HARD_MAX_ROWS,
    "default": DEFAULT_MAX_ROWS,
    "description": (
        f"Maximum rows to return (default {DEFAULT_MAX_ROWS}, hard ceiling {HARD_MAX_ROWS}). "
        "Results above the ceiling are truncated and reported as such."
    ),
}


def build_registry() -> ToolRegistry:
    """Builds the tool set.

    Descriptions are written for a reader who cannot see the code, because that is the only
    documentation the model gets -- it chooses a tool from this text alone. Saying what a tool
    refuses to do is as useful as saying what it does: it stops the model trying.
    """
    registry = ToolRegistry()

    registry.add(
        Tool(
            name="list_tables",
            title="List tables and views",
            description=(
                "List every table and view in the database. Start here when you do not yet know "
                "the schema -- it is cheap and tells you what the other tools can be pointed at."
            ),
            input_schema={"type": "object", "additionalProperties": False},
            handler=_list_tables,
            output_schema={
                "type": "object",
                "properties": {
                    "objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string", "enum": ["table", "view"]},
                            },
                            "required": ["name", "type"],
                        },
                    }
                },
                "required": ["objects"],
            },
        )
    )

    registry.add(
        Tool(
            name="describe_table",
            title="Describe a table",
            description=(
                "Show one table's columns with their types, nullability, primary key and defaults, "
                "plus its indexes and foreign keys. Prefer this over 'SELECT *' for learning a "
                "shape: it returns the schema rather than data, so it costs the same whether the "
                "table holds ten rows or ten million."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Name of the table or view."}
                },
                "required": ["table"],
                "additionalProperties": False,
            },
            handler=_describe_table,
        )
    )

    registry.add(
        Tool(
            name="query",
            title="Run a read-only SQL query",
            description=(
                "Run a single read-only SQL query and return the rows. Only SELECT, WITH, VALUES, "
                "EXPLAIN and informational PRAGMA statements are permitted -- anything that could "
                "write, and anything with more than one statement, is refused with an explanation. "
                "Use ? placeholders and the 'parameters' array for values rather than building the "
                "SQL by hand. Results are capped; aggregate in SQL when you need a conclusion "
                "about a whole table rather than a sample of it."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single read-only SQL statement.",
                    },
                    "parameters": {
                        "type": "array",
                        "description": "Values bound to ? placeholders, in order.",
                        "items": {"type": ["string", "number", "boolean", "null"]},
                    },
                    "max_rows": _MAX_ROWS_SCHEMA,
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
            handler=_query,
        )
    )

    registry.add(
        Tool(
            name="sample_rows",
            title="Sample rows from a table",
            description=(
                "Return the first few rows of a table, to see what the data actually looks like. "
                "A shortcut for 'SELECT * FROM t LIMIT n' that does not require writing SQL. The "
                "rows are whatever order the database returns and are not a random sample."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Name of the table or view."},
                    "max_rows": _MAX_ROWS_SCHEMA,
                },
                "required": ["table"],
                "additionalProperties": False,
            },
            handler=_sample_rows,
        )
    )

    return registry
