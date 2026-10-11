"""Tests for the tool surface."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from mcp_sqlite.tools import (
    DEFAULT_MAX_ROWS,
    HARD_MAX_ROWS,
    MAX_CELL_CHARS,
    Tool,
    ToolRegistry,
    ToolResult,
    build_registry,
)


def call(connection: sqlite3.Connection, name: str, **arguments: object) -> ToolResult:
    tool = build_registry().get(name)
    assert tool is not None, f"no tool named {name}"
    return tool.handler(connection, dict(arguments))


def structured(result: ToolResult) -> Any:
    """The structured payload, asserted present.

    Returns ``Any`` deliberately: the payload is JSON whose shape differs per tool, so narrowing it
    further would mean a TypedDict per tool that the assertions would then have to satisfy.
    """
    assert result.structured is not None, "expected structured content"
    return result.structured


class TestRegistry:
    def test_exposes_the_expected_tools_in_a_fixed_order(self) -> None:
        # Order is deliberate: the protocol asks for deterministic ordering so clients can cache
        # the list and the tool block stays byte-identical, which is what lets a prompt cache hit.
        names = [definition["name"] for definition in build_registry().definitions()]
        assert names == ["list_tables", "describe_table", "query", "sample_rows"]

    def test_duplicate_names_are_rejected(self) -> None:
        registry = ToolRegistry()
        tool = Tool("t", "T", "d", {"type": "object"}, lambda c, a: ToolResult("x"))
        registry.add(tool)
        with pytest.raises(ValueError, match="duplicate tool name"):
            registry.add(tool)

    def test_every_tool_has_a_valid_input_schema(self) -> None:
        # The spec requires inputSchema to be a JSON Schema object and explicitly not null.
        for definition in build_registry().definitions():
            schema = definition["inputSchema"]
            assert isinstance(schema, dict)
            assert schema["type"] == "object"

    def test_tool_names_satisfy_the_specs_character_rules(self) -> None:
        for definition in build_registry().definitions():
            name = definition["name"]
            assert 1 <= len(name) <= 128
            assert all(char.isalnum() or char in "_-." for char in name)

    def test_every_tool_describes_itself(self) -> None:
        # The description is the only documentation the model gets; it picks a tool from this text.
        for definition in build_registry().definitions():
            assert definition["title"]
            assert len(definition["description"]) > 40


class TestListTables:
    def test_lists_tables_and_views(self, connection: sqlite3.Connection) -> None:
        result = call(connection, "list_tables")
        assert not result.is_error
        names = {entry["name"] for entry in structured(result)["objects"]}
        assert {"artists", "albums", "recent"} <= names

    def test_hides_sqlite_internals(self, connection: sqlite3.Connection) -> None:
        # sqlite_* tables are implementation detail and would just be noise in the model's context.
        result = call(connection, "list_tables")
        assert not any(
            entry["name"].startswith("sqlite_") for entry in structured(result)["objects"]
        )

    def test_distinguishes_views_from_tables(self, connection: sqlite3.Connection) -> None:
        objects = {
            e["name"]: e["type"] for e in structured(call(connection, "list_tables"))["objects"]
        }
        assert objects["recent"] == "view"
        assert objects["artists"] == "table"


class TestDescribeTable:
    def test_reports_columns_with_their_constraints(self, connection: sqlite3.Connection) -> None:
        result = call(connection, "describe_table", table="albums")
        assert not result.is_error

        columns = {column["name"]: column for column in structured(result)["columns"]}
        assert columns["id"]["primary_key"] is True
        assert columns["title"]["notnull"] is True
        assert columns["year"]["notnull"] is False
        assert columns["title"]["type"] == "TEXT"

    def test_reports_indexes_and_foreign_keys(self, connection: sqlite3.Connection) -> None:
        result = call(connection, "describe_table", table="albums")
        assert any(index["name"] == "idx_albums_year" for index in structured(result)["indexes"])
        assert structured(result)["foreign_keys"] == [
            {"column": "artist_id", "references_table": "artists", "references_column": "id"}
        ]

    def test_marks_unique_indexes(self, connection: sqlite3.Connection) -> None:
        result = call(connection, "describe_table", table="artists")
        unique = {index["name"]: index["unique"] for index in structured(result)["indexes"]}
        assert unique["idx_artists_name"] is True

    def test_an_unknown_table_lists_what_exists(self, connection: sqlite3.Connection) -> None:
        # A tool error naming the alternatives lets the model correct itself on the next turn,
        # which is the whole point of the isError channel.
        result = call(connection, "describe_table", table="nope")
        assert result.is_error
        assert "artists" in result.text

    def test_a_missing_argument_is_a_tool_error(self, connection: sqlite3.Connection) -> None:
        assert call(connection, "describe_table").is_error


class TestQuery:
    def test_runs_a_select(self, connection: sqlite3.Connection) -> None:
        result = call(connection, "query", sql="SELECT name FROM artists ORDER BY name")
        assert not result.is_error
        assert structured(result)["columns"] == ["name"]
        assert [row[0] for row in structured(result)["rows"]] == [
            "Aphex Twin",
            "Boards of Canada",
            "Radiohead",
        ]

    def test_binds_parameters(self, connection: sqlite3.Connection) -> None:
        result = call(
            connection,
            "query",
            sql="SELECT title FROM albums WHERE year > ? ORDER BY year",
            parameters=[1999],
        )
        assert [row[0] for row in structured(result)["rows"]] == ["Kid A", "In Rainbows"]

    def test_a_refused_statement_is_a_tool_error_not_a_crash(
        self, connection: sqlite3.Connection
    ) -> None:
        result = call(connection, "query", sql="DELETE FROM albums")
        assert result.is_error
        assert "Refused" in result.text

    def test_a_sqlite_error_is_reported_with_the_sql(self, connection: sqlite3.Connection) -> None:
        # Echoing the SQL back matters: the model needs to see what it actually sent to fix it.
        result = call(connection, "query", sql="SELECT nope FROM artists")
        assert result.is_error
        assert "SELECT nope FROM artists" in result.text

    def test_a_missing_sql_argument_is_a_tool_error(self, connection: sqlite3.Connection) -> None:
        assert call(connection, "query").is_error

    def test_bad_parameters_type_is_a_tool_error(self, connection: sqlite3.Connection) -> None:
        assert call(connection, "query", sql="SELECT 1", parameters="not a list").is_error

    def test_an_empty_result_is_not_an_error(self, connection: sqlite3.Connection) -> None:
        result = call(connection, "query", sql="SELECT * FROM artists WHERE id = -1")
        assert not result.is_error
        assert structured(result)["rows"] == []


class TestOutputBounds:
    """A tool result lands in a context window, so an unbounded one is a denial of service."""

    def test_truncation_is_reported_not_silent(self, connection: sqlite3.Connection) -> None:
        # Silently returning the first N of many invites the model to conclude something about the
        # whole table from a sample it believes is complete.
        result = call(connection, "query", sql="SELECT * FROM albums", max_rows=2)
        assert structured(result)["truncated"] is True
        assert len(structured(result)["rows"]) == 2
        assert "TRUNCATED" in result.text

    def test_an_exact_fit_is_not_reported_as_truncated(
        self, connection: sqlite3.Connection
    ) -> None:
        # The reason the implementation fetches max_rows + 1: a result of exactly max_rows is
        # indistinguishable from a truncated one if you only compare counts.
        result = call(connection, "query", sql="SELECT * FROM albums", max_rows=4)
        assert structured(result)["truncated"] is False
        assert len(structured(result)["rows"]) == 4

    def test_max_rows_cannot_be_raised_past_the_ceiling(
        self, connection: sqlite3.Connection
    ) -> None:
        # A limit the caller can lift is a default, not a limit.
        result = call(connection, "query", sql="SELECT * FROM albums", max_rows=10**9)
        assert structured(result)["max_rows"] == HARD_MAX_ROWS

    def test_a_nonsensical_max_rows_falls_back_to_the_default(
        self, connection: sqlite3.Connection
    ) -> None:
        for bad in ("lots", None, -5, True):
            result = call(connection, "query", sql="SELECT * FROM albums", max_rows=bad)
            assert structured(result)["max_rows"] in (DEFAULT_MAX_ROWS, 1)

    def test_long_values_are_truncated(self, connection: sqlite3.Connection) -> None:
        result = call(connection, "query", sql=f"SELECT hex(zeroblob({MAX_CELL_CHARS}))")
        assert "truncated" in result.text

    def test_blobs_are_summarised_not_inlined(self, connection: sqlite3.Connection) -> None:
        # Binary in a context window is pure waste; its length is the only part a model can use.
        result = call(connection, "query", sql="SELECT zeroblob(2048)")
        assert "2048 bytes of BLOB" in result.text
        assert structured(result)["rows"][0][0] == {"blob_bytes": 2048}

    def test_nulls_render_distinguishably(self, connection: sqlite3.Connection) -> None:
        result = call(connection, "query", sql="SELECT NULL AS n")
        assert "NULL" in result.text


class TestSampleRows:
    def test_returns_rows_from_a_table(self, connection: sqlite3.Connection) -> None:
        result = call(connection, "sample_rows", table="artists", max_rows=2)
        assert not result.is_error
        assert len(structured(result)["rows"]) == 2

    def test_works_on_a_view(self, connection: sqlite3.Connection) -> None:
        assert not call(connection, "sample_rows", table="recent").is_error

    def test_an_unknown_table_is_a_tool_error(self, connection: sqlite3.Connection) -> None:
        assert call(connection, "sample_rows", table="nope").is_error

    def test_a_hostile_table_name_cannot_inject(self, connection: sqlite3.Connection) -> None:
        # The name is resolved against sqlite_schema before it is ever interpolated, so a crafted
        # value simply fails to resolve rather than reaching the SQL.
        result = call(connection, "sample_rows", table='artists"; DROP TABLE artists; --')
        assert result.is_error
        assert connection.execute("SELECT count(*) FROM artists").fetchone()[0] == 3


class TestResultPayload:
    def test_a_result_carries_text_and_the_error_flag(self) -> None:
        payload = ToolResult("hello").to_payload()
        assert payload["content"] == [{"type": "text", "text": "hello"}]
        assert payload["isError"] is False

    def test_structured_content_is_included_when_present(self) -> None:
        payload = ToolResult("hi", structured={"a": 1}).to_payload()
        assert payload["structuredContent"] == {"a": 1}

    def test_structured_content_is_omitted_when_absent(self) -> None:
        assert "structuredContent" not in ToolResult("hi").to_payload()
