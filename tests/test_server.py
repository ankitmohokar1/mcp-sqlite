"""End-to-end tests driving the real stdio loop.

These feed newline-delimited JSON through :func:`mcp_sqlite.server.serve` and assert on what comes
out, rather than calling handlers directly. Framing, dispatch, era selection, and which failures
become JSON-RPC errors versus tool results are exactly the parts that break without any single
unit test noticing.
"""

from __future__ import annotations

import io
import json
import sqlite3
from typing import Any

import pytest

from mcp_sqlite import protocol
from mcp_sqlite.jsonrpc import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    UNSUPPORTED_PROTOCOL_VERSION,
)
from mcp_sqlite.server import serve

MODERN_META: dict[str, Any] = {
    protocol.META_PROTOCOL_VERSION: protocol.MODERN_VERSION,
    protocol.META_CLIENT_CAPABILITIES: {},
    protocol.META_CLIENT_INFO: {"name": "test-client", "version": "1.0"},
}


def run(connection: sqlite3.Connection, *messages: dict[str, Any] | str) -> list[dict[str, Any]]:
    """Feeds messages through the server and returns the parsed responses."""
    lines = [m if isinstance(m, str) else json.dumps(m) for m in messages]
    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = io.StringIO()
    serve(connection, stdin, stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


def modern(method: str, request_id: int = 1, **params: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {**params, "_meta": MODERN_META},
    }


class TestFraming:
    def test_one_response_per_line(self, connection: sqlite3.Connection) -> None:
        stdin = io.StringIO(
            json.dumps(modern("tools/list", 1)) + "\n" + json.dumps(modern("ping", 2)) + "\n"
        )
        stdout = io.StringIO()
        serve(connection, stdin, stdout)

        lines = stdout.getvalue().splitlines()
        assert len(lines) == 2
        assert all(json.loads(line) for line in lines)

    def test_no_response_contains_an_embedded_newline(self, connection: sqlite3.Connection) -> None:
        # The framing requirement that breaks silently: a newline inside a value would split one
        # message into two unparseable halves.
        responses = run(
            connection,
            modern("tools/call", 1, name="query", arguments={"sql": "SELECT char(10) AS nl"}),
        )
        stdout_line = json.dumps(responses[0])
        assert "\n" not in stdout_line

    def test_blank_lines_are_skipped(self, connection: sqlite3.Connection) -> None:
        stdin = io.StringIO("\n\n" + json.dumps(modern("ping")) + "\n\n")
        stdout = io.StringIO()
        serve(connection, stdin, stdout)
        assert len(stdout.getvalue().splitlines()) == 1

    def test_unparseable_input_gets_an_error_with_a_null_id(
        self, connection: sqlite3.Connection
    ) -> None:
        responses = run(connection, "{not json")
        assert responses[0]["id"] is None
        assert responses[0]["error"]["code"] == PARSE_ERROR

    def test_the_loop_survives_a_bad_message_and_keeps_going(
        self, connection: sqlite3.Connection
    ) -> None:
        # One malformed line must not take down the connection; the client would see the server
        # die rather than an error it could report.
        responses = run(connection, "{bad", modern("ping", 2))
        assert len(responses) == 2
        assert responses[0]["error"]["code"] == PARSE_ERROR
        assert "result" in responses[1]

    def test_a_notification_draws_no_response(self, connection: sqlite3.Connection) -> None:
        # Answering a notification is a protocol violation, including to say it was not understood.
        responses = run(
            connection,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "method": "notifications/something/unknown"},
        )
        assert responses == []


class TestModernEra:
    def test_discover_works_without_any_prior_request(self, connection: sqlite3.Connection) -> None:
        # The protocol is stateless: there is no handshake to perform first.
        result = run(connection, modern("server/discover"))[0]["result"]
        assert result["resultType"] == "complete"
        assert result["supportedVersions"] == list(protocol.SUPPORTED_VERSIONS)
        assert result["capabilities"] == {"tools": {}}
        assert result["_meta"][protocol.META_SERVER_INFO]["name"] == protocol.SERVER_NAME
        assert result["instructions"]

    def test_tools_list_works_without_any_prior_request(
        self, connection: sqlite3.Connection
    ) -> None:
        result = run(connection, modern("tools/list"))[0]["result"]
        assert [tool["name"] for tool in result["tools"]] == [
            "list_tables",
            "describe_table",
            "query",
            "sample_rows",
        ]

    def test_no_next_cursor_when_there_is_one_page(self, connection: sqlite3.Connection) -> None:
        # Omitting nextCursor is how the protocol says "last page"; sending a null one would be a
        # different claim.
        assert "nextCursor" not in run(connection, modern("tools/list"))[0]["result"]

    def test_every_modern_result_is_typed(self, connection: sqlite3.Connection) -> None:
        for method in ("server/discover", "tools/list", "ping"):
            result = run(connection, modern(method))[0]["result"]
            assert result["resultType"] == "complete", method

    def test_a_tool_call_returns_content(self, connection: sqlite3.Connection) -> None:
        result = run(connection, modern("tools/call", name="list_tables", arguments={}))[0][
            "result"
        ]
        assert result["isError"] is False
        assert result["content"][0]["type"] == "text"
        assert "artists" in result["content"][0]["text"]

    def test_absent_arguments_are_treated_as_empty(self, connection: sqlite3.Connection) -> None:
        result = run(connection, modern("tools/call", name="list_tables"))[0]["result"]
        assert result["isError"] is False

    def test_missing_meta_is_invalid_params(self, connection: sqlite3.Connection) -> None:
        responses = run(connection, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert responses[0]["error"]["code"] == INVALID_PARAMS

    def test_an_unsupported_version_lists_alternatives(
        self, connection: sqlite3.Connection
    ) -> None:
        responses = run(
            connection,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {
                    "_meta": {
                        protocol.META_PROTOCOL_VERSION: "1999-01-01",
                        protocol.META_CLIENT_CAPABILITIES: {},
                    }
                },
            },
        )
        error = responses[0]["error"]
        assert error["code"] == UNSUPPORTED_PROTOCOL_VERSION
        assert protocol.MODERN_VERSION in error["data"]["supported"]

    def test_discover_answers_even_for_an_unsupported_version(
        self, connection: sqlite3.Connection
    ) -> None:
        # The probe has to work, or a client asking "what do you speak?" with the wrong version
        # gets an error instead of the list it needs to recover.
        responses = run(
            connection,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "server/discover",
                "params": {
                    "_meta": {
                        protocol.META_PROTOCOL_VERSION: "1999-01-01",
                        protocol.META_CLIENT_CAPABILITIES: {},
                    }
                },
            },
        )
        assert "result" in responses[0]
        assert responses[0]["result"]["supportedVersions"]


class TestLegacyEra:
    def test_the_handshake_is_answered(self, connection: sqlite3.Connection) -> None:
        result = run(
            connection,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "legacy", "version": "0.1"},
                },
            },
        )[0]["result"]

        assert result["protocolVersion"] == "2025-06-18"
        assert result["serverInfo"]["name"] == protocol.SERVER_NAME
        # resultType did not exist in that revision and its schemas reject extra fields.
        assert "resultType" not in result

    def test_tools_work_after_the_handshake_without_per_request_meta(
        self, connection: sqlite3.Connection
    ) -> None:
        responses = run(
            connection,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "list_tables", "arguments": {}},
            },
        )

        assert len(responses) == 3
        assert len(responses[1]["result"]["tools"]) == 4
        assert responses[2]["result"]["isError"] is False
        # Legacy results stay untyped throughout, not just at initialize.
        assert all("resultType" not in r["result"] for r in responses)

    def test_a_legacy_client_that_skips_the_handshake_is_told_so(
        self, connection: sqlite3.Connection
    ) -> None:
        # Reached by sending initialize and then a request with neither era's metadata.
        responses = run(
            connection,
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        assert "result" in responses[1]

    def test_an_unknown_requested_version_is_negotiated_down(
        self, connection: sqlite3.Connection
    ) -> None:
        result = run(
            connection,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "1999-01-01", "capabilities": {}},
            },
        )[0]["result"]
        assert result["protocolVersion"] == protocol.LEGACY_VERSIONS[0]


class TestErrorClassification:
    """Which failures are JSON-RPC errors and which are tool results.

    The distinction is in the spec and it matters in practice: tool errors are expected to reach
    the model so it can self-correct, while protocol errors are things the model cannot fix by
    adjusting arguments.
    """

    def test_an_unknown_method_is_method_not_found(self, connection: sqlite3.Connection) -> None:
        responses = run(connection, modern("tools/nonsense"))
        assert responses[0]["error"]["code"] == METHOD_NOT_FOUND

    def test_an_unknown_tool_is_a_protocol_error(self, connection: sqlite3.Connection) -> None:
        # Not a tool error: the model cannot fix a wrong tool name by changing arguments, and the
        # spec classifies it as a protocol failure.
        responses = run(connection, modern("tools/call", name="nope", arguments={}))
        assert responses[0]["error"]["code"] == INVALID_PARAMS
        assert "list_tables" in responses[0]["error"]["message"]

    def test_a_non_string_tool_name_is_a_protocol_error(
        self, connection: sqlite3.Connection
    ) -> None:
        responses = run(connection, modern("tools/call", name=42, arguments={}))
        assert responses[0]["error"]["code"] == INVALID_PARAMS

    def test_non_object_arguments_are_a_protocol_error(
        self, connection: sqlite3.Connection
    ) -> None:
        responses = run(connection, modern("tools/call", name="list_tables", arguments=[1, 2]))
        assert responses[0]["error"]["code"] == INVALID_PARAMS

    @pytest.mark.parametrize(
        "sql",
        ["DELETE FROM albums", "DROP TABLE artists", "SELECT 1; DROP TABLE artists"],
    )
    def test_a_refused_write_is_a_tool_error_so_the_model_sees_why(
        self, connection: sqlite3.Connection, sql: str
    ) -> None:
        result = run(connection, modern("tools/call", name="query", arguments={"sql": sql}))[0][
            "result"
        ]
        assert result["isError"] is True
        assert "Refused" in result["content"][0]["text"]

    def test_a_tool_raising_unexpectedly_becomes_a_tool_error(
        self, connection: sqlite3.Connection
    ) -> None:
        # A bug in a handler should not end the request with a bare internal error that tells the
        # model nothing; stderr keeps the detail for whoever is operating the server.
        from mcp_sqlite.server import Server
        from mcp_sqlite.tools import Tool, ToolRegistry

        def explode(conn: sqlite3.Connection, args: dict[str, Any]) -> Any:
            raise RuntimeError("deliberate")

        registry = ToolRegistry()
        registry.add(Tool("boom", "Boom", "always raises", {"type": "object"}, explode))

        server = Server(connection=connection, registry=registry)
        from mcp_sqlite.jsonrpc import parse_message

        response = server.handle(
            parse_message(json.dumps(modern("tools/call", name="boom", arguments={})))
        )
        assert response is not None
        assert response["result"]["isError"] is True
        assert "deliberate" in response["result"]["content"][0]["text"]


class TestWritesCannotGetThrough:
    """The end-to-end claim: a write attempted over the wire changes nothing."""

    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM artists",
            "DROP TABLE artists",
            "UPDATE artists SET name = 'hacked'",
            "INSERT INTO artists VALUES (99, 'x', 'y')",
            "ATTACH DATABASE '/tmp/evil.db' AS evil",
            "PRAGMA query_only = OFF",
            "/* hidden */ DELETE FROM artists",
            "WITH x AS (SELECT 1) DELETE FROM artists",
        ],
    )
    def test_the_database_is_unchanged(self, connection: sqlite3.Connection, sql: str) -> None:
        run(connection, modern("tools/call", name="query", arguments={"sql": sql}))

        assert connection.execute("SELECT count(*) FROM artists").fetchone()[0] == 3
        assert connection.execute("SELECT count(*) FROM albums").fetchone()[0] == 4
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
