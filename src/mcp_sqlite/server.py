"""The server: a stdio loop that dispatches JSON-RPC methods, in either protocol era.

The transport rules are short and strict, and two of them are easy to break by accident:

* Messages are **newline-delimited JSON** -- one per line, no embedded newlines. Not the
  ``Content-Length`` framing that the Language Server Protocol uses, which is the wrong answer most
  readily recalled.
* The server **MUST NOT** write anything to stdout that is not an MCP message. A stray ``print``
  while debugging corrupts the stream and the client's parse failure will not point anywhere near
  the cause. Logging goes to stderr, which the client is explicitly allowed to ignore.
* The server **SHOULD** exit when stdin reaches end of file. That is the only portable shutdown
  signal, and honouring it is what stops the client having to escalate to killing the process.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Any, TextIO

from mcp_sqlite import protocol
from mcp_sqlite.jsonrpc import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    JsonRpcError,
    Request,
    error_response,
    parse_message,
    success_response,
)
from mcp_sqlite.tools import ToolRegistry, build_registry

INSTRUCTIONS = (
    "Read-only access to a SQLite database. Call list_tables first to learn what exists, then "
    "describe_table for a table's shape, then query for data. Writes are refused by the server, "
    "not merely discouraged, so there is no need to be careful about that -- but results are "
    "capped, so aggregate in SQL rather than fetching rows and counting them yourself."
)


@dataclass
class Server:
    """Dispatches MCP methods against one read-only SQLite connection.

    Holds no per-connection protocol state beyond the era, which is what the current revision
    requires: every modern request carries its own version and capabilities, and the server is
    forbidden from inferring them from earlier requests.
    """

    connection: sqlite3.Connection
    registry: ToolRegistry = field(default_factory=build_registry)
    era: protocol.Era = protocol.Era.UNKNOWN

    # Captured at `initialize` for legacy clients only. A modern request brings its own, so this
    # stays None and nothing reads it.
    legacy_context: protocol.RequestContext | None = None

    def handle(self, request: Request) -> dict[str, Any] | None:
        """Handles one request.

        Returns:
            The response to write, or None for a notification -- which must never be answered, even
            to report that it was not understood.
        """
        if request.is_notification:
            self._handle_notification(request)
            return None

        assert request.id is not None  # Guaranteed by parse_message for non-notifications.

        try:
            result = self._dispatch(request)
        except JsonRpcError as error:
            return error_response(request.id, error)
        except sqlite3.Error as error:
            # A database failure during dispatch rather than inside a tool: the tool handlers turn
            # their own SQLite errors into tool results, so reaching here means something
            # structural, which the client should see as a server error.
            return error_response(
                request.id, JsonRpcError(INTERNAL_ERROR, f"Database error: {error}")
            )
        return success_response(request.id, result)

    # -- dispatch ------------------------------------------------------------------------------

    def _dispatch(self, request: Request) -> dict[str, Any]:
        method = request.method

        # `initialize` is the legacy handshake, and receiving it settles the era. It is the one
        # method that does not carry modern _meta, so it has to be checked before _meta is read.
        if method == "initialize":
            return self._handle_initialize(request)

        if method == "server/discover":
            return self._handle_discover(request)

        context = self._context_for(request)

        if method == "tools/list":
            return self._handle_tools_list(request, context)
        if method == "tools/call":
            return self._handle_tools_call(request, context)
        if method == "ping":
            # Trivial but specified, and a client health check failing would look like a server
            # crash rather than an unimplemented method.
            return self._result({})

        raise JsonRpcError(METHOD_NOT_FOUND, f"Method not found: {method}")

    def _context_for(self, request: Request) -> protocol.RequestContext:
        """Resolves the protocol metadata for a request, according to the era.

        Modern requests carry it; legacy requests rely on what ``initialize`` established. A request
        with modern ``_meta`` is treated as modern even if the era was previously unknown, which is
        how a dual-era server lets the client's first real request decide.
        """
        if self.era is protocol.Era.LEGACY:
            if self.legacy_context is None:
                # A legacy client that skipped the handshake. Rather than guess, say so: the
                # specification requires initialize before anything else in that era.
                raise JsonRpcError(
                    INVALID_PARAMS,
                    "Expected an 'initialize' request first, or modern per-request _meta carrying "
                    f"{protocol.META_PROTOCOL_VERSION!r}",
                )
            return self.legacy_context

        context = protocol.parse_modern_meta(request.params)
        protocol.require_supported_version(context.protocol_version)
        self.era = protocol.Era.MODERN
        return context

    # -- handlers ------------------------------------------------------------------------------

    def _handle_initialize(self, request: Request) -> dict[str, Any]:
        self.era = protocol.Era.LEGACY
        requested = request.params.get("protocolVersion")
        version = requested if isinstance(requested, str) else protocol.LEGACY_VERSIONS[0]

        client_capabilities = request.params.get("capabilities")
        client_info = request.params.get("clientInfo")
        self.legacy_context = protocol.RequestContext(
            protocol_version=version,
            client_capabilities=(
                client_capabilities if isinstance(client_capabilities, dict) else {}
            ),
            client_info=client_info if isinstance(client_info, dict) else None,
        )

        # No resultType here: a handshake-era client does not expect one, and the shape of an
        # InitializeResult is fixed by that revision.
        return protocol.legacy_initialize_result(version)

    def _handle_discover(self, request: Request) -> dict[str, Any]:
        """``server/discover`` -- mandatory, and the probe a dual-era client uses.

        The version in ``_meta`` is deliberately *not* required to be supported here. A client
        probing with a version this server does not implement still needs to learn which versions
        it does, and answering with an error that omits the list would leave it stuck.
        """
        return protocol.discover_result(INSTRUCTIONS)

    def _handle_tools_list(
        self, request: Request, context: protocol.RequestContext
    ) -> dict[str, Any]:
        # Pagination is part of tools/list, but with four tools there is nothing to page: omitting
        # nextCursor is how the protocol says "this is the last page".
        return self._result({"tools": self.registry.definitions()})

    def _handle_tools_call(
        self, request: Request, context: protocol.RequestContext
    ) -> dict[str, Any]:
        name = request.params.get("name")
        if not isinstance(name, str):
            raise JsonRpcError(INVALID_PARAMS, "Invalid params: 'name' must be a string")

        arguments = request.params.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise JsonRpcError(INVALID_PARAMS, "Invalid params: 'arguments' must be an object")

        tool = self.registry.get(name)
        if tool is None:
            # A protocol error, not a tool error: the model cannot fix an unknown tool name by
            # adjusting arguments, and the specification classifies it this way.
            available = ", ".join(definition["name"] for definition in self.registry.definitions())
            raise JsonRpcError(
                INVALID_PARAMS, f"Unknown tool: {name}. Available tools: {available}"
            )

        try:
            outcome = tool.handler(self.connection, arguments)
        except Exception as error:
            # Deliberately broad. A tool raising an unexpected exception is a bug in this server,
            # but letting it escape would end the request with a bare internal error and tell the
            # model nothing. Returning it as a tool error keeps the conversation alive and gives
            # the model something it can act on, while stderr keeps the detail for the operator.
            print(f"tool {name!r} raised: {error!r}", file=sys.stderr, flush=True)
            return self._result(
                {
                    "content": [
                        {"type": "text", "text": f"The {name} tool failed unexpectedly: {error}"}
                    ],
                    "isError": True,
                }
            )

        return self._result(outcome.to_payload())

    def _handle_notification(self, request: Request) -> None:
        """Notifications are absorbed.

        ``notifications/initialized`` is the legacy client confirming the handshake, and
        ``notifications/cancelled`` would matter for long-running work. Every tool here is a single
        bounded query, so there is nothing to cancel -- but a notification must still never draw a
        response, which is why this returns rather than raising for unknown methods.
        """
        if request.method not in {"notifications/initialized", "notifications/cancelled"}:
            print(f"ignoring unknown notification: {request.method}", file=sys.stderr, flush=True)

    def _result(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Wraps a payload for the current era.

        Modern results carry ``resultType`` and the server's identity; legacy ones must not, since
        that revision's schemas do not allow the extra fields. One call site, so the branch cannot
        be forgotten in half the handlers.
        """
        if self.era is protocol.Era.LEGACY:
            return payload
        return protocol.complete_result(payload)


def serve(connection: sqlite3.Connection, stdin: TextIO, stdout: TextIO) -> None:
    """Reads newline-delimited JSON-RPC from ``stdin`` and writes responses to ``stdout``.

    Streams are parameters rather than module globals so the whole loop is testable without a
    subprocess: a test drives it with StringIO and asserts on what came out.
    """
    server = Server(connection=connection)

    for line in stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = parse_message(line)
        except JsonRpcError as error:
            # The id is unknown for an unparseable message, which is the one case JSON-RPC allows a
            # null id in a response.
            _write(stdout, error_response(None, error))
            continue

        response = server.handle(request)
        if response is not None:
            _write(stdout, response)


def _write(stream: TextIO, payload: dict[str, Any]) -> None:
    """Writes one message as a single line.

    ``separators`` removes the spaces json adds by default, and ``ensure_ascii`` is left on so the
    output is pure ASCII regardless of what the database holds -- a newline inside a value would
    break the framing, and escaping everything non-ASCII makes that impossible by construction.
    """
    stream.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")
    stream.flush()
