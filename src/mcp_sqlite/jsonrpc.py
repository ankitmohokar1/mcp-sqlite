"""JSON-RPC 2.0 message handling, and the error codes MCP layers on top of it.

Kept separate from the MCP semantics in :mod:`mcp_sqlite.protocol` because they are genuinely
different concerns: this module knows what a well-formed JSON-RPC envelope looks like, and nothing
about tools, versions or capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

JSONRPC_VERSION = "2.0"

# Standard JSON-RPC 2.0 codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# MCP-specific codes. The specification reserves -32020..-32099 for itself and requires that an
# implementation emit *only* codes it defines, with their defined meanings -- so this list is
# exhaustive rather than a starting point, and inventing a code in that range would be a spec
# violation. Anything application-specific belongs outside -32768..-32000 entirely.
HEADER_MISMATCH = -32020
MISSING_REQUIRED_CLIENT_CAPABILITY = -32021
UNSUPPORTED_PROTOCOL_VERSION = -32022


class JsonRpcError(Exception):
    """An error that should be returned to the peer as a JSON-RPC error response.

    Raising rather than returning lets a handler fail from wherever it notices the problem, instead
    of threading an error value back up through every layer.
    """

    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_error_object(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return error


@dataclass(frozen=True)
class Request:
    """A parsed JSON-RPC request or notification.

    A notification is a request with no ``id``. The distinction matters at exactly one point -- a
    notification must never be answered -- so it is captured as a property rather than a separate
    type.
    """

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str | int | None = None
    is_notification: bool = False


def parse_message(raw: str) -> Request:
    """Parses one line of input into a request.

    Raises:
        JsonRpcError: with ``PARSE_ERROR`` for malformed JSON, or ``INVALID_REQUEST`` for JSON that
            is not a valid JSON-RPC request object.
    """
    import json

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JsonRpcError(PARSE_ERROR, f"Parse error: {exc}") from exc

    if not isinstance(payload, dict):
        raise JsonRpcError(INVALID_REQUEST, "Invalid Request: expected a JSON object")

    if payload.get("jsonrpc") != JSONRPC_VERSION:
        raise JsonRpcError(
            INVALID_REQUEST,
            f"Invalid Request: jsonrpc must be {JSONRPC_VERSION!r}",
        )

    method = payload.get("method")
    if not isinstance(method, str) or not method:
        raise JsonRpcError(INVALID_REQUEST, "Invalid Request: method must be a non-empty string")

    params = payload.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        # MCP only ever uses by-name parameters, so a positional array is malformed here even though
        # plain JSON-RPC would allow it.
        raise JsonRpcError(INVALID_PARAMS, "Invalid params: expected an object")

    # An id of null is legal JSON-RPC but explicitly forbidden by MCP, and it is not the same thing
    # as an absent id: absent means notification, null means a malformed request.
    if "id" in payload:
        identifier = payload["id"]
        if identifier is None:
            raise JsonRpcError(INVALID_REQUEST, "Invalid Request: id must not be null")
        if not isinstance(identifier, str | int) or isinstance(identifier, bool):
            raise JsonRpcError(INVALID_REQUEST, "Invalid Request: id must be a string or integer")
        return Request(method=method, params=params, id=identifier, is_notification=False)

    return Request(method=method, params=params, id=None, is_notification=True)


def success_response(request_id: str | int, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}


def error_response(request_id: str | int | None, error: JsonRpcError) -> dict[str, Any]:
    """Builds an error response.

    ``id`` is included even when null, which is the one case JSON-RPC permits: a request so
    malformed that its id could not be read.
    """
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": error.to_error_object()}
