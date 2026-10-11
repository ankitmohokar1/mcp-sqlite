"""MCP protocol semantics: versions, per-request metadata, and the two protocol eras.

The 2026-07-28 revision changed the shape of MCP substantially, and the change is easy to get wrong
from memory:

* **There is no handshake.** Earlier revisions opened a session with an ``initialize`` request.
  The current protocol is stateless -- every request carries its own protocol version and client
  capabilities in ``_meta``, and the server accepts or rejects each one independently. A server may
  not infer anything from previous requests on the same connection.
* **``server/discover`` is mandatory.** It reports supported versions, capabilities and identity in
  one call, and doubles as the probe a dual-era client uses to tell the two eras apart.
* **Results are typed.** Every result carries a ``resultType`` field; ``"complete"`` means the
  normal case. A client talking to an older server treats an absent ``resultType`` as complete.

This server speaks both eras, which is a deliberate choice rather than completeness for its own
sake. A modern-only server would reject every client still built on the ``initialize`` handshake --
which today is most of them -- and a legacy-only server is writing to a deprecated protocol. The
cost is one branch at the dispatch boundary; the tool layer below it is shared and era-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from mcp_sqlite.jsonrpc import (
    INVALID_PARAMS,
    MISSING_REQUIRED_CLIENT_CAPABILITY,
    UNSUPPORTED_PROTOCOL_VERSION,
    JsonRpcError,
)

SERVER_NAME: Final = "mcp-sqlite-explorer"
SERVER_VERSION: Final = "1.0.0"

MODERN_VERSION: Final = "2026-07-28"
"""The current revision: stateless, per-request metadata, mandatory ``server/discover``."""

LEGACY_VERSIONS: Final = ("2025-11-25", "2025-06-18", "2025-03-26")
"""Handshake-based revisions this server still answers, newest first.

Kept because real clients are still built on them. Only the framing differs -- the tool surface is
identical -- so supporting them costs a branch rather than a parallel implementation.
"""

SUPPORTED_VERSIONS: Final = (MODERN_VERSION, *LEGACY_VERSIONS)

META_PROTOCOL_VERSION: Final = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_INFO: Final = "io.modelcontextprotocol/clientInfo"
META_CLIENT_CAPABILITIES: Final = "io.modelcontextprotocol/clientCapabilities"
META_SERVER_INFO: Final = "io.modelcontextprotocol/serverInfo"


class Era(Enum):
    """Which protocol generation a connection is speaking.

    Determined by how the client opens the conversation, and then fixed: a request carrying modern
    ``_meta`` selects :attr:`MODERN`, an ``initialize`` request selects :attr:`LEGACY`.
    """

    UNKNOWN = "unknown"
    MODERN = "modern"
    LEGACY = "legacy"


@dataclass(frozen=True)
class RequestContext:
    """What a single request tells the server about itself.

    On the modern protocol this is read from ``_meta`` on every request, because the server is
    forbidden from remembering it between requests. On the legacy protocol it is captured once at
    ``initialize`` and reused, which is exactly the statefulness the new revision removes.
    """

    protocol_version: str
    client_capabilities: dict[str, Any]
    client_info: dict[str, Any] | None = None

    def capability(self, name: str) -> Any | None:
        return self.client_capabilities.get(name)


def server_info() -> dict[str, str]:
    return {"name": SERVER_NAME, "version": SERVER_VERSION}


def server_capabilities() -> dict[str, Any]:
    """What this server offers.

    Only ``tools``, and without ``listChanged``: the tool set is fixed at startup from the database
    given on the command line, so advertising change notifications would promise something that can
    never happen.
    """
    return {"tools": {}}


def parse_modern_meta(params: dict[str, Any]) -> RequestContext:
    """Reads the per-request protocol metadata the modern revision requires.

    Raises:
        JsonRpcError: ``-32602`` if a required field is absent or the wrong type. The specification
            makes this a malformed request rather than a version error -- the server cannot tell
            which version the client meant, so it cannot helpfully list alternatives.
    """
    meta = params.get("_meta")
    if not isinstance(meta, dict):
        raise JsonRpcError(
            INVALID_PARAMS,
            f"Invalid params: _meta is required and must be an object, carrying "
            f"{META_PROTOCOL_VERSION!r} and {META_CLIENT_CAPABILITIES!r}",
        )

    version = meta.get(META_PROTOCOL_VERSION)
    if not isinstance(version, str):
        raise JsonRpcError(
            INVALID_PARAMS,
            f"Invalid params: _meta[{META_PROTOCOL_VERSION!r}] is required and must be a string",
        )

    capabilities = meta.get(META_CLIENT_CAPABILITIES)
    if not isinstance(capabilities, dict):
        raise JsonRpcError(
            INVALID_PARAMS,
            f"Invalid params: _meta[{META_CLIENT_CAPABILITIES!r}] is required and must be "
            f"an object",
        )

    client_info = meta.get(META_CLIENT_INFO)
    return RequestContext(
        protocol_version=version,
        client_capabilities=capabilities,
        client_info=client_info if isinstance(client_info, dict) else None,
    )


def require_supported_version(version: str) -> None:
    """Rejects a version this server does not implement.

    Raises:
        JsonRpcError: ``-32022``, carrying the versions that would work. Listing them is the point:
            it lets the client retry successfully instead of just failing.
    """
    if version not in SUPPORTED_VERSIONS:
        raise JsonRpcError(
            UNSUPPORTED_PROTOCOL_VERSION,
            "Unsupported protocol version",
            {"supported": list(SUPPORTED_VERSIONS), "requested": version},
        )


def require_client_capability(context: RequestContext, *names: str) -> None:
    """Rejects a request needing a client capability that was not declared.

    Unused by the tools here -- reading a database needs nothing from the client -- but part of the
    protocol surface, and the error shape is specified precisely enough that guessing it later
    would be a mistake.

    Raises:
        JsonRpcError: ``-32021`` listing what was missing.
    """
    missing = [name for name in names if name not in context.client_capabilities]
    if missing:
        raise JsonRpcError(
            MISSING_REQUIRED_CLIENT_CAPABILITY,
            "Missing required client capability",
            {"requiredCapabilities": missing},
        )


def complete_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Wraps a payload as a completed result.

    Adds the two fields every modern result carries: ``resultType``, which tells the client how to
    read the rest, and the server's identity under ``_meta``. Legacy clients ignore both, so one
    shape serves both eras.
    """
    result = {"resultType": "complete", **payload}
    meta = result.setdefault("_meta", {})
    if isinstance(meta, dict):
        meta[META_SERVER_INFO] = server_info()
    return result


def discover_result(instructions: str) -> dict[str, Any]:
    """The ``server/discover`` payload: supported versions, capabilities and identity."""
    return complete_result(
        {
            "supportedVersions": list(SUPPORTED_VERSIONS),
            "capabilities": server_capabilities(),
            "instructions": instructions,
        }
    )


def legacy_initialize_result(requested_version: str) -> dict[str, Any]:
    """The ``initialize`` payload for a handshake-era client.

    The echoed ``protocolVersion`` is the negotiated one: if the client asked for something this
    server does not implement, it is answered with the newest legacy revision instead of being
    refused, which is what the handshake-era negotiation expects. A client that cannot live with
    that is free to disconnect.
    """
    negotiated = requested_version if requested_version in LEGACY_VERSIONS else LEGACY_VERSIONS[0]
    return {
        "protocolVersion": negotiated,
        "capabilities": server_capabilities(),
        "serverInfo": server_info(),
    }
