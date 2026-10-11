"""Tests for MCP protocol semantics and the dual-era handling."""

from __future__ import annotations

from typing import Any

import pytest

from mcp_sqlite import protocol
from mcp_sqlite.jsonrpc import (
    INVALID_PARAMS,
    MISSING_REQUIRED_CLIENT_CAPABILITY,
    UNSUPPORTED_PROTOCOL_VERSION,
    JsonRpcError,
)


def modern_meta(
    version: str = protocol.MODERN_VERSION,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        protocol.META_PROTOCOL_VERSION: version,
        protocol.META_CLIENT_CAPABILITIES: {},
    }
    meta.update(extra or {})
    return {"_meta": meta}


class TestPerRequestMetadata:
    """The modern revision carries version and capabilities on every request, not in a handshake."""

    def test_parses_a_well_formed_request(self) -> None:
        context = protocol.parse_modern_meta(
            modern_meta(extra={protocol.META_CLIENT_INFO: {"name": "c", "version": "1"}})
        )
        assert context.protocol_version == protocol.MODERN_VERSION
        assert context.client_capabilities == {}
        assert context.client_info == {"name": "c", "version": "1"}

    def test_client_info_is_optional(self) -> None:
        assert protocol.parse_modern_meta(modern_meta()).client_info is None

    @pytest.mark.parametrize(
        "params",
        [
            {},
            {"_meta": None},
            {"_meta": "not an object"},
            {"_meta": {}},
            {"_meta": {protocol.META_PROTOCOL_VERSION: protocol.MODERN_VERSION}},
            {"_meta": {protocol.META_CLIENT_CAPABILITIES: {}}},
            {
                "_meta": {
                    protocol.META_PROTOCOL_VERSION: 2026,
                    protocol.META_CLIENT_CAPABILITIES: {},
                }
            },
        ],
    )
    def test_missing_or_malformed_metadata_is_invalid_params(self, params: dict[str, Any]) -> None:
        # Specified as -32602 rather than a version error: the server cannot tell which version the
        # client meant, so it has nothing useful to offer as an alternative.
        with pytest.raises(JsonRpcError) as caught:
            protocol.parse_modern_meta(params)
        assert caught.value.code == INVALID_PARAMS


class TestVersionNegotiation:
    def test_accepts_every_advertised_version(self) -> None:
        for version in protocol.SUPPORTED_VERSIONS:
            protocol.require_supported_version(version)

    def test_rejects_an_unknown_version_and_lists_alternatives(self) -> None:
        # Listing the supported versions is the point: it lets the client retry successfully
        # instead of merely learning that it failed.
        with pytest.raises(JsonRpcError) as caught:
            protocol.require_supported_version("1999-01-01")

        error = caught.value
        assert error.code == UNSUPPORTED_PROTOCOL_VERSION
        assert error.data == {
            "supported": list(protocol.SUPPORTED_VERSIONS),
            "requested": "1999-01-01",
        }

    def test_the_modern_version_is_first(self) -> None:
        # Clients pick from this list; the newest should be the obvious choice.
        assert protocol.SUPPORTED_VERSIONS[0] == protocol.MODERN_VERSION


class TestClientCapabilities:
    def test_passes_when_declared(self) -> None:
        context = protocol.RequestContext(protocol.MODERN_VERSION, {"roots": {}})
        protocol.require_client_capability(context, "roots")

    def test_reports_what_was_missing(self) -> None:
        context = protocol.RequestContext(protocol.MODERN_VERSION, {})
        with pytest.raises(JsonRpcError) as caught:
            protocol.require_client_capability(context, "roots", "sampling")

        assert caught.value.code == MISSING_REQUIRED_CLIENT_CAPABILITY
        assert caught.value.data == {"requiredCapabilities": ["roots", "sampling"]}


class TestResultShapes:
    def test_a_modern_result_is_typed_and_identifies_the_server(self) -> None:
        result = protocol.complete_result({"tools": []})
        assert result["resultType"] == "complete"
        assert result["_meta"][protocol.META_SERVER_INFO] == {
            "name": protocol.SERVER_NAME,
            "version": protocol.SERVER_VERSION,
        }

    def test_discover_reports_versions_capabilities_and_identity(self) -> None:
        result = protocol.discover_result("how to use me")
        assert result["resultType"] == "complete"
        assert result["supportedVersions"] == list(protocol.SUPPORTED_VERSIONS)
        assert result["capabilities"] == {"tools": {}}
        assert result["instructions"] == "how to use me"
        assert protocol.META_SERVER_INFO in result["_meta"]

    def test_capabilities_do_not_promise_list_changed(self) -> None:
        # The tool set is fixed at startup from the database on the command line, so advertising
        # change notifications would promise something that can never happen.
        assert protocol.server_capabilities() == {"tools": {}}


class TestLegacyHandshake:
    def test_echoes_a_supported_version(self) -> None:
        result = protocol.legacy_initialize_result("2025-06-18")
        assert result["protocolVersion"] == "2025-06-18"
        assert result["serverInfo"]["name"] == protocol.SERVER_NAME

    def test_falls_back_to_the_newest_legacy_version(self) -> None:
        # Handshake-era negotiation expects the server to answer with a version it can speak rather
        # than refusing outright.
        result = protocol.legacy_initialize_result("1999-01-01")
        assert result["protocolVersion"] == protocol.LEGACY_VERSIONS[0]

    def test_a_legacy_result_is_not_typed(self) -> None:
        # resultType did not exist in that revision, and its schemas do not permit extra fields.
        assert "resultType" not in protocol.legacy_initialize_result("2025-06-18")
