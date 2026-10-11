"""Tests for JSON-RPC envelope parsing."""

from __future__ import annotations

import pytest

from mcp_sqlite.jsonrpc import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    PARSE_ERROR,
    JsonRpcError,
    error_response,
    parse_message,
    success_response,
)


class TestParsing:
    def test_parses_a_request(self) -> None:
        request = parse_message('{"jsonrpc":"2.0","id":1,"method":"ping","params":{"a":1}}')
        assert request.method == "ping"
        assert request.id == 1
        assert request.params == {"a": 1}
        assert not request.is_notification

    def test_parses_a_notification(self) -> None:
        request = parse_message('{"jsonrpc":"2.0","method":"notifications/initialized"}')
        assert request.is_notification
        assert request.id is None

    def test_a_string_id_is_allowed(self) -> None:
        assert parse_message('{"jsonrpc":"2.0","id":"abc","method":"ping"}').id == "abc"

    def test_absent_params_becomes_an_empty_dict(self) -> None:
        assert parse_message('{"jsonrpc":"2.0","id":1,"method":"ping"}').params == {}

    def test_null_params_becomes_an_empty_dict(self) -> None:
        assert parse_message('{"jsonrpc":"2.0","id":1,"method":"ping","params":null}').params == {}

    def test_malformed_json_is_a_parse_error(self) -> None:
        with pytest.raises(JsonRpcError) as caught:
            parse_message("{not json")
        assert caught.value.code == PARSE_ERROR

    @pytest.mark.parametrize(
        "raw",
        [
            "[1,2,3]",
            '"a string"',
            '{"id":1,"method":"ping"}',
            '{"jsonrpc":"1.0","id":1,"method":"ping"}',
            '{"jsonrpc":"2.0","id":1}',
            '{"jsonrpc":"2.0","id":1,"method":""}',
            '{"jsonrpc":"2.0","id":1,"method":42}',
        ],
    )
    def test_invalid_envelopes_are_invalid_request(self, raw: str) -> None:
        with pytest.raises(JsonRpcError) as caught:
            parse_message(raw)
        assert caught.value.code == INVALID_REQUEST

    def test_a_null_id_is_rejected(self) -> None:
        # Legal in plain JSON-RPC, forbidden by MCP -- and importantly not the same as an absent
        # id, which means notification.
        with pytest.raises(JsonRpcError, match="must not be null"):
            parse_message('{"jsonrpc":"2.0","id":null,"method":"ping"}')

    def test_a_boolean_id_is_rejected(self) -> None:
        # bool is a subclass of int in Python, so this needs an explicit check or it slips through.
        with pytest.raises(JsonRpcError, match="string or integer"):
            parse_message('{"jsonrpc":"2.0","id":true,"method":"ping"}')

    def test_positional_params_are_rejected(self) -> None:
        # MCP only uses by-name parameters, so an array is malformed here even though plain
        # JSON-RPC permits it.
        with pytest.raises(JsonRpcError) as caught:
            parse_message('{"jsonrpc":"2.0","id":1,"method":"ping","params":[1,2]}')
        assert caught.value.code == INVALID_PARAMS


class TestResponses:
    def test_success_response_shape(self) -> None:
        assert success_response(1, {"ok": True}) == {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"ok": True},
        }

    def test_error_response_shape(self) -> None:
        response = error_response(1, JsonRpcError(-32602, "bad", {"detail": "x"}))
        assert response["error"] == {"code": -32602, "message": "bad", "data": {"detail": "x"}}

    def test_error_data_is_omitted_when_absent(self) -> None:
        assert "data" not in error_response(1, JsonRpcError(-32603, "boom"))["error"]

    def test_a_null_id_is_allowed_on_an_error(self) -> None:
        # The one case the spec permits: a request so malformed its id could not be read.
        assert error_response(None, JsonRpcError(PARSE_ERROR, "nope"))["id"] is None
