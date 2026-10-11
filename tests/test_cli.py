"""Tests for the command line entry point.

An MCP server is launched as a subprocess from a client's config file, so startup failures are the
ones a user actually meets -- and they meet them as a client that silently will not connect. Each
of these produces a clear message on stderr and a non-zero exit instead.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mcp_sqlite.__main__ import main


class TestStartupFailures:
    def test_a_missing_database_exits_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main([str(tmp_path / "absent.db")]) == 2
        assert "no such database file" in capsys.readouterr().err

    def test_a_non_database_file_exits_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        decoy = tmp_path / "notes.txt"
        decoy.write_text("not a database")
        assert main([str(decoy)]) == 2
        assert "not a SQLite database" in capsys.readouterr().err

    def test_no_argument_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            main([])


class TestSuccessfulRun:
    def test_serves_until_stdin_closes(
        self,
        database_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # End of file on stdin is the protocol's graceful shutdown signal, and the only portable
        # one -- honouring it is what saves the client from having to kill the process.
        import io

        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        assert main([str(database_path)]) == 0

    def test_startup_logging_goes_to_stderr_only(
        self,
        database_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Anything on stdout that is not an MCP message corrupts the stream, and the client's parse
        # error would give no hint where it came from.
        import io

        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        main([str(database_path)])

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "read-only" in captured.err

    def test_a_request_is_answered_on_stdout(
        self,
        database_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import io
        import json

        from mcp_sqlite import protocol

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": {
                "_meta": {
                    protocol.META_PROTOCOL_VERSION: protocol.MODERN_VERSION,
                    protocol.META_CLIENT_CAPABILITIES: {},
                }
            },
        }
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(request) + "\n"))
        assert main([str(database_path)]) == 0

        lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["result"]["supportedVersions"]

    def test_the_database_is_opened_read_only(self, database_path: Path) -> None:
        from mcp_sqlite.readonly import open_readonly

        connection = open_readonly(database_path)
        try:
            with pytest.raises(sqlite3.OperationalError, match="readonly database"):
                connection.execute("DELETE FROM artists")
        finally:
            connection.close()
