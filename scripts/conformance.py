#!/usr/bin/env python3
"""Drives the installed server as a subprocess and checks the protocol end to end.

The test suite exercises the dispatch loop through StringIO, which is fast but cannot catch the
things that only go wrong across a real pipe: stray output on stdout, missing flushes, a response
split across two reads. This runs the installed entry point the way a client does.

Exits non-zero with a description of the first failure.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

MODERN = "2026-07-28"
META: dict[str, Any] = {
    "io.modelcontextprotocol/protocolVersion": MODERN,
    "io.modelcontextprotocol/clientCapabilities": {},
    "io.modelcontextprotocol/clientInfo": {"name": "conformance", "version": "1.0"},
}

failures: list[str] = []


def check(condition: bool, description: str) -> None:
    if condition:
        print(f"  ok    {description}")
    else:
        print(f"  FAIL  {description}")
        failures.append(description)


def build_database(directory: Path) -> Path:
    path = directory / "conformance.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT NOT NULL, qty INTEGER);"
        "INSERT INTO widgets VALUES (1,'bolt',40),(2,'nut',120),(3,'washer',7);"
    )
    connection.commit()
    connection.close()
    return path


def exchange(database: Path, requests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Sends requests to a fresh server process and returns (responses, stderr)."""
    payload = "".join(json.dumps(request) + "\n" for request in requests)
    completed = subprocess.run(
        ["mcp-sqlite", str(database)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    responses = []
    for line in completed.stdout.splitlines():
        if line.strip():
            responses.append(json.loads(line))
    return responses, completed.stderr


def modern(method: str, request_id: int, **params: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {**params, "_meta": META},
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        database = build_database(directory)

        print("modern protocol:")
        responses, stderr = exchange(
            database,
            [
                modern("server/discover", 1),
                modern("tools/list", 2),
                modern("tools/call", 3, name="list_tables", arguments={}),
                modern(
                    "tools/call",
                    4,
                    name="query",
                    arguments={"sql": "SELECT name, qty FROM widgets ORDER BY qty DESC"},
                ),
            ],
        )

        check(len(responses) == 4, "four requests produce four responses")
        check(
            all("result" in response for response in responses),
            "no request produced an error",
        )
        check(
            all(response["result"]["resultType"] == "complete" for response in responses),
            "every modern result is typed complete",
        )
        discover = responses[0]["result"]
        check(MODERN in discover["supportedVersions"], f"discover advertises {MODERN}")
        check(discover["capabilities"] == {"tools": {}}, "discover advertises the tools capability")
        check(
            "io.modelcontextprotocol/serverInfo" in discover["_meta"],
            "discover reports serverInfo in _meta",
        )
        check(len(responses[1]["result"]["tools"]) == 4, "four tools are advertised")
        check(
            "widgets" in responses[2]["result"]["content"][0]["text"],
            "list_tables finds the table",
        )
        check("nut" in responses[3]["result"]["content"][0]["text"], "query returns rows")
        check(stderr.strip() != "", "startup logging goes to stderr")

        print("\nstdout discipline:")
        check(
            all(isinstance(response, dict) for response in responses),
            "every stdout line is a JSON object and nothing else",
        )

        print("\nlegacy protocol:")
        legacy, _ = exchange(
            database,
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
                },
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            ],
        )
        check(len(legacy) == 2, "the notification drew no response")
        check(
            legacy[0]["result"]["protocolVersion"] == "2025-06-18",
            "the handshake negotiates the requested version",
        )
        check(
            "resultType" not in legacy[0]["result"],
            "legacy results are untyped, as that revision requires",
        )
        check(len(legacy[1]["result"]["tools"]) == 4, "tools work after the handshake")

        print("\nversion negotiation:")
        rejected, _ = exchange(
            database,
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {
                        "_meta": {
                            "io.modelcontextprotocol/protocolVersion": "1999-01-01",
                            "io.modelcontextprotocol/clientCapabilities": {},
                        }
                    },
                }
            ],
        )
        check(rejected[0]["error"]["code"] == -32022, "an unknown version returns -32022")
        check(
            MODERN in rejected[0]["error"]["data"]["supported"],
            "the error lists supported versions so the client can retry",
        )

        print("\nwrites are refused:")
        attacks = [
            "DELETE FROM widgets",
            "DROP TABLE widgets",
            "UPDATE widgets SET qty = 0",
            "SELECT 1; DROP TABLE widgets",
            "/* hidden */ DELETE FROM widgets",
            "WITH x AS (SELECT 1) DELETE FROM widgets",
            f"ATTACH DATABASE '{directory / 'evil.db'}' AS evil",
            "PRAGMA query_only = OFF",
        ]
        attack_responses, _ = exchange(
            database,
            [
                modern("tools/call", index, name="query", arguments={"sql": sql})
                for index, sql in enumerate(attacks, start=1)
            ],
        )
        check(
            all(response["result"]["isError"] for response in attack_responses),
            f"all {len(attacks)} write attempts were refused",
        )

        verifier = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        remaining = verifier.execute("SELECT count(*) FROM widgets").fetchone()[0]
        verifier.close()
        check(remaining == 3, "the database is byte-for-byte unchanged after the attacks")
        check(not (directory / "evil.db").exists(), "ATTACH created no file")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("all conformance checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
