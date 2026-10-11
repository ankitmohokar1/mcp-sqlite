"""Tests for the three read-only enforcement layers.

The structure mirrors the design: one class per layer, each testing what that layer uniquely
covers, plus a class for the gaps that justify having three. If any one layer were removed, tests
in the "DefenceInDepth" class below would fail -- which is the only honest way to claim the layers
are not redundant.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mcp_sqlite.readonly import (
    ReadOnlyViolationError,
    ensure_read_only,
    open_readonly,
    split_statements,
    strip_sql_noise,
)


class TestConnectionLayer:
    """Layer one: the ``mode=ro`` URI, enforced by SQLite rather than by this code."""

    @pytest.mark.parametrize(
        "statement",
        [
            "INSERT INTO artists VALUES (99, 'Nobody', 'XX')",
            "UPDATE artists SET name = 'changed'",
            "DELETE FROM albums",
            "DROP TABLE albums",
            "CREATE TABLE sneaky (x INTEGER)",
            "ALTER TABLE artists ADD COLUMN sneaky TEXT",
        ],
    )
    def test_sqlite_itself_refuses_writes(
        self, connection: sqlite3.Connection, statement: str
    ) -> None:
        # Not going through the gate -- this is SQLite refusing, which is what makes this layer
        # worth having: it holds even for a statement the gate misclassified.
        with pytest.raises(sqlite3.OperationalError, match="readonly database"):
            connection.execute(statement)

    def test_reads_work(self, connection: sqlite3.Connection) -> None:
        assert connection.execute("SELECT count(*) FROM artists").fetchone()[0] == 3

    def test_a_missing_database_is_not_created(self, tmp_path: Path) -> None:
        # mode=ro will not create a file. That is the behaviour we want: a typo should fail loudly
        # rather than silently serve an empty database and leave the model drawing conclusions
        # from no data.
        with pytest.raises(FileNotFoundError):
            open_readonly(tmp_path / "absent.db")

    def test_a_non_database_file_fails_at_open(self, tmp_path: Path) -> None:
        # Caught at startup rather than on the first tool call, where it would surface to the model
        # as a baffling tool failure.
        decoy = tmp_path / "not.db"
        decoy.write_text("this is not a database")
        with pytest.raises(sqlite3.DatabaseError):
            open_readonly(decoy)

    def test_query_only_is_on(self, connection: sqlite3.Connection) -> None:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1


class TestStatementGate:
    """Layer three: the whitelist."""

    @pytest.mark.parametrize(
        "statement",
        [
            "SELECT 1",
            "select * from artists",
            "  SELECT name FROM artists  ",
            "WITH x AS (SELECT 1 AS n) SELECT n FROM x",
            "VALUES (1), (2)",
            "EXPLAIN SELECT * FROM albums",
            "EXPLAIN QUERY PLAN SELECT * FROM albums",
            "PRAGMA table_info(artists)",
            "PRAGMA table_list",
            "PRAGMA main.table_info(artists)",
        ],
    )
    def test_allows_reads(self, statement: str) -> None:
        ensure_read_only(statement)

    @pytest.mark.parametrize(
        "statement",
        [
            "INSERT INTO artists VALUES (9, 'x', 'y')",
            "UPDATE artists SET name = 'x'",
            "DELETE FROM albums",
            "DROP TABLE albums",
            "CREATE TABLE t (x)",
            "ALTER TABLE artists RENAME TO a2",
            "REPLACE INTO artists VALUES (1, 'x', 'y')",
            "VACUUM",
            "REINDEX",
        ],
    )
    def test_refuses_writes(self, statement: str) -> None:
        with pytest.raises(ReadOnlyViolationError):
            ensure_read_only(statement)

    def test_refuses_multiple_statements(self) -> None:
        # The classic smuggling route: a harmless-looking first statement carrying a second.
        with pytest.raises(ReadOnlyViolationError, match="only one statement"):
            ensure_read_only("SELECT 1; DROP TABLE albums")

    def test_refuses_a_write_hidden_behind_a_comment(self) -> None:
        # Without comment stripping, the leading keyword looks like a comment and the check reads
        # the wrong token.
        with pytest.raises(ReadOnlyViolationError):
            ensure_read_only("/* harmless */ DELETE FROM albums")
        with pytest.raises(ReadOnlyViolationError):
            ensure_read_only("-- harmless\nDELETE FROM albums")

    def test_refuses_a_write_inside_a_permitted_form(self) -> None:
        # Leads with WITH, which is allowed, and is still a delete. This is why the gate checks for
        # write keywords anywhere, not just in the leading position.
        with pytest.raises(ReadOnlyViolationError, match="contains DELETE"):
            ensure_read_only("WITH x AS (SELECT 1) DELETE FROM albums")

    def test_refuses_attach(self) -> None:
        # The documented escape from mode=ro: attach a second database and write to that instead.
        with pytest.raises(ReadOnlyViolationError):
            ensure_read_only("ATTACH DATABASE '/tmp/evil.db' AS evil")

    def test_refuses_pragma_assignment(self) -> None:
        with pytest.raises(ReadOnlyViolationError, match="assigning a PRAGMA"):
            ensure_read_only("PRAGMA query_only = OFF")
        with pytest.raises(ReadOnlyViolationError, match="assigning a PRAGMA"):
            ensure_read_only("PRAGMA journal_mode = DELETE")

    def test_refuses_pragmas_outside_the_allowlist(self) -> None:
        with pytest.raises(ReadOnlyViolationError, match="not on the read-only allowlist"):
            ensure_read_only("PRAGMA writable_schema")

    def test_refuses_empty_input(self) -> None:
        with pytest.raises(ReadOnlyViolationError, match="empty query"):
            ensure_read_only("   ")

    def test_refuses_comment_only_input(self) -> None:
        with pytest.raises(ReadOnlyViolationError, match="no SQL statement"):
            ensure_read_only("-- just a thought")


class TestNoFalseRefusals:
    """A gate that refuses legitimate queries is a broken gate.

    These are the cases a naive keyword blacklist gets wrong, and each would make the server
    unusable for ordinary work in a way that looks like a safety feature.
    """

    def test_a_write_keyword_inside_a_string_literal_is_fine(self) -> None:
        ensure_read_only("SELECT * FROM albums WHERE title = 'DELETE'")
        ensure_read_only('SELECT * FROM albums WHERE title = "DROP TABLE"')

    def test_a_column_named_like_a_keyword_is_fine(self) -> None:
        # "updated_at" contains "UPDATE"; word boundaries are what stop this being refused.
        ensure_read_only("SELECT updated_at, created_at FROM albums")
        ensure_read_only("SELECT deleted FROM albums")

    def test_a_semicolon_inside_a_literal_is_not_a_separator(self) -> None:
        ensure_read_only("SELECT * FROM albums WHERE title = 'a;b'")

    def test_a_trailing_semicolon_is_fine(self) -> None:
        ensure_read_only("SELECT 1;")

    def test_a_doubled_quote_escape_is_handled(self) -> None:
        # SQLite escapes a quote by doubling it; a scanner that stops at the second quote would
        # mis-read the rest of the statement.
        ensure_read_only("SELECT * FROM artists WHERE name = 'it''s fine'")


class TestScanner:
    """The comment and literal scanner the gate depends on."""

    def test_strips_line_comments(self) -> None:
        assert "secret" not in strip_sql_noise("SELECT 1 -- secret\n")

    def test_strips_block_comments(self) -> None:
        assert "secret" not in strip_sql_noise("SELECT /* secret */ 1")

    def test_strips_unterminated_comments(self) -> None:
        # An unclosed comment swallows the rest of the input, which is also how SQLite reads it.
        assert strip_sql_noise("SELECT 1 /* never closed").strip() == "SELECT 1"

    def test_blanks_string_contents_but_keeps_the_quotes(self) -> None:
        cleaned = strip_sql_noise("SELECT 'DELETE FROM t'")
        assert "DELETE" not in cleaned
        assert "''" in cleaned

    def test_splits_on_real_separators_only(self) -> None:
        assert len(split_statements("SELECT 1; SELECT 2")) == 2
        assert len(split_statements("SELECT ';'")) == 1
        assert len(split_statements("SELECT 1 -- ;\n")) == 1


class TestDefenceInDepth:
    """The gaps that make three layers necessary rather than redundant.

    Each test documents a hole in one layer that another covers. These are measured facts about
    SQLite, not assumptions -- and if a future version changed one of them, the test would say so.
    """

    def test_mode_ro_alone_does_not_stop_attach(self, database_path: Path, tmp_path: Path) -> None:
        # The finding that justifies the statement gate. A connection opened read-only can attach a
        # second database and write to it freely: the flag constrains the main database, not the
        # connection's authority. Only the gate refusing ATTACH closes this.
        bare = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        try:
            bare.execute(f"ATTACH DATABASE '{tmp_path / 'side.db'}' AS side")
            bare.execute("CREATE TABLE side.written (x INTEGER)")
            bare.execute("INSERT INTO side.written VALUES (1)")
            assert bare.execute("SELECT count(*) FROM side.written").fetchone()[0] == 1
        finally:
            bare.close()

        # And the gate is what stops a caller getting there.
        with pytest.raises(ReadOnlyViolationError):
            ensure_read_only(f"ATTACH DATABASE '{tmp_path / 'side.db'}' AS side")

    def test_query_only_closes_the_attach_hole(
        self, connection: sqlite3.Connection, tmp_path: Path
    ) -> None:
        # The second layer covers what the first does not: with query_only on, the attached
        # database is read-only too.
        connection.execute(f"ATTACH DATABASE '{tmp_path / 'side2.db'}' AS side2")
        with pytest.raises(sqlite3.OperationalError, match="readonly database"):
            connection.execute("CREATE TABLE side2.written (x INTEGER)")

    def test_query_only_is_reversible_so_the_gate_must_refuse_it(
        self, connection: sqlite3.Connection
    ) -> None:
        # Why the gate refuses PRAGMA assignment. query_only is not a latch: any statement allowed
        # through could switch the second layer off.
        connection.execute("PRAGMA query_only = OFF")
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 0

        with pytest.raises(ReadOnlyViolationError):
            ensure_read_only("PRAGMA query_only = OFF")
