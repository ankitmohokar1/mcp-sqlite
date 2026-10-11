"""Read-only enforcement -- three independent layers, because none of them is sufficient alone.

Exposing a database to a language model means accepting that the model will sometimes be wrong, and
occasionally that its input has been shaped by someone hostile. "The model was told not to write"
is not a control. These three are, and the reason there are three is that each one has a hole the
others cover. All of this was measured against SQLite 3.53, not assumed:

===================  =================  ======================  ==============================
Layer                Main-DB writes     Attached-DB writes      Reversible from SQL?
===================  =================  ======================  ==============================
``mode=ro`` URI      blocked by SQLite  **allowed**             n/a
``query_only=ON``    blocked            blocked                 **yes** -- ``PRAGMA ... = OFF``
Statement gate       blocked            ``ATTACH`` refused       closes the pragma hole
===================  =================  ======================  ==============================

The middle column is the interesting one. A connection opened read-only can still ``ATTACH`` a
second database file and write to *that* freely -- the read-only flag applies to the main database,
not to the connection's authority. ``PRAGMA query_only`` closes it, but that pragma can be switched
back off by any statement allowed through, so the gate has to refuse it. Remove any one layer and
there is a path to a write.

The gate is deliberately a whitelist. A blacklist of dangerous keywords is the design that looks
reasonable and loses: it has to anticipate every spelling of every write, and SQL offers a great
many (``INSERT OR REPLACE``, ``REPLACE INTO``, ``CREATE TRIGGER``, ``VACUUM INTO``, recursive CTEs
ending in ``DELETE``). Enumerating what is safe is a much shorter list and fails closed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Final

ALLOWED_LEADING_KEYWORDS: Final = frozenset({"SELECT", "WITH", "VALUES", "EXPLAIN"})
"""Statement forms that cannot write.

``WITH`` is included because a common table expression is the normal way to write a readable query
-- but note a CTE *can* end in a write (``WITH x AS (...) DELETE FROM t``), which the gate catches
separately by refusing any statement containing a write keyword at the top level. ``EXPLAIN`` is a
query about a query and executes nothing.
"""

ALLOWED_PRAGMAS: Final = frozenset(
    {
        "table_info",
        "table_xinfo",
        "table_list",
        "index_list",
        "index_info",
        "index_xinfo",
        "foreign_key_list",
        "database_list",
        "collation_list",
        "function_list",
        "module_list",
        "pragma_list",
        "compile_options",
    }
)
"""Pragmas that only report and cannot change anything.

Notably absent: ``query_only`` itself, which could switch off the second layer, and ``journal_mode``
/ ``user_version`` / ``writable_schema``, which mutate. The whitelist also only accepts the
*reading* form -- ``PRAGMA x`` or ``PRAGMA x(arg)``, never ``PRAGMA x = value``.
"""

FORBIDDEN_KEYWORDS: Final = frozenset(
    {
        "ATTACH",
        "DETACH",
        "INSERT",
        "UPDATE",
        "DELETE",
        "REPLACE",
        "DROP",
        "CREATE",
        "ALTER",
        "TRUNCATE",
        "VACUUM",
        "REINDEX",
        "GRANT",
        "REVOKE",
    }
)
"""Keywords refused anywhere in a statement, not just in the leading position.

This is a second line behind the leading-keyword whitelist, for the case where a permitted form
contains a write further in -- ``WITH x AS (SELECT 1) DELETE FROM t`` leads with ``WITH`` and is
still a delete. ``ATTACH`` is here because it is the documented escape from ``mode=ro``.
"""


class ReadOnlyViolationError(Exception):
    """Raised when a statement is refused by the gate.

    Carries a message intended to be shown to the model: it says what was refused and why, so the
    model can rewrite the query rather than retrying the same thing.
    """


def open_readonly(path: Path) -> sqlite3.Connection:
    """Opens a database for reading only.

    Applies the first two layers. The third is applied per statement by :func:`ensure_read_only`.

    Raises:
        FileNotFoundError: if the database does not exist. ``mode=ro`` will not create one, which is
            the behaviour we want -- a typo should fail rather than silently produce an empty
            database and a confusing set of empty tool results.
        sqlite3.DatabaseError: if the file exists but is not a database.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"no such database file: {resolved}")

    # mode=ro is enforced by SQLite itself rather than by this code, which is what makes it worth
    # having: it holds even for a statement the gate below failed to classify correctly.
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True, check_same_thread=False)
    connection.row_factory = sqlite3.Row

    # Covers writes to databases attached after the fact, which mode=ro does not.
    connection.execute("PRAGMA query_only = ON")

    # Touch the schema so a file that is not a database fails here, at startup, rather than on the
    # first tool call where the error would reach the model as a confusing tool failure.
    connection.execute("SELECT count(*) FROM sqlite_schema").fetchone()
    return connection


def strip_sql_noise(sql: str) -> str:
    """Returns ``sql`` with comments removed and string literals blanked.

    Both are necessary before looking for keywords. Comments can hide the leading keyword
    (``/* hi */ DELETE ...``), and string literals can contain words that look like keywords
    (``SELECT 'DELETE'``) -- stripping comments but not literals produces false refusals, and
    stripping neither produces false acceptances. Literals are replaced with empty quotes rather
    than deleted so the result stays recognisable as SQL.

    This is a scanner, not a parser. It only needs to be right about where comments and literals
    begin and end, which is a much smaller problem than parsing SQL.
    """
    out: list[str] = []
    index = 0
    length = len(sql)

    while index < length:
        char = sql[index]
        pair = sql[index : index + 2]

        if pair == "--":
            newline = sql.find("\n", index)
            index = length if newline == -1 else newline
        elif pair == "/*":
            close = sql.find("*/", index + 2)
            index = length if close == -1 else close + 2
        elif char in "'\"":
            # SQLite escapes a quote by doubling it, so scan to the first undoubled one.
            quote = char
            out.append(quote * 2)
            index += 1
            while index < length:
                if sql[index] == quote:
                    if sql[index : index + 2] == quote * 2:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
        else:
            out.append(char)
            index += 1

    return "".join(out)


def split_statements(sql: str) -> list[str]:
    """Splits on semicolons that are real separators, ignoring those inside comments or literals.

    Needed because the gate must see every statement, not just the first. Python's ``sqlite3``
    refuses multiple statements in one ``execute`` -- a useful backstop -- but refusing them here
    gives a message that says what was wrong instead of a driver error.
    """
    cleaned = strip_sql_noise(sql)
    return [part.strip() for part in cleaned.split(";") if part.strip()]


def _leading_word(statement: str) -> str:
    for token in statement.replace("(", " ").split():
        if token:
            return token.upper()
    return ""


def ensure_read_only(sql: str) -> None:
    """The statement gate. Returns normally if ``sql`` is a single, read-only statement.

    Raises:
        ReadOnlyViolationError: with a message explaining the refusal, suitable for returning to the
            model as a tool execution error so it can correct itself.
    """
    if not sql.strip():
        raise ReadOnlyViolationError("empty query")

    statements = split_statements(sql)
    if not statements:
        raise ReadOnlyViolationError("the query contains no SQL statement, only comments")
    if len(statements) > 1:
        raise ReadOnlyViolationError(
            f"only one statement per call is allowed, found {len(statements)}. "
            "Batching statements is how a read-only query smuggles in a write, so they are "
            "refused even when every statement looks harmless."
        )

    statement = statements[0]
    cleaned_upper = statement.upper()
    leading = _leading_word(statement)

    if leading == "PRAGMA":
        _ensure_read_only_pragma(statement)
        return

    if leading not in ALLOWED_LEADING_KEYWORDS:
        allowed = ", ".join(sorted(ALLOWED_LEADING_KEYWORDS))
        raise ReadOnlyViolationError(
            f"{leading or 'that'} is not a readable statement. This server only runs queries; "
            f"allowed forms are {allowed}, or an informational PRAGMA."
        )

    # Catch a write buried inside an otherwise permitted form. Word boundaries matter: a column
    # named "updated_at" must not trip the UPDATE rule.
    for keyword in FORBIDDEN_KEYWORDS:
        if _contains_word(cleaned_upper, keyword):
            raise ReadOnlyViolationError(
                f"the statement leads with {leading} but contains {keyword}, which can modify "
                "data or the connection's reach. Only pure queries are allowed."
            )


def _ensure_read_only_pragma(statement: str) -> None:
    body = statement[len("PRAGMA") :].strip()

    if "=" in body:
        raise ReadOnlyViolationError(
            "assigning a PRAGMA is not allowed -- several of them change what the connection may "
            "do, including query_only, which is part of how this server stays read-only."
        )

    name = body.split("(")[0].strip().lower()
    # A schema-qualified pragma (main.table_info) names the same pragma.
    name = name.rsplit(".", 1)[-1]

    if name not in ALLOWED_PRAGMAS:
        allowed = ", ".join(sorted(ALLOWED_PRAGMAS))
        raise ReadOnlyViolationError(
            f"PRAGMA {name} is not on the read-only allowlist. Allowed: {allowed}."
        )


def _contains_word(haystack_upper: str, word: str) -> bool:
    """Whether ``word`` appears as a whole word, so ``updated_at`` does not match ``UPDATE``."""
    start = 0
    while True:
        found = haystack_upper.find(word, start)
        if found == -1:
            return False
        previous = haystack_upper[found - 1] if found else ""
        before_ok = not previous or not (previous.isalnum() or previous == "_")
        end = found + len(word)
        after_ok = end == len(haystack_upper) or not (
            haystack_upper[end].isalnum() or haystack_upper[end] == "_"
        )
        if before_ok and after_ok:
            return True
        start = found + 1
