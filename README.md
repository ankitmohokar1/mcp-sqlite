# mcp-sqlite-explorer

Gives an AI assistant read-only access to a SQLite database over the
[Model Context Protocol](https://modelcontextprotocol.io). Pure Python, **no runtime dependencies**.

[![CI](https://github.com/ankitmohokar1/mcp-sqlite/actions/workflows/ci.yml/badge.svg)](https://github.com/ankitmohokar1/mcp-sqlite/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![MCP](https://img.shields.io/badge/MCP-2026--07--28%20%2B%20legacy-5A4FCF)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

```bash
pip install -e .
claude mcp add sqlite -- mcp-sqlite /path/to/your.db
```

Then ask: *"what's in this database?"* — the assistant gets four tools (`list_tables`,
`describe_table`, `query`, `sample_rows`) and cannot write, whatever it or anyone else asks it to do.

---

## "Read-only" means three enforcement layers, not a promise

Handing a language model a database means accepting that the model will sometimes be wrong, and
occasionally that its input was shaped by someone hostile. *Telling* it not to write is not a
control. These are — and there are three because **each one has a hole the others cover.** All of
this was measured against SQLite 3.53, not assumed:

| | main-DB writes | attached-DB writes | reversible from SQL? |
|---|---|---|---|
| `file:…?mode=ro` URI | blocked by SQLite | **allowed** | n/a |
| `PRAGMA query_only = ON` | blocked | blocked | **yes** — `PRAGMA query_only = OFF` |
| Statement gate | blocked | refuses `ATTACH` | closes the pragma hole |

The middle column is the one that surprised me. **A connection opened read-only can still `ATTACH` a
second database file and write to that freely** — `mode=ro` constrains the main database, not the
connection's authority:

```python
ro = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
ro.execute("DELETE FROM t")  # OperationalError: readonly database
ro.execute("ATTACH DATABASE '/tmp/evil.db' AS w")  # ...allowed
ro.execute("CREATE TABLE w.anything (x)")  # ...and this works
```

`query_only` closes that. But `query_only` is not a latch — any statement allowed through can switch
it back off — so the gate has to refuse pragma assignment. **Remove any one layer and there is a path
to a write.** Each of those three facts has a test, so if a future SQLite changes one, the suite says
so instead of the guarantee quietly weakening.

### The gate is a whitelist, deliberately

A blacklist of dangerous keywords is the design that looks reasonable and loses: it has to anticipate
every spelling of every write, and SQL has many (`INSERT OR REPLACE`, `REPLACE INTO`,
`CREATE TRIGGER`, `VACUUM INTO`, a CTE that ends in `DELETE`). Enumerating what is *safe* is a much
shorter list and fails closed.

It also has to avoid false refusals, which is where a naive keyword scan breaks — a gate that rejects
legitimate queries is broken in a way that looks like a safety feature:

```sql
SELECT * FROM albums WHERE title = 'DELETE'   -- allowed: literals are blanked before scanning
SELECT updated_at FROM albums                 -- allowed: UPDATE needs a word boundary
SELECT * FROM t WHERE name = 'a;b'            -- allowed: that semicolon is not a separator
/* harmless */ DELETE FROM albums             -- refused: comments stripped before reading the verb
WITH x AS (SELECT 1) DELETE FROM albums       -- refused: leads with WITH, is still a delete
```

## The protocol moved, and my prior was stale

MCP's current revision is **2026-07-28**, and writing this from memory would have produced a server
that does not connect. Three things changed:

| | What I'd have written | What the spec says |
|---|---|---|
| Handshake | `initialize` → `notifications/initialized`, then session state | **No handshake.** Stateless; every request carries its own version and capabilities in `_meta` |
| Discovery | derive from `initialize` | **`server/discover` is mandatory** — versions, capabilities and identity in one call |
| Results | bare payload | Every result carries **`resultType`** (`"complete"` for the normal case) |

There are also MCP-specific error codes in the range the spec reserves for itself — `-32022`
unsupported version, `-32021` missing client capability, `-32020` header mismatch — and an
implementation must emit *only* those, with their defined meanings.

### Dual-era, because modern-only would not connect

This server speaks both: a request arriving with modern `_meta` is served statelessly, an
`initialize` request selects legacy semantics. That is a deliberate trade, not completeness for its
own sake — most clients in the wild are still handshake-based, and a legacy-only server is writing to
a deprecated protocol. The cost is one branch at the dispatch boundary; everything below it is shared.

The legacy path also has to *not* send the new fields: that revision's schemas reject the extras, so
`resultType` and the `_meta` server identity are added in exactly one place that knows which era it
is in.

## Errors are addressed to the model

MCP draws a distinction worth implementing carefully, because getting it backwards produces either a
stuck conversation or a retry loop:

- **Protocol errors** (JSON-RPC `error`) — malformed request, unknown tool. Things the model cannot
  fix by adjusting arguments.
- **Tool execution errors** (a *result* with `isError: true`) — a refused query, a bad table name, a
  SQL syntax error. Things the model can act on, so they say what was wrong *and* what exists:

```
Refused: the statement leads with WITH but contains DELETE, which can modify data or the
connection's reach. Only pure queries are allowed.

No table or view named 'album'. Available: albums, artists, recent
```

## Output is bounded, always

A tool result lands in a context window, so an unbounded one is a denial of service against the thing
calling you. `SELECT * FROM events` should not return a large string — it should return a useful one.

- Rows capped (default 100, hard ceiling 1000 the caller **cannot** raise — a limit the caller can
  lift is a default, not a limit)
- Cells capped at 500 chars; BLOBs summarised as `<2048 bytes of BLOB>` rather than inlined
- **Truncation is stated, never silent** — returning the first 100 of 4,000,000 rows without saying
  so invites the model to conclude something about the whole table from a sample it thinks is complete

The implementation fetches `max_rows + 1` for this reason: a result of exactly `max_rows` is otherwise
indistinguishable from a truncated one.

Results are rendered as an aligned text table rather than JSON — JSON repeats every column name on
every row, which for a wide result is mostly tokens spent on punctuation the model already knows.

## Tools

| Tool | What it does |
|---|---|
| `list_tables` | Tables and views. Cheap; the place to start. |
| `describe_table` | Columns with types, nullability, PK and defaults, plus indexes and foreign keys. Costs the same whether the table holds ten rows or ten million. |
| `query` | One read-only statement, with `?` parameter binding. |
| `sample_rows` | First *n* rows, so the model need not write SQL just to look. |

Tool descriptions are written for a reader who cannot see the code, because that is all the model
gets — it picks a tool from that text alone. Saying what a tool *refuses* is as useful as saying what
it does: it stops the model trying.

## Testing

```
166 tests · 96% coverage · ruff + mypy --strict clean · runs in 0.3s
```

Beyond the unit tests, two things:

**The end-to-end tests drive the real dispatch loop**, feeding newline-delimited JSON through
`serve()` and asserting on what comes out. Framing, era selection, and which failures become
JSON-RPC errors versus tool results are precisely the parts that break without any single unit test
noticing.

**A conformance script runs the installed binary as a subprocess**, over actual stdin/stdout, which
is how a client launches it. The unit tests use `StringIO`, so this is the only place the real stream
plumbing is checked — including the rule that *nothing but MCP may reach stdout*, which a stray
`print` during debugging would violate in a way the client reports as an unrelated parse failure.
CI runs it on every push:

```
$ python scripts/conformance.py
modern protocol:
  ok    every modern result is typed complete
  ok    discover advertises 2026-07-28
legacy protocol:
  ok    legacy results are untyped, as that revision requires
writes are refused:
  ok    all 8 write attempts were refused
  ok    the database is byte-for-byte unchanged after the attacks
  ok    ATTACH created no file
all conformance checks passed
```

## Not implemented

- **Resources and prompts.** Only the `tools` capability is advertised, and the server does not claim
  `listChanged`: the tool set is fixed at startup from the database on the command line, so promising
  change notifications would promise something that can never happen.
- **Pagination.** `tools/list` supports a cursor; with four tools there is nothing to page, and
  omitting `nextCursor` is how the protocol says "last page".
- **Streamable HTTP transport.** stdio only. HTTP brings an authorization framework, `Origin`
  validation and DNS-rebinding concerns that are a project of their own — and stdio is what a local
  database wants anyway.
- **Write support**, behind a flag or otherwise. The three-layer guarantee is the point; a
  `--allow-writes` switch would make it a promise again.

## License

Apache 2.0 — see [LICENSE](LICENSE).
