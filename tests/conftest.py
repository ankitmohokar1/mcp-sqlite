"""Shared fixtures."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from mcp_sqlite.readonly import open_readonly

SCHEMA = """
CREATE TABLE artists (id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT);
CREATE TABLE albums (
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  year INTEGER,
  artist_id INTEGER REFERENCES artists(id)
);
CREATE INDEX idx_albums_year ON albums(year);
CREATE UNIQUE INDEX idx_artists_name ON artists(name);
CREATE VIEW recent AS SELECT title, year FROM albums WHERE year >= 2000;
INSERT INTO artists VALUES (1,'Radiohead','UK'),(2,'Boards of Canada','UK'),(3,'Aphex Twin','UK');
INSERT INTO albums VALUES
  (1,'Kid A',2000,1),
  (2,'In Rainbows',2007,1),
  (3,'Music Has the Right to Children',1998,2),
  (4,'Selected Ambient Works 85-92',1992,3);
"""


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    """A real SQLite file on disk.

    A file rather than ``:memory:`` on purpose: the read-only guarantee this project is about comes
    from the ``file:...?mode=ro`` URI, which has no meaning for an in-memory database. Testing
    against memory would exercise a different code path from the one that ships.
    """
    path = tmp_path / "test.db"
    writer = sqlite3.connect(path)
    writer.executescript(SCHEMA)
    writer.commit()
    writer.close()
    return path


@pytest.fixture
def connection(database_path: Path) -> Iterator[sqlite3.Connection]:
    """A read-only connection, opened exactly as the server opens one."""
    conn = open_readonly(database_path)
    yield conn
    conn.close()
