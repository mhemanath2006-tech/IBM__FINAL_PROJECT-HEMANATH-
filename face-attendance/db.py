"""Tiny database layer that works with SQLite (local) and PostgreSQL (hosted).

Set DATABASE_URL to a postgres:// URL to use PostgreSQL (e.g. a free Neon database,
so data survives restarts on free hosts). Otherwise a local SQLite file is used.

SQL is written with ? placeholders and avoids the percent sign, so it runs on both.
"""
import os
import sqlite3

from flask import g

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
USE_PG = DATABASE_URL.startswith(("postgres://", "postgresql://"))
SQLITE_PATH = os.environ.get("SQLITE_PATH", "attendance.db")

if USE_PG:
    import psycopg2

    IntegrityErrors = (psycopg2.IntegrityError,)
else:
    IntegrityErrors = (sqlite3.IntegrityError,)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id {pk},
  username TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL,
  name TEXT NOT NULL,
  roll_no TEXT,
  section TEXT,
  active INTEGER NOT NULL DEFAULT 1,
  consent_at TEXT
);
CREATE TABLE IF NOT EXISTS face_templates (
  user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  embeddings TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS class_sessions (
  id {pk},
  section TEXT NOT NULL,
  teacher_id INTEGER NOT NULL REFERENCES users(id),
  day TEXT NOT NULL,
  opens_at TEXT NOT NULL,
  late_after TEXT NOT NULL,
  closes_at TEXT NOT NULL,
  closed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS attendance (
  id {pk},
  session_id INTEGER NOT NULL REFERENCES class_sessions(id) ON DELETE CASCADE,
  student_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  status TEXT NOT NULL,
  method TEXT NOT NULL,
  marked_at TEXT NOT NULL,
  similarity REAL,
  UNIQUE (session_id, student_id)
);
CREATE TABLE IF NOT EXISTS challenges (
  nonce TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL,
  direction TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  used INTEGER NOT NULL DEFAULT 0
)
"""


class Conn:
    def __init__(self, raw):
        self.raw = raw

    @staticmethod
    def _sql(sql):
        return sql.replace("?", "%s") if USE_PG else sql

    def all(self, sql, params=()):
        cur = self.raw.cursor()
        try:
            cur.execute(self._sql(sql), tuple(params))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        finally:
            cur.close()

    def one(self, sql, params=()):
        rows = self.all(sql, params)
        return rows[0] if rows else None

    def run(self, sql, params=()):
        """Execute a write; returns affected row count."""
        cur = self.raw.cursor()
        try:
            cur.execute(self._sql(sql), tuple(params))
            n = cur.rowcount
            self.raw.commit()
            return n
        except Exception:
            self.raw.rollback()
            raise
        finally:
            cur.close()

    def insert(self, sql, params=()):
        """Execute an INSERT and return the new row id."""
        cur = self.raw.cursor()
        try:
            cur.execute(self._sql(sql) + " RETURNING id", tuple(params))
            new_id = cur.fetchone()[0]
            self.raw.commit()
            return new_id
        except Exception:
            self.raw.rollback()
            raise
        finally:
            cur.close()

    def close(self):
        try:
            self.raw.close()
        except Exception:
            pass


def _connect():
    if USE_PG:
        return Conn(psycopg2.connect(DATABASE_URL))
    raw = sqlite3.connect(SQLITE_PATH, timeout=15)
    raw.execute("PRAGMA foreign_keys = ON")
    return Conn(raw)


def get_db():
    if "db" not in g:
        g.db = _connect()
    return g.db


def close_db(_exc=None):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_db():
    conn = _connect()
    try:
        pk = "SERIAL PRIMARY KEY" if USE_PG else "INTEGER PRIMARY KEY AUTOINCREMENT"
        for stmt in SCHEMA.format(pk=pk).split(";"):
            if stmt.strip():
                conn.run(stmt)
    finally:
        conn.close()
