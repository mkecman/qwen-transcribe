"""SQLite storage for sessions (transcripts), their audio recordings, and settings."""
import json
import sqlite3
import threading
import time
import uuid
from typing import Optional

from . import config

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,              -- live | file
    title       TEXT,                       -- NULL = derived from text
    text        TEXT NOT NULL DEFAULT '',
    language    TEXT,
    status      TEXT NOT NULL DEFAULT 'done', -- recording | queued | processing | done | error
    progress    REAL,
    error       TEXT,
    source_name TEXT,
    duration    REAL NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS recordings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    path        TEXT NOT NULL,              -- relative to DATA_DIR
    duration    REAL NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def init():
    global _conn
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    with _lock:
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
        _conn.executescript(SCHEMA)
        # anything mid-flight when the server stopped
        _conn.execute("UPDATE sessions SET status='done' WHERE status='recording'")
        _conn.execute("UPDATE sessions SET status='error', error='Interrupted when the server stopped', "
                      "progress=NULL WHERE status IN ('queued','processing')")
        _conn.commit()


def _exec(sql: str, args=()):
    with _lock:
        cur = _conn.execute(sql, args)
        _conn.commit()
        return cur


def _query(sql: str, args=()) -> list[dict]:
    with _lock:
        return [dict(r) for r in _conn.execute(sql, args).fetchall()]


# ---- sessions ----------------------------------------------------------
def create_session(kind: str, language: str, status: str = "done", source_name: Optional[str] = None) -> dict:
    now = time.time()
    sid = uuid.uuid4().hex[:12]
    _exec("INSERT INTO sessions (id, kind, language, status, source_name, created_at, updated_at) "
          "VALUES (?,?,?,?,?,?,?)", (sid, kind, language, status, source_name, now, now))
    return get_session(sid)


def get_session(sid: str) -> Optional[dict]:
    rows = _query("SELECT * FROM sessions WHERE id=?", (sid,))
    return rows[0] if rows else None


def list_sessions(q: str = "", limit: int = 500) -> list[dict]:
    if q:
        like = f"%{q}%"
        return _query("SELECT * FROM sessions WHERE text LIKE ? OR title LIKE ? OR source_name LIKE ? "
                      "ORDER BY created_at DESC LIMIT ?", (like, like, like, limit))
    return _query("SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,))


_UPDATABLE = {"title", "text", "language", "status", "progress", "error", "duration", "source_name"}


def update_session(sid: str, touch: bool = True, **fields):
    fields = {k: v for k, v in fields.items() if k in _UPDATABLE}
    if touch:
        fields["updated_at"] = time.time()
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    _exec(f"UPDATE sessions SET {cols} WHERE id=?", (*fields.values(), sid))


def delete_session(sid: str) -> list[str]:
    paths = [r["path"] for r in get_recordings(sid)]
    _exec("DELETE FROM sessions WHERE id=?", (sid,))
    return paths


# ---- recordings --------------------------------------------------------
def add_recording(sid: str, path: str, duration: float) -> int:
    return _exec("INSERT INTO recordings (session_id, path, duration, created_at) VALUES (?,?,?,?)",
                 (sid, path, duration, time.time())).lastrowid


def update_recording(rid: int, duration: float):
    _exec("UPDATE recordings SET duration=? WHERE id=?", (duration, rid))


def get_recordings(sid: str) -> list[dict]:
    return _query("SELECT * FROM recordings WHERE session_id=? ORDER BY id", (sid,))


def delete_recordings(sid: str):
    _exec("DELETE FROM recordings WHERE session_id=?", (sid,))


# ---- settings ----------------------------------------------------------
def get_settings() -> dict:
    s = dict(config.DEFAULT_SETTINGS)
    for r in _query("SELECT key, value FROM settings"):
        if r["key"] in s:
            s[r["key"]] = json.loads(r["value"])
    return s


def save_settings(values: dict) -> dict:
    with _lock:
        for k, v in values.items():
            if k in config.DEFAULT_SETTINGS:
                _conn.execute("INSERT INTO settings (key, value) VALUES (?,?) "
                              "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (k, json.dumps(v)))
        _conn.commit()
    return get_settings()
