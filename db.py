"""db.py — SQLite-хранилище записей транскрибации.

Одна строка = один загруженный аудиофайл. Адресация по uuid.
WAL-режим + busy_timeout, чтобы фоновый воркер и HTTP-хендлеры
не мешали друг другу. Файлы (mp3, txt) лежат на диске, в БД — метаданные.
"""

import sqlite3
import time
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "data" / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS transcriptions (
    id             TEXT PRIMARY KEY,
    filename       TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'queued',
    created_at     REAL NOT NULL,
    started_at     REAL,
    finished_at    REAL,
    duration_sec   REAL,
    progress_pct   REAL NOT NULL DEFAULT 0,
    segments_count INTEGER NOT NULL DEFAULT 0,
    error          TEXT
);
CREATE INDEX IF NOT EXISTS idx_status_created ON transcriptions(status, created_at);
"""

# Допустимые статусы: queued → processing → done | error


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def create(job_id: str, filename: str):
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO transcriptions (id, filename, status, created_at) "
            "VALUES (?, ?, 'queued', ?)",
            (job_id, filename, time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def get(job_id: str) -> Optional[dict]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM transcriptions WHERE id = ?", (job_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_jobs(status: Optional[str] = None, limit: int = 50, offset: int = 0):
    """Возвращает (items, total). Сортировка — новые сверху."""
    conn = _connect()
    try:
        if status:
            total = conn.execute(
                "SELECT COUNT(*) FROM transcriptions WHERE status = ?", (status,)
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM transcriptions WHERE status = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset),
            ).fetchall()
        else:
            total = conn.execute(
                "SELECT COUNT(*) FROM transcriptions"
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM transcriptions "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows], total
    finally:
        conn.close()


def claim_next() -> Optional[dict]:
    """Атомарно берёт самую старую queued-задачу → processing.

    BEGIN IMMEDIATE держит блокировку записи, поэтому даже несколько
    процессов не заберут одну задачу дважды. Возвращает строку или None.
    """
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM transcriptions WHERE status = 'queued' "
            "ORDER BY created_at LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            return None
        conn.execute(
            "UPDATE transcriptions SET status = 'processing', started_at = ? "
            "WHERE id = ?",
            (time.time(), row["id"]),
        )
        conn.commit()
        job = dict(row)
        job["status"] = "processing"
        return job
    finally:
        conn.close()


def update_progress(
    job_id: str,
    progress_pct: float,
    segments_count: int,
    duration_sec: Optional[float] = None,
):
    conn = _connect()
    try:
        if duration_sec is not None:
            conn.execute(
                "UPDATE transcriptions SET progress_pct = ?, segments_count = ?, "
                "duration_sec = ? WHERE id = ?",
                (progress_pct, segments_count, duration_sec, job_id),
            )
        else:
            conn.execute(
                "UPDATE transcriptions SET progress_pct = ?, segments_count = ? "
                "WHERE id = ?",
                (progress_pct, segments_count, job_id),
            )
        conn.commit()
    finally:
        conn.close()


def mark_done(job_id: str, duration_sec: Optional[float] = None):
    conn = _connect()
    try:
        conn.execute(
            "UPDATE transcriptions SET status = 'done', progress_pct = 100, "
            "finished_at = ?, duration_sec = COALESCE(?, duration_sec) WHERE id = ?",
            (time.time(), duration_sec, job_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_error(job_id: str, error: str):
    conn = _connect()
    try:
        conn.execute(
            "UPDATE transcriptions SET status = 'error', finished_at = ?, "
            "error = ? WHERE id = ?",
            (time.time(), error[:2000], job_id),
        )
        conn.commit()
    finally:
        conn.close()


def requeue_stale() -> int:
    """При старте: зависшие processing → обратно в queued (воркер мог упасть)."""
    conn = _connect()
    try:
        cur = conn.execute(
            "UPDATE transcriptions SET status = 'queued', started_at = NULL, "
            "progress_pct = 0 WHERE status = 'processing'"
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def delete(job_id: str) -> bool:
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM transcriptions WHERE id = ?", (job_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def counts() -> dict:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM transcriptions GROUP BY status"
        ).fetchall()
        out = {"queued": 0, "processing": 0, "done": 0, "error": 0}
        for r in rows:
            out[r["status"]] = r["n"]
        out["total"] = sum(out.values())
        return out
    finally:
        conn.close()
