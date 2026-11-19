"""The jobs table: a small, durable queue in SQLite.

server.py adds a job for each webhook; worker.py claims and runs them.
A job's life: pending -> running -> done | failed.

delivery_id is UNIQUE, so the same GitHub delivery can only be queued once:
the database itself does the deduplication.
"""

import json
import sqlite3
from contextlib import contextmanager

from duo.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY,
    delivery_id TEXT NOT NULL UNIQUE,
    source      TEXT NOT NULL,              -- "github" now, "schedule" later
    thread_id   TEXT NOT NULL,              -- LangGraph thread, e.g. "owner/repo#7"
    payload     TEXT NOT NULL,              -- JSON the worker needs to run the job
    status      TEXT NOT NULL DEFAULT 'pending',
    error       TEXT,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


@contextmanager
def connect():
    """Open a connection, commit if the block succeeds, always close.

    A fresh connection per call keeps this safe to use from any thread or
    process (server and worker both open duo.db).
    """
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        with conn:  # commits on success, rolls back on an exception
            conn.execute(SCHEMA)
            yield conn
    finally:
        conn.close()


def add_job(delivery_id: str, source: str, thread_id: str, payload: dict) -> bool:
    """Queue a job. Returns False if this delivery was already queued."""
    with connect() as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO jobs (delivery_id, source, thread_id, payload) "
            "VALUES (?, ?, ?, ?)",
            (delivery_id, source, thread_id, json.dumps(payload)),
        )
        return cursor.rowcount == 1


def claim_next_job() -> dict | None:
    """Mark the oldest pending job as running and return it, or None if none wait."""
    with connect() as conn:
        rows = conn.execute(
            "UPDATE jobs SET status = 'running' "
            "WHERE id = (SELECT id FROM jobs WHERE status = 'pending' ORDER BY id LIMIT 1) "
            "RETURNING id, source, thread_id, payload"
        ).fetchall()
    if not rows:
        return None
    row = rows[0]
    return {
        "id": row["id"],
        "source": row["source"],
        "thread_id": row["thread_id"],
        "payload": json.loads(row["payload"]),
    }


def finish_job(job_id: int, status: str, error: str | None = None) -> None:
    """Record how a job ended: status is "done" or "failed"."""
    with connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, error = ? WHERE id = ?",
            (status, error, job_id),
        )


def reset_running_jobs() -> int:
    """On worker startup: jobs left 'running' by a crash go back to 'pending'."""
    with connect() as conn:
        cursor = conn.execute(
            "UPDATE jobs SET status = 'pending' WHERE status = 'running'"
        )
        return cursor.rowcount
