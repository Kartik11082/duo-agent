"""The jobs table, a small durable queue in Postgres, and chat_threads.

server.py adds a job for each webhook; worker.py claims and runs them.
A job's life: pending -> running -> done | failed.

chat_threads remembers which thread each Telegram chat is on, so a restart
continues the same conversation.

delivery_id is UNIQUE, so the same GitHub delivery can only be queued once:
the database itself does the deduplication.

Call setup() once at startup, before the other functions.
"""

import psycopg
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb

from duo.config import settings

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector, for long-term memory later

CREATE TABLE IF NOT EXISTS jobs (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    delivery_id TEXT NOT NULL UNIQUE,
    source      TEXT NOT NULL,              -- "github" now, "schedule" later
    thread_id   TEXT NOT NULL,              -- LangGraph thread, e.g. "owner/repo#7"
    payload     JSONB NOT NULL,             -- what the worker needs to run the job
    status      TEXT NOT NULL DEFAULT 'pending',
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The chat agent's current LangGraph thread for each Telegram chat; /new replaces it.
CREATE TABLE IF NOT EXISTS chat_threads (
    chat_id   BIGINT PRIMARY KEY,
    thread_id TEXT NOT NULL
);
"""


def connect() -> psycopg.Connection[DictRow]:
    """Open a connection. Use it as `with connect() as conn:`.

    The with block commits if it succeeds, rolls back on an exception, and
    always closes. Rows come back as dicts, e.g. row["id"].
    """
    # Connection[DictRow].connect is psycopg.connect, but typed to return dict rows.
    return psycopg.Connection[DictRow].connect(
        settings.database_url, row_factory=dict_row
    )


def setup() -> None:
    """Create the pgvector extension and the jobs table if they don't exist."""
    with connect() as conn:
        conn.execute(SCHEMA)


def add_job(delivery_id: str, source: str, thread_id: str, payload: dict) -> bool:
    """Queue a job. Returns False if this delivery was already queued."""
    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO jobs (delivery_id, source, thread_id, payload) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (delivery_id) DO NOTHING",
            (delivery_id, source, thread_id, Jsonb(payload)),
        )
        return cursor.rowcount == 1


def claim_next_job() -> dict | None:
    """Mark the oldest pending job as running and return it, or None if none wait.

    FOR UPDATE locks the chosen row until this transaction ends; SKIP LOCKED
    makes a second worker pass over rows another worker has locked, instead
    of waiting for them. So two workers never claim the same job.
    """
    with connect() as conn:
        # payload comes back as a dict: psycopg decodes JSONB for us.
        return conn.execute(
            "UPDATE jobs SET status = 'running' "
            "WHERE id = ("
            "    SELECT id FROM jobs WHERE status = 'pending' "
            "    ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED"
            ") "
            "RETURNING id, source, thread_id, payload"
        ).fetchone()


def finish_job(job_id: int, status: str, error: str | None = None) -> None:
    """Record how a job ended: status is "done" or "failed"."""
    with connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = %s, error = %s WHERE id = %s",
            (status, error, job_id),
        )


def reset_running_jobs() -> int:
    """On worker startup: jobs left 'running' by a crash go back to 'pending'."""
    with connect() as conn:
        cursor = conn.execute(
            "UPDATE jobs SET status = 'pending' WHERE status = 'running'"
        )
        return cursor.rowcount


def get_chat_thread(chat_id: int) -> str | None:
    """The chat's current thread id, or None if it has none yet."""
    with connect() as conn:
        row = conn.execute(
            "SELECT thread_id FROM chat_threads WHERE chat_id = %s", (chat_id,)
        ).fetchone()
        return row["thread_id"] if row else None


def set_chat_thread(chat_id: int, thread_id: str) -> None:
    """Make thread_id the chat's current thread, replacing any earlier one."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO chat_threads (chat_id, thread_id) VALUES (%s, %s) "
            "ON CONFLICT (chat_id) DO UPDATE SET thread_id = EXCLUDED.thread_id",
            (chat_id, thread_id),
        )
