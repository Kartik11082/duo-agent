"""storage.py, the jobs queue, against a real Postgres test database (duo_test).

Skipped when Postgres isn't reachable; see test_database_url in conftest.py.
"""

import threading

import psycopg
import pytest

from duo import storage
from duo.config import settings


@pytest.fixture(autouse=True)
def db(monkeypatch, test_database_url):
    """Point storage at duo_test and start each test with empty tables."""
    monkeypatch.setattr(settings, "database_url", test_database_url)
    storage.setup()
    with storage.connect() as conn:
        conn.execute("TRUNCATE jobs, chat_threads RESTART IDENTITY")


def add_jobs(count):
    for n in range(1, count + 1):
        assert storage.add_job(f"d-{n}", "github", f"owner/repo#{n}", {"number": n})


def statuses():
    with storage.connect() as conn:
        rows = conn.execute("SELECT id, status, error FROM jobs ORDER BY id")
        return [(row["id"], row["status"], row["error"]) for row in rows]


def test_same_delivery_is_queued_once():
    assert storage.add_job("d-1", "github", "owner/repo#1", {"number": 1}) is True
    assert storage.add_job("d-1", "github", "owner/repo#1", {"number": 1}) is False
    assert len(statuses()) == 1


def test_claims_oldest_pending_first():
    add_jobs(2)
    first = storage.claim_next_job()
    second = storage.claim_next_job()
    assert first is not None and second is not None
    assert first["thread_id"] == "owner/repo#1"
    assert first["payload"] == {"number": 1}  # JSONB comes back as a dict
    assert second["thread_id"] == "owner/repo#2"
    assert storage.claim_next_job() is None  # nothing left pending


def test_finish_records_status_and_error():
    add_jobs(2)
    first = storage.claim_next_job()
    second = storage.claim_next_job()
    assert first is not None and second is not None
    storage.finish_job(first["id"], "done")
    storage.finish_job(second["id"], "failed", "LLM timed out")
    assert statuses() == [(1, "done", None), (2, "failed", "LLM timed out")]


def test_crash_reset_puts_running_jobs_back():
    add_jobs(3)
    storage.claim_next_job()
    storage.claim_next_job()  # two jobs "running" when the worker crashes
    assert storage.reset_running_jobs() == 2
    assert [status for _, status, _ in statuses()] == ["pending"] * 3
    job = storage.claim_next_job()
    assert job is not None and job["id"] == 1


def test_claim_skips_a_job_another_worker_has_locked():
    add_jobs(2)
    # Another worker's open transaction holds job 1's row lock.
    with psycopg.connect(settings.database_url) as other_worker:
        other_worker.execute("SELECT id FROM jobs WHERE id = 1 FOR UPDATE")
        job = storage.claim_next_job()  # SKIP LOCKED: doesn't wait for job 1
        assert job is not None and job["id"] == 2


def test_concurrent_claimers_never_claim_the_same_job():
    add_jobs(40)
    claimed = []  # list.append is thread-safe

    def worker():
        while (job := storage.claim_next_job()) is not None:
            claimed.append(job["id"])

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(claimed) == list(range(1, 41))  # each job exactly once


def test_chat_thread_is_remembered_and_replaced():
    assert storage.get_chat_thread(1234) is None
    storage.set_chat_thread(1234, "chat-1234-aaaa")
    storage.set_chat_thread(1234, "chat-1234-bbbb")  # /new
    assert storage.get_chat_thread(1234) == "chat-1234-bbbb"
    assert storage.get_chat_thread(5678) is None  # other chats are separate
