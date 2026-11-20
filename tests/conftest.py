"""Shared test setup: every test runs offline, whatever is in the real .env.

settings is loaded from .env when duo.config is first imported, so a test
that forgot a stub could reach the real Telegram bot, GitHub App or Ollama.
The autouse fixture below blanks those settings for every test, so a missing
stub fails instead of calling out.
"""

import os

import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from duo.agents import triage
from duo.config import settings

# No LangSmith tracing in tests, even if the shell turned it on. Done here,
# before any LLM runs, because LangSmith caches these the first time it reads
# them. (The worker turns tracing on with llm.enable_tracing(); tests never do.)
for name in (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
):
    os.environ.pop(name, None)

WEBHOOK_SECRET = "test-webhook-secret"
CHAT_ID = 1234  # the one chat allowed to use the bot in tests
TEST_DB = "duo_test"

# Captured before offline_settings blanks it: test_database_url derives a
# separate test database from it, so tests never touch the real "duo" data.
REAL_DATABASE_URL = settings.database_url


@pytest.fixture(autouse=True)
def offline_settings(monkeypatch):
    monkeypatch.setattr(settings, "telegram_bot_token", "")
    monkeypatch.setattr(settings, "telegram_chat_id", CHAT_ID)
    monkeypatch.setattr(settings, "github_app_id", 0)
    monkeypatch.setattr(settings, "github_private_key_path", "")
    monkeypatch.setattr(settings, "github_webhook_secret", WEBHOOK_SECRET)
    monkeypatch.setattr(settings, "ollama_base_url", "http://127.0.0.1:9")  # nothing
    monkeypatch.setattr(settings, "database_url", "")  # only test_storage.py sets one
    monkeypatch.setattr(settings, "langsmith_api_key", "")
    monkeypatch.setattr(settings, "langsmith_tracing", True)
    for name in (
        "langsmith_endpoint",
        "langsmith_agent_id",
        "langsmith_agent_environment",
    ):
        monkeypatch.setattr(settings, name, "")
    monkeypatch.setattr(settings, "tavily_api_key", "")


@pytest.fixture(scope="session")
def test_database_url():
    """The URL of the duo_test database, created if missing (once per test run).

    Skips the requesting tests if Postgres isn't reachable, so the suite still
    runs without Docker. In CI (where GitHub sets CI=true) Postgres must be
    there, so a missing database fails instead of passing by skipping.
    """

    def unavailable(reason):
        if os.environ.get("CI"):
            pytest.fail(f"Postgres is required in CI: {reason}")
        pytest.skip(f"Postgres not available: {reason}")

    if not REAL_DATABASE_URL:
        unavailable("DATABASE_URL is not set")
    try:
        with psycopg.connect(
            REAL_DATABASE_URL, autocommit=True, connect_timeout=3
        ) as conn:
            row = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)
            ).fetchone()
            if row is None:  # CREATE DATABASE has no IF NOT EXISTS
                conn.execute(f"CREATE DATABASE {TEST_DB}")
    except psycopg.OperationalError as error:
        unavailable(str(error).splitlines()[0])
    # lock_timeout: if SKIP LOCKED ever broke, a test fails instead of hanging.
    return make_conninfo(
        REAL_DATABASE_URL, dbname=TEST_DB, options="-c lock_timeout=5s"
    )


class FakeLLM:
    """Stands in for get_llm()'s model: always labels the issue "bug"."""

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _prompt):
        return triage.Triage(label="bug", reasoning="Stub reasoning.")


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the triage LLM call. Patched in triage, where get_llm is looked up."""
    # get_llm() returns a model; calling the class FakeLLM() returns one too.
    monkeypatch.setattr(triage, "get_llm", FakeLLM)
