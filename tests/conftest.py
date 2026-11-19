"""Shared test setup: every test runs offline, whatever is in the real .env.

settings is loaded from .env when duo.config is first imported, so a test
that forgot a stub could reach the real Telegram bot, GitHub App or Ollama.
The autouse fixture below blanks those settings for every test, so a missing
stub fails instead of calling out.
"""

import pytest

from duo.config import settings

WEBHOOK_SECRET = "test-webhook-secret"
CHAT_ID = 1234  # the one chat allowed to use the bot in tests

# The storage tests derive their test database from this; see test_storage.py.
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
