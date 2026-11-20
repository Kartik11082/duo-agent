"""POST /webhook: signature check, event filtering, and queueing.

storage.add_job is replaced with an in-memory fake, so no database is needed.
"""

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from duo import storage
from duo.apps.server import app
from duo.config import agent, settings

client = TestClient(app)

ISSUE_OPENED = {
    "action": "opened",
    "repository": {"full_name": "owner/repo"},
    "issue": {
        "number": 7,
        "title": "Crash on upload",
        "body": None,  # GitHub sends null for an empty body
        "html_url": "https://github.com/owner/repo/issues/7",
    },
}


@pytest.fixture
def queued(monkeypatch):
    """Fake add_job: remembers delivery ids, like the UNIQUE column does."""
    jobs = {}

    def add_job(delivery_id, source, thread_id, payload):
        if delivery_id in jobs:
            return False
        jobs[delivery_id] = (source, thread_id, payload)
        return True

    monkeypatch.setattr(storage, "add_job", add_job)
    return jobs


def sign(body: bytes, secret: str | None = None) -> str:
    """Sign like GitHub does; by default with the test secret set in conftest.py."""
    key = (secret or settings.github_webhook_secret).encode()
    return "sha256=" + hmac.new(key, body, hashlib.sha256).hexdigest()


def post(event, payload, delivery="d-1", signature=None):
    # Send exact bytes and sign those same bytes, as GitHub does.
    body = json.dumps(payload).encode()
    headers = {"X-GitHub-Event": event, "X-GitHub-Delivery": delivery}
    headers["X-Hub-Signature-256"] = signature or sign(body)
    return client.post("/webhook", content=body, headers=headers)


def test_ping():
    response = post("ping", {"zen": "Keep it simple."})
    assert response.status_code == 202
    assert response.json() == {"status": "pong"}


def test_missing_signature_is_rejected(queued):
    body = json.dumps(ISSUE_OPENED).encode()
    response = client.post(
        "/webhook", content=body, headers={"X-GitHub-Event": "issues"}
    )
    assert response.status_code == 401
    assert queued == {}


def test_wrong_signature_is_rejected(queued):
    body = json.dumps(ISSUE_OPENED).encode()
    response = post("issues", ISSUE_OPENED, signature=sign(body, "wrong-secret"))
    assert response.status_code == 401
    assert queued == {}


def test_issue_opened_is_queued(queued):
    response = post("issues", ISSUE_OPENED, delivery="d-1")
    assert response.json() == {"status": "queued"}
    assert queued["d-1"] == (
        "github",
        "owner/repo#7",
        {
            "repo": "owner/repo",
            "number": 7,
            "title": "Crash on upload",
            "body": "",
            "url": "https://github.com/owner/repo/issues/7",
        },
    )


def test_same_delivery_twice_is_a_duplicate(queued):
    post("issues", ISSUE_OPENED, delivery="d-1")
    response = post("issues", ISSUE_OPENED, delivery="d-1")
    assert response.json() == {"status": "duplicate"}
    assert len(queued) == 1


def test_issue_edited_is_ignored(queued):
    response = post("issues", {**ISSUE_OPENED, "action": "edited"})
    assert response.json() == {"status": "ignored"}
    assert queued == {}


def test_push_is_ignored(queued):
    response = post("push", {"ref": "refs/heads/main"})
    assert response.json() == {"status": "ignored"}
    assert queued == {}


def test_issue_opened_is_ignored_when_github_is_disabled(queued, monkeypatch):
    monkeypatch.setattr(agent.connectors["github"], "enabled", False)
    response = post("issues", ISSUE_OPENED)
    assert response.json() == {"status": "ignored"}
    assert queued == {}
