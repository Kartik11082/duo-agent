"""Receives GitHub webhooks and queues them as jobs.

It only verifies, filters and stores: no LLM calls here, so it answers GitHub
in milliseconds. worker.py does the slow work.

Run:  uv run uvicorn server:app --port 8000
"""

import hashlib
import hmac
import json
import logging

from fastapi import FastAPI, Header, HTTPException, Request

import db
from config import settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("duo.server")
app = FastAPI()


def signature_is_valid(body: bytes, signature: str | None) -> bool:
    """Check GitHub's X-Hub-Signature-256: an HMAC of the raw body with our secret."""
    if not settings.github_webhook_secret or not signature:
        return False
    digest = hmac.new(
        settings.github_webhook_secret.encode(), body, hashlib.sha256
    ).hexdigest()
    # compare_digest takes the same time whether or not the strings match.
    return hmac.compare_digest(f"sha256={digest}", signature)


@app.post("/webhook", status_code=202)
async def webhook(
    request: Request,
    x_github_event: str = Header(""),
    x_github_delivery: str = Header(""),
    x_hub_signature_256: str | None = Header(None),
):
    # Verify against the exact bytes GitHub signed, before parsing anything.
    body = await request.body()
    if not signature_is_valid(body, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="invalid signature")

    if x_github_event == "ping":  # sent once when the webhook is created
        return {"status": "pong"}

    payload = json.loads(body)
    if x_github_event != "issues" or payload.get("action") != "opened":
        return {"status": "ignored"}

    issue = payload["issue"]
    repo = payload["repository"]["full_name"]
    job = {
        "repo": repo,
        "number": issue["number"],
        "title": issue["title"],
        "body": issue["body"] or "",  # GitHub sends null for an empty body
        "url": issue["html_url"],
    }
    queued = db.add_job(x_github_delivery, "github", f"{repo}#{issue['number']}", job)
    log.info(
        "Delivery %s for %s#%s: %s",
        x_github_delivery,
        repo,
        issue["number"],
        "queued" if queued else "duplicate",
    )
    return {"status": "queued" if queued else "duplicate"}
