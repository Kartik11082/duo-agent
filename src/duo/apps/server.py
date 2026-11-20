"""Receives GitHub webhooks and queues them as jobs.

It only verifies, filters and stores: no LLM calls here, so it answers GitHub
in milliseconds. apps/worker.py does the slow work.

Run:  uv run duo-server
"""

import json
import logging

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request

from duo import storage
from duo.connectors import registry
from duo.connectors.github.webhook import JOB_BUILDERS, signature_is_valid

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("duo.server")
app = FastAPI()


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

    # Only events the GitHub manifest lists, and only while agent.yaml enables it.
    payload = json.loads(body)
    trigger = f"{x_github_event}.{payload.get('action')}"
    if trigger not in registry.triggers_for("github"):
        return {"status": "ignored"}

    job = JOB_BUILDERS[trigger](payload)
    thread_id = f"{job['repo']}#{job['number']}"
    queued = storage.add_job(x_github_delivery, "github", thread_id, job)
    log.info(
        "Delivery %s for %s: %s",
        x_github_delivery,
        thread_id,
        "queued" if queued else "duplicate",
    )
    return {"status": "queued" if queued else "duplicate"}


def main() -> None:
    """Entry point for `uv run duo-server`."""
    storage.setup()
    uvicorn.run(app, port=8000)
