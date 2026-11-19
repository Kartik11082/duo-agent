"""GitHub webhook handling: check the signature, turn a payload into a job."""

import hashlib
import hmac

from duo.config import settings


def signature_is_valid(body: bytes, signature: str | None) -> bool:
    """Check GitHub's X-Hub-Signature-256: an HMAC of the raw body with our secret."""
    if not settings.github_webhook_secret or not signature:
        return False
    digest = hmac.new(
        settings.github_webhook_secret.encode(), body, hashlib.sha256
    ).hexdigest()
    # compare_digest takes the same time whether or not the strings match.
    return hmac.compare_digest(f"sha256={digest}", signature)


def job_from_issue_opened(payload: dict) -> dict:
    """The fields the worker needs from an issues/opened payload."""
    issue = payload["issue"]
    return {
        "repo": payload["repository"]["full_name"],
        "number": issue["number"],
        "title": issue["title"],
        "body": issue["body"] or "",  # GitHub sends null for an empty body
        "url": issue["html_url"],
    }
