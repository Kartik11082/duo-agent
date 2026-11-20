"""What the agent can read and do on GitHub, as its GitHub App bot account.

Each function returns a short text result. The triage workflow runs the write
actions through ACTIONS; the chat agent calls them through the LangChain tools
in chat_tools.py. A new tool also needs an entry, with its default policy, in
manifest.yaml.
"""

from duo.connectors.github.auth import get_repo

# Hidden in the bot's comment so it can tell it already commented.
COMMENT_MARKER = "<!-- duo-agent:triage -->"

# Keep read results small: they go into the LLM's context window.
MAX_ISSUES = 20
MAX_BODY_CHARS = 2000


def list_issues(repo: str, state: str = "open") -> str:
    """List up to MAX_ISSUES issues, newest first, one line each. Skips pull requests."""
    lines = []
    # GitHub's issues API returns pull requests too; they have .pull_request set.
    # get_issues() fetches page by page, so stopping early saves requests.
    for issue in get_repo(repo).get_issues(state=state):
        if issue.pull_request is not None:
            continue
        labels = ", ".join(label.name for label in issue.labels)
        lines.append(f"#{issue.number} [{labels}] {issue.title}")
        if len(lines) == MAX_ISSUES:
            break
    return "\n".join(lines) or f"No {state} issues."


def get_issue(repo: str, number: int) -> str:
    """One issue's title, state, labels and body (cut to MAX_BODY_CHARS)."""
    issue = get_repo(repo).get_issue(number)
    labels = ", ".join(label.name for label in issue.labels) or "none"
    body = (issue.body or "")[:MAX_BODY_CHARS]
    return (
        f"#{issue.number} {issue.title}\n"
        f"State: {issue.state}\n"
        f"Labels: {labels}\n"
        f"Comments: {issue.comments}\n\n"
        f"{body}"
    )


def add_label(repo: str, number: int, label: str) -> str:
    """Add a label to an issue. Adding a label it already has changes nothing."""
    issue = get_repo(repo).get_issue(number)
    issue.add_to_labels(label)
    return f"Labeled #{number} '{label}'"


def post_comment(
    repo: str, number: int, text: str, marker: str | None = COMMENT_MARKER
) -> str:
    """Post a comment.

    With a marker (by default the triage one), skip if a comment with that marker
    is already there, and hide the marker in the new comment. With marker=None
    (the chat agent), always post the text as it is.
    """
    issue = get_repo(repo).get_issue(number)
    if marker is not None:
        for comment in issue.get_comments():
            if marker in comment.body:
                return f"Skipped comment on #{number}: already posted"
        text = f"{text}\n\n{marker}"
    issue.create_comment(text)
    return f"Commented on #{number}"


# Name -> function. The graph runs proposed actions by looking them up here.
ACTIONS = {
    "add_label": add_label,
    "post_comment": post_comment,
}
