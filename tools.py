"""Actions the agent can take on GitHub, as its GitHub App bot account.

To add an action: write a function here, add it to ACTIONS, and give it a
policy entry (milestone 6, step C). Each action returns a short summary line.
"""

from github import Auth, GithubIntegration
from github.Repository import Repository

from config import settings

# Hidden in the bot's comment so it can tell it already commented.
COMMENT_MARKER = "<!-- duo-agent:triage -->"


def get_repo(full_name: str) -> Repository:
    """Authenticate as the App's installation on this repo and return the repo.

    Private key -> JWT (the App) -> installation token (this repo, 1 hour).
    """
    with open(settings.github_private_key_path) as f:
        private_key = f.read()
    integration = GithubIntegration(
        auth=Auth.AppAuth(settings.github_app_id, private_key)
    )
    owner, name = full_name.split("/")
    installation = integration.get_repo_installation(owner, name)
    return installation.get_github_for_installation().get_repo(full_name)


def add_label(repo: str, number: int, label: str) -> str:
    """Add a label to an issue. Adding a label it already has changes nothing."""
    issue = get_repo(repo).get_issue(number)
    issue.add_to_labels(label)
    return f"Labeled #{number} '{label}'"


def post_comment(repo: str, number: int, text: str) -> str:
    """Post the bot's triage comment, unless it already posted one."""
    issue = get_repo(repo).get_issue(number)
    for comment in issue.get_comments():
        if COMMENT_MARKER in comment.body:
            return f"Skipped comment on #{number}: already posted"
    issue.create_comment(f"{text}\n\n{COMMENT_MARKER}")
    return f"Commented on #{number}"


# Name -> function. The graph runs proposed actions by looking them up here.
ACTIONS = {
    "add_label": add_label,
    "post_comment": post_comment,
}
