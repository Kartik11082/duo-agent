"""Authenticates as the duo-agent GitHub App, so the agent acts as its bot account."""

from github import Auth, GithubIntegration
from github.Repository import Repository

from duo.config import settings


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
