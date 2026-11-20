"""The GitHub tools the chat agent can call, as LangChain tools.

Thin wrappers over connectors/github/tools.py, which holds the GitHub logic.
The docstrings and type hints here are what the LLM sees: each tool's
description and argument schema. Tool names must match the tools in
manifest.yaml, where their policy is set.
"""

from typing import Literal

from langchain_core.tools import tool

from duo.connectors.github import tools as github


@tool(parse_docstring=True)
def list_issues(repo: str, state: Literal["open", "closed", "all"] = "open") -> str:
    """List a GitHub repository's issues, newest first (at most 20).

    Args:
        repo: The repository as "owner/name".
        state: Which issues to list: "open", "closed" or "all".
    """
    return github.list_issues(repo, state)


@tool(parse_docstring=True)
def get_issue(repo: str, number: int) -> str:
    """Read one GitHub issue: its title, state, labels and body.

    Args:
        repo: The repository as "owner/name".
        number: The issue number.
    """
    return github.get_issue(repo, number)


@tool(parse_docstring=True)
def add_label(repo: str, number: int, label: str) -> str:
    """Add a label to a GitHub issue.

    Args:
        repo: The repository as "owner/name".
        number: The issue number.
        label: The label name, e.g. "bug", "enhancement", "question" or "documentation".
    """
    return github.add_label(repo, number, label)


@tool(parse_docstring=True)
def post_comment(repo: str, number: int, text: str) -> str:
    """Post a comment on a GitHub issue.

    Args:
        repo: The repository as "owner/name".
        number: The issue number.
        text: The comment, in Markdown.
    """
    return github.post_comment(repo, number, text, marker=None)


CHAT_TOOLS = [list_issues, get_issue, add_label, post_comment]
