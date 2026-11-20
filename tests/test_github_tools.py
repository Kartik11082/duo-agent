"""connectors/github: the GitHub functions and the chat agent's tools.

get_repo is replaced with a fake repository, so nothing reaches GitHub.
"""

from types import SimpleNamespace

import pytest

from duo.connectors.github import chat_tools, tools


def make_issue(number, title="Title", labels=(), is_pr=False, body="Body"):
    issue = SimpleNamespace(
        number=number,
        title=title,
        labels=[SimpleNamespace(name=name) for name in labels],
        pull_request=SimpleNamespace() if is_pr else None,
        state="open",
        body=body,
        comments=0,
        posted=[],
    )
    issue.get_comments = lambda: [SimpleNamespace(body=text) for text in issue.posted]
    issue.create_comment = issue.posted.append
    issue.add_to_labels = lambda name: issue.labels.append(SimpleNamespace(name=name))
    return issue


@pytest.fixture
def repo(monkeypatch):
    """A fake repository with issues #1 and #2 and pull request #3."""
    issues = [
        make_issue(3, "A pull request", is_pr=True),
        make_issue(2, "Crash on upload", labels=["bug"]),
        make_issue(1, "Docs typo"),
    ]
    fake = SimpleNamespace(
        issues=issues,
        get_issues=lambda state: issues,  # noqa: ARG005 (same call as PyGithub)
        get_issue=lambda number: next(i for i in issues if i.number == number),
    )
    monkeypatch.setattr(tools, "get_repo", lambda _full_name: fake)
    return fake


def test_list_issues_skips_pull_requests(repo):  # noqa: ARG001
    assert tools.list_issues("o/r") == "#2 [bug] Crash on upload\n#1 [] Docs typo"


def test_list_issues_stops_at_the_cap(repo, monkeypatch):
    monkeypatch.setattr(tools, "MAX_ISSUES", 1)
    assert tools.list_issues("o/r") == "#2 [bug] Crash on upload"
    repo.issues.clear()
    assert tools.list_issues("o/r", "closed") == "No closed issues."


def test_get_issue_cuts_a_long_body(repo, monkeypatch):  # noqa: ARG001
    monkeypatch.setattr(tools, "MAX_BODY_CHARS", 4)
    text = tools.get_issue("o/r", 2)
    assert text.startswith("#2 Crash on upload\nState: open\nLabels: bug\n")
    assert text.endswith("\n\nBody")


def test_triage_comment_is_posted_once(repo):
    assert tools.post_comment("o/r", 1, "Hi") == "Commented on #1"
    assert tools.post_comment("o/r", 1, "Hi") == "Skipped comment on #1: already posted"
    assert repo.get_issue(1).posted == [f"Hi\n\n{tools.COMMENT_MARKER}"]


def test_chat_comment_is_posted_even_after_triage_commented(repo):
    tools.post_comment("o/r", 1, "Triage comment")
    result = chat_tools.post_comment.invoke({"repo": "o/r", "number": 1, "text": "Hi"})
    assert result == "Commented on #1"
    assert repo.get_issue(1).posted[-1] == "Hi"  # no hidden marker


def test_chat_tools_call_through_to_github(repo):
    args = {"repo": "o/r", "number": 2, "label": "question"}
    assert chat_tools.add_label.invoke(args) == "Labeled #2 'question'"
    assert [label.name for label in repo.get_issue(2).labels] == ["bug", "question"]
    assert chat_tools.list_issues.invoke({"repo": "o/r"}).startswith(
        "#2 [bug, question]"
    )
    assert chat_tools.get_issue.invoke({"repo": "o/r", "number": 1}).startswith("#1")


def test_chat_tool_names_match_the_policy_keys():
    names = [t.name for t in chat_tools.CHAT_TOOLS]
    assert names == ["list_issues", "get_issue", "add_label", "post_comment"]
