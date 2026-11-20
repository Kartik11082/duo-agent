"""policy.py and the triage routing that uses it.

A tool runs on its own only if its policy is exactly "auto". The policy is the
agent.yaml override if there is one, else the manifest's default.
"""

import pytest

from duo import policy
from duo.agents import triage
from duo.config import agent


@pytest.fixture
def github_policy(monkeypatch):
    """Set agent.yaml-style overrides for GitHub tools, undone after the test."""
    overrides = agent.connectors["github"].policy

    def set_policy(tool, value):
        monkeypatch.setitem(overrides, tool, value)

    return set_policy


def state_with(*names):
    return {"actions": [{"name": name, "args": {}} for name in names]}


# decide(): the rule itself.


def test_manifest_default_is_used_without_override():
    assert policy.decide("approve", None) == "approve"
    assert policy.decide("auto", None) == "auto"


def test_override_wins_over_manifest_default():
    assert policy.decide("approve", "auto") == "auto"
    assert policy.decide("auto", "approve") == "approve"


def test_misspelled_or_missing_policy_needs_approval():
    assert policy.decide("Auto", None) == "approve"
    assert policy.decide("approve", "atuo") == "approve"
    assert policy.decide(None, None) == "approve"


# policy_for(): the GitHub manifest plus agent.yaml.


def test_github_tools_need_approval_by_default():
    assert policy.policy_for("github", "add_label") == "approve"
    assert policy.policy_for("github", "post_comment") == "approve"


def test_agent_yaml_override_makes_post_comment_auto(github_policy):
    github_policy("post_comment", "auto")
    assert policy.policy_for("github", "post_comment") == "auto"
    assert policy.policy_for("github", "add_label") == "approve"


def test_unknown_tool_or_connector_needs_approval():
    assert policy.policy_for("github", "close_issue") == "approve"
    assert policy.policy_for("gitlab", "add_label") == "approve"


# route_by_policy(): pause unless every action is "auto".


def test_default_policy_asks_for_approval():
    assert triage.route_by_policy(state_with("add_label", "post_comment")) == "approve"


def test_one_auto_action_still_asks_for_approval(github_policy):
    github_policy("post_comment", "auto")  # add_label is still "approve"
    assert triage.route_by_policy(state_with("add_label", "post_comment")) == "approve"
    assert triage.route_by_policy(state_with("post_comment")) == "execute"


def test_all_auto_executes(github_policy):
    github_policy("add_label", "auto")
    github_policy("post_comment", "auto")
    assert triage.route_by_policy(state_with("add_label", "post_comment")) == "execute"


def test_one_unlisted_action_asks_for_approval(github_policy):
    github_policy("add_label", "auto")
    assert triage.route_by_policy(state_with("add_label", "close_issue")) == "approve"
