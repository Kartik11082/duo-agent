"""agents/chat.build_interrupt_on: a tool runs on its own only if its policy is exactly "auto".

The policy comes from policy.py: the GitHub manifest's default, unless
agent.yaml overrides it. Tests set overrides the way agent.yaml would.
"""

from types import SimpleNamespace

import pytest
from langchain.agents.middleware import HumanInTheLoopMiddleware

from duo.agents import chat
from duo.config import agent
from duo.connectors.github.chat_tools import CHAT_TOOLS


@pytest.fixture
def github_policy(monkeypatch):
    """Set agent.yaml-style overrides for GitHub tools, undone after the test."""
    overrides = agent.connectors["github"].policy

    def set_policy(tool, value):
        monkeypatch.setitem(overrides, tool, value)

    return set_policy


def tools_named(*names):
    return [SimpleNamespace(name=name) for name in names]


def test_default_policy_gates_only_the_write_tools():
    interrupt_on = chat.build_interrupt_on(CHAT_TOOLS)
    assert sorted(interrupt_on) == ["add_label", "post_comment"]
    assert interrupt_on["add_label"] == {"allowed_decisions": ["approve", "reject"]}


def test_the_middleware_accepts_it():
    HumanInTheLoopMiddleware(interrupt_on=chat.build_interrupt_on(CHAT_TOOLS))


def test_override_to_auto_ungates_a_write_tool(github_policy):
    github_policy("add_label", "auto")
    assert list(chat.build_interrupt_on(CHAT_TOOLS)) == ["post_comment"]


def test_tool_missing_from_the_manifest_needs_approval():
    interrupt_on = chat.build_interrupt_on(tools_named("list_issues", "close_issue"))
    assert list(interrupt_on) == ["close_issue"]


def test_mistyped_value_needs_approval(github_policy):
    # agent.yaml's validation already rejects this; policy.py is the second line.
    github_policy("list_issues", "Auto")
    assert list(chat.build_interrupt_on(tools_named("list_issues"))) == ["list_issues"]


def test_override_for_a_misspelled_tool_changes_nothing(github_policy):
    github_policy("add_lable", "auto")
    assert list(chat.build_interrupt_on(tools_named("add_label"))) == ["add_label"]
