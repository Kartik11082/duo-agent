"""agents/chat: the agent loop, approval pauses, GitHub errors and trimming.

The model is a fake that replays scripted replies (including tool calls), the
GitHub functions are replaced, and threads are kept in memory.
"""

from typing import Any

import pytest
from github import GithubException
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import Field

from duo import config
from duo.agents import chat
from duo.connectors.github import tools as github_tools

CONFIG = {"configurable": {"thread_id": "test-chat"}}


class FakeChatModel(GenericFakeChatModel):
    """Replays `messages` in order and records what it was sent in `seen`."""

    seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002 (the base class raises)
        return self

    def _generate(self, messages, *args: Any, **kwargs: Any):
        self.seen.append(messages)
        return super()._generate(messages, *args, **kwargs)


def call(name, **args):
    """A model reply that calls one tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "c1"}])


@pytest.fixture
def github(monkeypatch):
    """Fake GitHub functions that record their calls."""
    calls = []

    def list_issues(repo, state):
        calls.append(("list_issues", repo, state))
        return "#7 [] Docs typo"

    def add_label(repo, number, label):
        calls.append(("add_label", repo, number, label))
        return f"Labeled #{number} '{label}'"

    monkeypatch.setattr(github_tools, "list_issues", list_issues)
    monkeypatch.setattr(github_tools, "add_label", add_label)
    return calls


def make_agent(monkeypatch, *replies):
    """An in-memory agent whose model gives these replies, in order."""
    model = FakeChatModel(messages=iter(replies))
    monkeypatch.setattr(chat, "get_llm", lambda: model)
    return chat.build_agent(InMemorySaver()), model


def ask(agent, text):
    return agent.invoke({"messages": [{"role": "user", "content": text}]}, CONFIG)


def test_read_tool_runs_without_pausing(monkeypatch, github):
    agent, _ = make_agent(
        monkeypatch, call("list_issues", repo="o/r"), AIMessage("One open issue: #7.")
    )
    result = ask(agent, "What's open on o/r?")
    assert "__interrupt__" not in result
    assert result["messages"][-1].content == "One open issue: #7."
    assert github == [("list_issues", "o/r", "open")]


def test_write_tool_pauses_with_the_exact_call(monkeypatch, github):
    agent, _ = make_agent(
        monkeypatch, call("add_label", repo="o/r", number=7, label="bug")
    )
    result = ask(agent, "Label #7 as bug")
    [request] = result["__interrupt__"][0].value["action_requests"]
    assert request["name"] == "add_label"
    assert request["args"] == {"repo": "o/r", "number": 7, "label": "bug"}
    assert github == []  # nothing ran yet


def test_approve_runs_the_tool(monkeypatch, github):
    agent, _ = make_agent(
        monkeypatch,
        call("add_label", repo="o/r", number=7, label="bug"),
        AIMessage("Done: #7 is labeled bug."),
    )
    ask(agent, "Label #7 as bug")
    assert github == []  # waits for the tap
    result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), CONFIG)
    assert github == [("add_label", "o/r", 7, "bug")]
    assert result["messages"][-1].content == "Done: #7 is labeled bug."


def test_reject_does_not_run_the_tool(monkeypatch, github):
    agent, _ = make_agent(
        monkeypatch,
        call("add_label", repo="o/r", number=7, label="bug"),
        AIMessage("OK, I left #7 alone."),
    )
    ask(agent, "Label #7 as bug")
    result = agent.invoke(Command(resume={"decisions": [{"type": "reject"}]}), CONFIG)
    assert github == []
    tool_message = result["messages"][-2]
    assert isinstance(tool_message, ToolMessage)
    assert "rejected" in str(tool_message.content)
    assert result["messages"][-1].content == "OK, I left #7 alone."


def test_github_error_goes_back_to_the_model(monkeypatch):
    def get_issue(repo, number):  # noqa: ARG001
        raise GithubException(404, {"message": "Not Found"})

    monkeypatch.setattr(github_tools, "get_issue", get_issue)
    agent, _ = make_agent(
        monkeypatch, call("get_issue", repo="o/r", number=99), AIMessage("No #99.")
    )
    result = ask(agent, "Show #99")
    assert result["messages"][-2].content == "GitHub error 404: Not Found"
    assert result["messages"][-1].content == "No #99."


def test_old_messages_are_trimmed_but_kept(monkeypatch):
    monkeypatch.setattr(config.agent.model, "num_ctx", chat.RESERVED_TOKENS + 50)
    agent, model = make_agent(monkeypatch, AIMessage("ok"), AIMessage("hi"))
    ask(agent, "x" * 1000)  # about 250 tokens: over the 50-token budget next turn
    ask(agent, "hello")
    assert [m.content for m in model.seen[-1]][1:] == [
        "hello"
    ]  # [0] is the system prompt
    assert (
        len(agent.get_state(CONFIG).values["messages"]) == 4
    )  # the checkpoint has all


def test_a_turn_over_budget_is_sent_whole(monkeypatch):
    monkeypatch.setattr(config.agent.model, "num_ctx", chat.RESERVED_TOKENS + 50)
    agent, model = make_agent(monkeypatch, AIMessage("ok"))
    ask(agent, "y" * 1000)
    assert [m.content for m in model.seen[-1]][1:] == ["y" * 1000]
