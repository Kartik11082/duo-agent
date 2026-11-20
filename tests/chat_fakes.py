"""Fakes for the chat agent's tests: a scripted model and recording GitHub functions."""

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from duo.agents import chat
from duo.connectors.github import tools as github_tools


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


def make_agent(monkeypatch, *replies, checkpointer=None):
    """A chat agent whose model gives these replies, in order. In memory unless
    given a checkpointer."""
    model = FakeChatModel(messages=iter(replies))
    monkeypatch.setattr(chat, "get_llm", lambda: model)
    return chat.build_agent(checkpointer or InMemorySaver()), model


def fake_github(monkeypatch):
    """Replace list_issues and add_label; returns the list their calls go into."""
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
