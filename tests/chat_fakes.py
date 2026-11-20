"""Fakes for the chat agent's tests: a scripted model and recording GitHub functions."""

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
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


def make_agent(monkeypatch, *replies, checkpointer=None, mcp_tools=None):
    """A chat agent whose model gives these replies, in order. In memory unless
    given a checkpointer; mcp_tools is {connector id: [tools]}."""
    model = FakeChatModel(messages=iter(replies))
    monkeypatch.setattr(chat, "get_llm", lambda: model)
    agent = chat.build_agent(checkpointer or InMemorySaver(), mcp_tools)
    return agent, model


def fake_mcp_tool(name, calls):
    """A tool like the MCP adapter makes: async only, so calling it on the sync
    path would fail. Its calls go into `calls`."""

    async def run(url: str) -> str:
        calls.append((name, url))
        return f"# Page at {url}"

    return StructuredTool.from_function(
        coroutine=run, name=name, description="Fetch a URL."
    )


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
