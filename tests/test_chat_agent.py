"""agents/chat: the agent loop, approval pauses, GitHub errors and trimming.

The model is a fake that replays scripted replies (including tool calls), the
GitHub functions are replaced, and threads are kept in memory.
"""

import asyncio
import logging

import pytest
from chat_fakes import call, fake_github, fake_mcp_tool, make_agent
from github import GithubException
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from duo import config
from duo.agents import chat
from duo.connectors.github import tools as github_tools

CONFIG = {"configurable": {"thread_id": "test-chat"}}


@pytest.fixture
def github(monkeypatch):
    return fake_github(monkeypatch)


def ask(agent, text):
    """One user turn, run async like the worker does (agents/chat.py)."""
    turn = {"messages": [{"role": "user", "content": text}]}
    return asyncio.run(agent.ainvoke(turn, CONFIG))


def resume(agent, decision):
    """Answer the pause with "approve" or "reject"."""
    command = Command(resume={"decisions": [{"type": decision}]})
    return asyncio.run(agent.ainvoke(command, CONFIG))


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
    result = resume(agent, "approve")
    assert github == [("add_label", "o/r", 7, "bug")]
    assert result["messages"][-1].content == "Done: #7 is labeled bug."


def test_reject_does_not_run_the_tool(monkeypatch, github):
    agent, _ = make_agent(
        monkeypatch,
        call("add_label", repo="o/r", number=7, label="bug"),
        AIMessage("OK, I left #7 alone."),
    )
    ask(agent, "Label #7 as bug")
    result = resume(agent, "reject")
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


def test_tools_come_from_the_github_manifest():
    names = [tool.name for tool in chat.github_tools()]
    assert names == ["list_issues", "get_issue", "add_label", "post_comment"]


def test_no_github_tools_when_agent_yaml_turns_github_off(monkeypatch):
    monkeypatch.setattr(config.agent.connectors["github"], "enabled", False)
    assert chat.github_tools() == []


def test_auto_mcp_tool_runs_without_pausing_and_is_logged(monkeypatch, caplog):
    calls = []
    agent, _ = make_agent(
        monkeypatch,
        call("fetch", url="https://example.com"),
        AIMessage("It's the Example Domain page."),
        mcp_tools={"fetch": [fake_mcp_tool("fetch", calls)]},
    )
    with caplog.at_level(logging.INFO, logger="duo.chat"):
        result = ask(agent, "Summarize https://example.com")
    assert "__interrupt__" not in result
    assert calls == [("fetch", "https://example.com")]
    assert result["messages"][-2].content == "# Page at https://example.com"
    assert "Tool call fetch {'url': 'https://example.com'}" in caplog.text


def test_mcp_tool_its_manifest_does_not_list_pauses(monkeypatch):
    calls = []
    agent, _ = make_agent(
        monkeypatch,
        call("fetch_raw", url="https://example.com"),
        mcp_tools={"fetch": [fake_mcp_tool("fetch_raw", calls)]},
    )
    result = ask(agent, "Get the raw page")
    [request] = result["__interrupt__"][0].value["action_requests"]
    assert request["name"] == "fetch_raw"
    assert calls == []


def test_agent_has_github_and_mcp_tools(monkeypatch):
    agent, _ = make_agent(
        monkeypatch, AIMessage("hi"), mcp_tools={"fetch": [fake_mcp_tool("fetch", [])]}
    )
    tool_node = agent.get_graph().nodes["tools"].data
    assert sorted(tool_node.tools_by_name) == [
        "add_label",
        "fetch",
        "get_issue",
        "list_issues",
        "post_comment",
    ]
