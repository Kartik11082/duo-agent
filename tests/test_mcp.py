"""connectors/mcp.py: which MCP servers start, which of their tools the agent
gets, and that sessions close on stop.

The MCP client is a fake that records sessions opening and closing, so no
server starts and nothing leaves the process.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import pytest
from langchain_core.tools import StructuredTool

from duo.config import agent, settings
from duo.connectors import mcp
from duo.connectors.registry import MANIFESTS


def server_tool(name):
    """A tool like the adapter makes: async only."""

    async def run():
        return name

    return StructuredTool.from_function(coroutine=run, name=name, description=name)


@pytest.fixture
def servers(monkeypatch):
    """Fake MCP servers. Set `servers.tools[connector id]` to the tools one has;
    `servers.events` records ("open"/"close", connector id)."""
    fake = type("FakeServers", (), {"tools": {}, "events": []})()

    class FakeClient:
        def __init__(self, connections):
            self.connections = connections

        @asynccontextmanager
        async def session(self, connector_id):
            fake.events.append(("open", connector_id))
            yield connector_id  # the fake session is just its name
            fake.events.append(("close", connector_id))

    async def load_mcp_tools(session, server_name):  # noqa: ARG001
        return fake.tools[session]

    monkeypatch.setattr(mcp, "MultiServerMCPClient", FakeClient)
    monkeypatch.setattr(mcp, "load_mcp_tools", load_mcp_tools)
    return fake


def enable(monkeypatch, *connector_ids):
    """Turn on exactly these MCP connectors, as agent.yaml would."""
    for connector_id in ("fetch", "web_search"):
        on = connector_id in connector_ids
        monkeypatch.setattr(agent.connectors[connector_id], "enabled", on)


def start_and_stop():
    """Start the servers, then stop them; returns what start() gave."""

    async def run():
        servers = mcp.McpServers()
        tools = await servers.start()
        await servers.stop()
        return tools

    return asyncio.run(run())


def test_disabled_connectors_are_not_started(monkeypatch, servers):
    enable(monkeypatch)
    assert start_and_stop() == {}
    assert servers.events == []


def test_only_listed_tools_are_kept_and_sessions_close(monkeypatch, servers, caplog):
    enable(monkeypatch, "fetch")
    servers.tools["fetch"] = [server_tool("fetch_raw"), server_tool("fetch")]
    with caplog.at_level(logging.INFO, logger="duo.mcp"):
        tools = start_and_stop()
    assert {cid: [t.name for t in ts] for cid, ts in tools.items()} == {
        "fetch": ["fetch"]
    }
    assert "fetch_raw" in caplog.text  # logged as left out
    assert servers.events == [("open", "fetch"), ("close", "fetch")]


def test_listed_tool_the_server_lacks_fails_startup(monkeypatch, servers):
    enable(monkeypatch, "fetch")
    servers.tools["fetch"] = [server_tool("get_page")]
    with pytest.raises(ValueError, match=r"no tool\(s\) fetch. It has: get_page"):
        start_and_stop()


def test_missing_key_fails_before_any_server_starts(monkeypatch, servers):
    enable(monkeypatch, "fetch", "web_search")  # tavily_api_key is blank in tests
    with pytest.raises(ValueError, match="web_search needs TAVILY_API_KEY"):
        start_and_stop()
    assert servers.events == []


def test_connection_settings_come_from_the_manifest(monkeypatch):
    assert mcp.connection_for(MANIFESTS["fetch"]) == {
        "transport": "stdio",
        "command": "uvx",
        "args": ["mcp-server-fetch==2026.11.18"],
    }
    monkeypatch.setattr(settings, "tavily_api_key", "test-key")
    assert mcp.connection_for(MANIFESTS["web_search"]) == {
        "transport": "streamable_http",
        "url": "https://mcp.tavily.com/mcp/",
        "headers": {"Authorization": "Bearer test-key"},
    }
