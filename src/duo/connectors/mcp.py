"""MCP connectors: start or connect to their servers, and get their tools.

An MCP connector's manifest says how to reach its server (`server:`) and which
of the server's tools the agent may use (`tools:`). McpServers opens a session
to every enabled MCP connector when the worker starts, and closes them when it
stops:

    servers = McpServers()
    tools = await servers.start()  # {connector id: [tools]}
    ...
    await servers.stop()

A disabled connector's server is never started. A stdio server gets only a few
safe environment variables (PATH and the like), not duo's secrets.
"""

import asyncio
import logging
from contextlib import AsyncExitStack

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection
from langchain_mcp_adapters.tools import load_mcp_tools

from duo.config import get_secret
from duo.connectors.registry import MANIFESTS, HttpServer, Manifest, is_enabled

log = logging.getLogger("duo.mcp")

Tools = dict[str, list[BaseTool]]  # connector id -> its tools


def enabled_mcp_connectors() -> list[str]:
    return [
        connector_id
        for connector_id, manifest in MANIFESTS.items()
        if manifest.kind == "mcp" and is_enabled(connector_id)
    ]


def connection_for(manifest: Manifest) -> Connection:
    """The adapter's connection settings for a manifest's `server:`.

    Raises ValueError if one of its credentials isn't set.
    """
    missing = [name.upper() for name in manifest.credentials if not get_secret(name)]
    if missing:
        raise ValueError(
            f"{manifest.id} needs {', '.join(missing)} in .env, or set "
            f"connectors.{manifest.id}.enabled: false in agent.yaml"
        )
    server = manifest.server
    if server is None:
        raise ValueError(f"{manifest.id} is not an MCP connector")
    if isinstance(server, HttpServer):
        headers = {}
        if server.bearer_token is not None:
            headers["Authorization"] = f"Bearer {get_secret(server.bearer_token)}"
        return {"transport": "streamable_http", "url": server.url, "headers": headers}
    return {"transport": "stdio", "command": server.command, "args": server.args}


def listed_tools(connector_id: str, server_tools: list[BaseTool]) -> list[BaseTool]:
    """The server's tools that the manifest lists, in the manifest's order.

    The others are left out: each tool costs context tokens, and a tool nobody
    reviewed shouldn't reach the agent. A listed tool the server doesn't have
    (renamed, say) is an error, so it can't quietly go missing.
    """
    listed = MANIFESTS[connector_id].tools
    by_name = {tool.name: tool for tool in server_tools}
    missing = sorted(set(listed) - set(by_name))
    if missing:
        raise ValueError(
            f"{connector_id}: the MCP server has no tool(s) {', '.join(missing)}. "
            f"It has: {', '.join(sorted(by_name))}"
        )
    unlisted = sorted(set(by_name) - set(listed))
    if unlisted:
        log.info("%s: leaving out unlisted tool(s) %s", connector_id, unlisted)
    return [by_name[name] for name in listed]


class McpServers:
    """Sessions to the enabled MCP connectors, open from start() until stop().

    An MCP session must be closed by the same asyncio task that opened it, and
    the worker's post_init and post_shutdown run in different tasks. So one
    task of our own (_serve) opens them all, hands back the tools, and waits
    for stop() to tell it to close them.
    """

    def __init__(self) -> None:
        self._stopping = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> Tools:
        """Open every session and return the listed tools. Raises if one fails."""
        ready: asyncio.Future[Tools] = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(self._serve(ready))
        return await ready

    async def stop(self) -> None:
        """Close the sessions (and stop the servers duo started)."""
        self._stopping.set()
        if self._task is not None:
            await self._task

    async def _serve(self, ready: asyncio.Future[Tools]) -> None:
        try:
            connector_ids = enabled_mcp_connectors()
            # Checks every connector's credentials before starting anything.
            client = MultiServerMCPClient(
                {cid: connection_for(MANIFESTS[cid]) for cid in connector_ids}
            )
            async with AsyncExitStack() as sessions:
                tools: Tools = {}
                for connector_id in connector_ids:
                    log.info("Starting MCP connector %s", connector_id)
                    session = await sessions.enter_async_context(
                        client.session(connector_id)
                    )
                    server_tools = await load_mcp_tools(
                        session, server_name=connector_id
                    )
                    tools[connector_id] = listed_tools(connector_id, server_tools)
                ready.set_result(tools)
                await self._stopping.wait()
        except Exception as error:
            if ready.done():  # failed while closing: let stop() see it
                raise
            ready.set_exception(error)  # failed while starting: start() raises
