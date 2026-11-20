"""Record the enabled MCP connectors' tool schemas into mcp_tools.json.

test_context_budget.py reads that file, so the budget is checked offline
against what the real servers send. Rerun this after changing a connector's
server or tools (it starts the servers, so it needs the network and the keys):

    uv run python tests/fixtures/record_mcp_tools.py
"""

import asyncio
import json
import sys
from pathlib import Path

from duo.connectors.mcp import McpServers

OUT = Path(__file__).with_name("mcp_tools.json")


async def record() -> dict:
    servers = McpServers()
    tools = await servers.start()
    await servers.stop()
    return {
        connector_id: [
            {"name": t.name, "description": t.description, "args_schema": t.args_schema}
            for t in connector_tools
        ]
        for connector_id, connector_tools in tools.items()
    }


if __name__ == "__main__":
    # A selector loop on Windows, as in apps/worker.py.
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    schemas = asyncio.run(record(), loop_factory=loop_factory)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(schemas, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {OUT}: {[t['name'] for ts in schemas.values() for t in ts]}")
