"""The chat agent's fixed cost per model call fits in its context reserve.

Every model call sends the system prompt and every tool's schema. Those, plus
Ollama's tool template and the reply, must fit in chat.RESERVED_TOKENS; the
conversation gets the rest of num_ctx (chat.trim_history). Tokens are
counted as about 4 characters each, like trim_history's "approximate" counter.

The MCP schemas come from fixtures/mcp_tools.json, recorded from the real
servers (fixtures/record_mcp_tools.py), so this runs offline.
"""

import json
import math
from pathlib import Path

from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from duo.agents import chat
from duo.config import agent
from duo.connectors.github.chat_tools import CHAT_TOOLS
from duo.connectors.registry import MANIFESTS

RECORDED = json.loads(
    (Path(__file__).parent / "fixtures" / "mcp_tools.json").read_text("utf-8")
)
# Left of the reserve for Ollama's tool template and a short reply.
REPLY_TOKENS = 600


def tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def recorded_tools():
    """The recorded MCP tools, built like langchain-mcp-adapters builds them."""
    return [
        StructuredTool(
            name=t["name"],
            description=t["description"],
            args_schema=t["args_schema"],
            coroutine=lambda: None,  # never called here
        )
        for connector_tools in RECORDED.values()
        for t in connector_tools
    ]


def schema_tokens(tool) -> int:
    """What bind_tools sends the model for this tool."""
    return tokens(json.dumps(convert_to_openai_tool(tool)))


def test_recording_has_exactly_the_listed_mcp_tools():
    # Rerun fixtures/record_mcp_tools.py if this fails after a manifest change.
    listed = {cid: list(m.tools) for cid, m in MANIFESTS.items() if m.kind == "mcp"}
    recorded = {cid: [t["name"] for t in ts] for cid, ts in RECORDED.items()}
    assert recorded == listed


def test_prompt_and_all_tool_schemas_fit_the_reserve():
    tools = [*CHAT_TOOLS, *recorded_tools()]
    fixed = tokens(chat.SYSTEM_PROMPT) + sum(schema_tokens(t) for t in tools)
    # 2131 today (tavily_search alone is 1259), of 2400.
    assert fixed <= chat.RESERVED_TOKENS - REPLY_TOKENS, fixed


def test_a_capped_tool_result_fits_the_conversation_budget():
    conversation = agent.model.num_ctx - chat.RESERVED_TOKENS  # about 5200
    assert tokens("x" * chat.MAX_TOOL_RESULT_CHARS) <= conversation // 2
