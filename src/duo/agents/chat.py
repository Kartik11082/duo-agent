"""The chat agent: an LLM that answers Telegram messages by calling tools.

Tools whose policy isn't exactly "auto" pause the agent for approval.
"""

from collections.abc import Sequence

from langchain.agents.middleware import InterruptOnConfig
from langchain_core.tools import BaseTool

from duo.policy import policy_for


def needs_approval(tool_name: str) -> bool:
    """Only an exact "auto" runs on its own (policy.py: manifest, then agent.yaml)."""
    return policy_for("github", tool_name) != "auto"


def build_interrupt_on(
    tools: Sequence[BaseTool],
) -> dict[str, bool | InterruptOnConfig]:
    """HumanInTheLoopMiddleware's interrupt_on: the tools that must pause.

    Built from the agent's own tools, not from the policy's keys: the middleware
    runs any tool it isn't told about without asking, so a tool missing from the
    policy must still be listed here.
    """
    return {
        # Telegram only has Approve and Reject buttons, so no "edit" or "respond".
        tool.name: {"allowed_decisions": ["approve", "reject"]}
        for tool in tools
        if needs_approval(tool.name)
    }
