"""The chat agent: an LLM that answers Telegram messages by calling tools.

model -> tool calls? --no--> reply
             |
             `--yes--> any needs approval? --yes--> pause (interrupt) --approve/reject--.
                               |                                                     |
                               `--no----------------------------------> run tools <--'
                                                                            |
                                                                    back to the model

create_agent builds that loop as a LangGraph graph. Middleware hooks into it:
- trim_history: the model sees only the newest messages that fit its context.
- HumanInTheLoopMiddleware: pauses before tools whose policy isn't exactly "auto".
- ToolErrorMiddleware: a GitHub error goes back to the model as the tool's result.

Each Telegram chat is one thread, saved by the checkpointer like triage's runs,
so a pause for approval survives restarts.
"""

import atexit
from collections.abc import Callable, Sequence
from typing import cast

from github import GithubException
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolErrorMiddleware,
    wrap_model_call,
)
from langchain_core.messages import AnyMessage, HumanMessage, trim_messages
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from duo.config import agent, settings
from duo.connectors.github.chat_tools import CHAT_TOOLS
from duo.connectors.registry import tools_for
from duo.llm import get_llm
from duo.policy import policy_for

SYSTEM_PROMPT = """You are duo, a personal assistant that works on GitHub for the user.

- Repositories are written "owner/name". Use the tools to look things up; don't guess.
- Adding a label or posting a comment waits for the user to approve it. If they
  reject it, say so and don't try again unless they ask.
- Replies are read on a phone: keep them short.
"""

# Room left in the context window for the system prompt, the tool schemas and
# the reply. The rest (about 5000 of agent.yaml's 8192 tokens) is for the
# conversation.
RESERVED_TOKENS = 3000
# The window to assume when agent.yaml leaves num_ctx unset: Ollama's default.
DEFAULT_NUM_CTX = 4096


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


@wrap_model_call
def trim_history(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Send the model only the newest messages that fit. The checkpoint keeps all."""
    num_ctx = agent.model.num_ctx or DEFAULT_NUM_CTX
    # A cast, because trim_messages is typed to return list[BaseMessage]; it
    # returns some of the same message objects it was given.
    messages = cast(
        "list[AnyMessage]",
        trim_messages(
            request.messages,
            max_tokens=num_ctx - RESERVED_TOKENS,
            token_counter="approximate",  # about 4 characters per token
            strategy="last",  # keep the newest
            start_on="human",  # never start with a tool result cut off from its call
        ),
    )
    if not messages:
        # The current turn alone is over budget: send it whole rather than nothing.
        last_human = max(
            i for i, m in enumerate(request.messages) if isinstance(m, HumanMessage)
        )
        messages = request.messages[last_human:]
    return handler(request.override(messages=messages))


def github_error(error: Exception, _request: ToolCallRequest) -> str | None:
    """A GitHub API error (e.g. 404 for a missing issue) becomes the tool's result,
    so the model can tell the user. Anything else fails the run."""
    if isinstance(error, GithubException):
        message = error.data.get("message") if isinstance(error.data, dict) else None
        return f"GitHub error {error.status}: {message or 'no details'}"
    return None


def github_tools() -> list[BaseTool]:
    """The chat tools for the tools in GitHub's manifest; none if agent.yaml turns
    GitHub off. A manifest tool without a chat_tools wrapper is a KeyError here
    (test_github_tools.py checks they match)."""
    by_name = {tool.name: tool for tool in CHAT_TOOLS}
    return [by_name[name] for name in tools_for("github")]


def build_agent(checkpointer: BaseCheckpointSaver | None = None):
    """Build the agent, saving threads to Postgres unless given another checkpointer.

    Tests pass InMemorySaver() so they don't need a database.
    """
    if checkpointer is None:
        # The same setup as triage.build_graph(); see the comments there.
        pool: ConnectionPool[Connection[DictRow]] = ConnectionPool(
            settings.database_url,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
            open=True,
        )
        atexit.register(pool.close)
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()

    tools = github_tools()
    return create_agent(
        get_llm(),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            trim_history,
            HumanInTheLoopMiddleware(interrupt_on=build_interrupt_on(tools)),
            ToolErrorMiddleware(github_error),
        ],
        checkpointer=checkpointer,
    )
