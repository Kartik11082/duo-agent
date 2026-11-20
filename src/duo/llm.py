"""The one place that knows how to build an LLM.

Everything else calls get_llm() and gets back a LangChain chat model. It
doesn't matter whether that model runs on Ollama or (later) Bedrock; the
interface is the same: .invoke(), .bind_tools(), .with_structured_output().

Smoke test:  uv run python -m duo.llm
"""

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from duo.config import agent, get_secret, settings


def enable_tracing() -> bool:
    """Turn on LangSmith tracing if LANGSMITH_API_KEY is set. Returns whether it's on.

    Call it once at startup, before any LLM runs: LangChain reads its tracing
    settings from os.environ, and caches them on first use. pydantic-settings
    reads .env but doesn't put the values in os.environ, so copy them over.
    The project name comes from agent.yaml's `tracing:`.
    """
    api_key = get_secret("langsmith_api_key")
    if not api_key:
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = agent.tracing.project
    return True


def get_llm() -> BaseChatModel:
    """Build the chat model described by agent.yaml's `model:`."""
    model = agent.model
    kwargs = {}

    if model.provider == "ollama":
        kwargs["base_url"] = settings.ollama_base_url
        # Passed only when set, so Ollama otherwise keeps its own default.
        if model.num_ctx is not None:
            kwargs["num_ctx"] = model.num_ctx

    return init_chat_model(
        model=model.name,
        model_provider=model.provider,
        temperature=model.temperature,
        **kwargs,
    )


if __name__ == "__main__":
    llm = get_llm()
    print(f"Using {agent.model.provider}:{agent.model.name}")
    reply = llm.invoke("In one sentence, what does a GitHub issue triage bot do?")
    print(reply.content)
