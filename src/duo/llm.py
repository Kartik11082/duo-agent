"""The one place that knows how to build an LLM.

Everything else calls get_llm() and gets back a LangChain chat model. It
doesn't matter whether that model runs on Ollama or (later) Bedrock; the
interface is the same: .invoke(), .bind_tools(), .with_structured_output().

Smoke test:  uv run python -m duo.llm
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from duo.config import agent, settings


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
