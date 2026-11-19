"""The one place that knows how to build an LLM.

Everything else calls get_llm() and gets back a LangChain chat model. It
doesn't matter whether that model runs on Ollama or (later) Bedrock; the
interface is the same: .invoke(), .bind_tools(), .with_structured_output().

Smoke test:  uv run python -m duo.llm
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from duo.config import settings


def get_llm() -> BaseChatModel:
    """Build the chat model described by settings."""

    kwargs = {}

    if settings.model_provider == "ollama":
        kwargs["base_url"] = settings.ollama_base_url

    return init_chat_model(
        model=settings.model_name,
        model_provider=settings.model_provider,
        temperature=settings.temperature,
        **kwargs,
    )


if __name__ == "__main__":
    llm = get_llm()
    print(f"Using {settings.model_provider}:{settings.model_name}")
    reply = llm.invoke("In one sentence, what does a GitHub issue triage bot do?")
    print(reply.content)
