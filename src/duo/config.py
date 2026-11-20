"""All settings for duo, in one place.

Two sources:
- agent.yaml: the non-secret settings (model, connectors, policy overrides),
  validated into `agent` below. A typo there stops duo at startup.
- Environment variables or a local .env file: secrets and per-machine values,
  loaded into `settings`. Anything that's missing falls back to the defaults
  below.

Both files are read from the current directory, so run duo from the repo root.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# What a tool's policy may be. Only "auto" runs without asking (see policy.py).
Policy = Literal["auto", "approve"]


class ModelConfig(BaseModel):
    """Which LLM to use. Switching to Bedrock later is an agent.yaml change."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    name: str
    temperature: float = 0.0
    num_ctx: int | None = None  # None: the provider's default context size


class TracingConfig(BaseModel):
    """LangSmith tracing. It's on only when LANGSMITH_API_KEY is set in .env."""

    model_config = ConfigDict(extra="forbid")

    project: str = "duo"  # the LangSmith project the runs go to


class ConnectorConfig(BaseModel):
    """One entry under `connectors:`, keyed by the connector's manifest id."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # Tool name -> policy, overriding the manifest's default for that tool.
    policy: dict[str, Policy] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """The whole of agent.yaml. extra="forbid" turns a misspelled key into an error."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    connectors: dict[str, ConnectorConfig] = Field(default_factory=dict)
    tracing: TracingConfig = Field(default_factory=TracingConfig)


def load_agent_config(path: Path) -> AgentConfig:
    """Read and validate agent.yaml. Raises ValueError naming the bad key."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    try:
        return AgentConfig.model_validate(data)
    except ValidationError as error:
        raise ValueError(f"Invalid {path}:\n{error}") from error


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Where Ollama listens, when agent.yaml's model.provider is "ollama".
    ollama_base_url: str = "http://localhost:11434"

    # Optional LangSmith tracing (llm.enable_tracing): on only when the key is set.
    langsmith_api_key: str = ""
    # false keeps the key but turns tracing off.
    langsmith_tracing: bool = True
    # Passed on to LangSmith as LANGSMITH_* variables when tracing is on.
    # Empty means LangSmith's default.
    langsmith_endpoint: str = ""
    langsmith_agent_id: str = ""
    langsmith_agent_environment: str = ""

    # Telegram approvals (worker.py). Empty defaults so the rest of the
    # project still runs without a bot; worker.py checks they are set.
    telegram_bot_token: str = ""
    telegram_chat_id: int = 0  # only this chat may approve or reject
    approval_timeout_hours: float = 24.0

    # Postgres for the LangGraph checkpointer and the jobs table, e.g.
    # postgresql://user:password@127.0.0.1:5432/duo. Set it in .env.
    database_url: str = ""
    # Read by docker-compose.yml, not by Python. Declared here only because
    # unknown keys in .env are an error (which catches typos in the others).
    postgres_user: str = ""
    postgres_password: str = ""
    postgres_db: str = ""

    # GitHub webhooks (connectors/github/webhook.py). Must match the secret set on the webhook.
    github_webhook_secret: str = ""

    # GitHub App (connectors/github/auth.py): the agent acts on GitHub as its own bot account.
    github_app_id: int = 0
    # Keep the .pem outside this folder: OneDrive would sync it to the cloud.
    github_private_key_path: str = ""

    # Tavily web search (connectors/web_search): needed only if it's enabled.
    tavily_api_key: str = ""


# Import these everywhere: `from duo.config import agent, settings`
agent = load_agent_config(Path("agent.yaml"))
settings = Settings()


def get_secret(name: str) -> str:
    """A secret by its .env name in lowercase, e.g. get_secret("github_webhook_secret").

    Reads `settings` (environment variables or .env), so tests that blank a
    setting also blank the secret. On AWS (P8) this is where SSM Parameter
    Store will be read instead; callers won't change.
    """
    if name not in Settings.model_fields:
        raise KeyError(f"Unknown secret {name!r}: not a setting in config.Settings")
    return str(getattr(settings, name))
