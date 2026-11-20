"""The connectors duo knows about, and which of them agent.yaml enables.

Each connector has a manifest (connectors/<id>/manifest.yaml): plain data that
says what it is, which secrets it needs, its tools with their default policies,
and the events it triggers on. A native connector also has Python functions
for its tools (NATIVE_TOOLS) and for turning its triggers into jobs
(NATIVE_TRIGGERS).

An MCP connector has no Python here: its manifest's `server:` says how to
start or reach its MCP server, and the server provides the tools.

To add a connector: write its manifest, add it to MANIFESTS and, if it is
native, add its functions to NATIVE_TOOLS and NATIVE_TRIGGERS. A connector is
on only if agent.yaml lists it under `connectors:` with enabled: true (the
default).

check() runs on import, so a mismatch between agent.yaml, the manifests and
the code stops duo at startup.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from duo.config import AgentConfig, Policy, Settings, agent
from duo.connectors.github import tools as github_tools
from duo.connectors.github import webhook as github_webhook


class StdioServer(BaseModel):
    """An MCP server duo starts as a child process, talking over stdin/stdout."""

    model_config = ConfigDict(extra="forbid")

    transport: Literal["stdio"]
    command: str  # e.g. "uvx"
    args: list[str] = Field(default_factory=list)


class HttpServer(BaseModel):
    """An MCP server duo connects to over HTTP (MCP's "streamable HTTP")."""

    model_config = ConfigDict(extra="forbid")

    transport: Literal["streamable_http"]
    url: str
    # A credential's name; its value is sent as "Authorization: Bearer <value>".
    bearer_token: str | None = None


# `transport` picks the model, so e.g. a stdio server without a command fails.
McpServer = Annotated[StdioServer | HttpServer, Field(discriminator="transport")]


class Manifest(BaseModel):
    """One connector's manifest. extra="forbid" turns a misspelled key into an error."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["native", "mcp"]
    auth: Literal["github_app", "api_key", "oauth2", "none"]
    credentials: list[str] = Field(default_factory=list)
    tools: dict[str, Policy]  # tool name -> default policy
    triggers: list[str] = Field(default_factory=list)  # e.g. "issues.opened"
    server: McpServer | None = None  # how to start or reach it; mcp only

    @model_validator(mode="after")
    def check_server(self) -> Self:
        if (self.kind == "mcp") != (self.server is not None):
            raise ValueError("`server` is required for kind: mcp, and only for it")
        token = getattr(self.server, "bearer_token", None)
        if token is not None and token not in self.credentials:
            raise ValueError(f"bearer_token {token!r} must be one of `credentials`")
        return self


def load_manifest(path: Path) -> Manifest:
    """Read and validate a manifest. Raises ValueError naming the bad key."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    try:
        return Manifest.model_validate(data)
    except ValidationError as error:
        raise ValueError(f"Invalid {path}:\n{error}") from error


HERE = Path(__file__).parent

# Connector id -> manifest.
MANIFESTS: dict[str, Manifest] = {
    "github": load_manifest(HERE / "github" / "manifest.yaml"),
    "fetch": load_manifest(HERE / "fetch" / "manifest.yaml"),
    "web_search": load_manifest(HERE / "web_search" / "manifest.yaml"),
}

# Connector id -> {tool name -> function}, for native connectors.
NATIVE_TOOLS: dict[str, dict[str, Callable[..., str]]] = {
    "github": github_tools.ACTIONS,
}

# Connector id -> {trigger -> function that turns its payload into a job}.
NATIVE_TRIGGERS: dict[str, dict[str, Callable[[dict], dict]]] = {
    "github": github_webhook.JOB_BUILDERS,
}


def is_enabled(connector_id: str) -> bool:
    connector = agent.connectors.get(connector_id)
    return connector is not None and connector.enabled


def tools_for(connector_id: str) -> dict[str, Callable[..., str]]:
    """The manifest's tools, as functions; empty if the connector is off."""
    if not is_enabled(connector_id):
        return {}
    functions = NATIVE_TOOLS[connector_id]
    return {name: functions[name] for name in MANIFESTS[connector_id].tools}


def triggers_for(connector_id: str) -> list[str]:
    """The manifest's triggers; empty if the connector is off."""
    if not is_enabled(connector_id):
        return []
    return MANIFESTS[connector_id].triggers


def check(config: AgentConfig) -> None:
    """Check agent.yaml against the manifests, and the manifests against the code."""
    for connector_id, connector in config.connectors.items():
        manifest = MANIFESTS.get(connector_id)
        if manifest is None:
            raise ValueError(
                f"agent.yaml: unknown connector {connector_id!r}. "
                f"Known: {', '.join(sorted(MANIFESTS))}"
            )
        unknown = sorted(set(connector.policy) - set(manifest.tools))
        if unknown:
            raise ValueError(
                f"agent.yaml: connectors.{connector_id}.policy names unknown "
                f"tool(s) {', '.join(unknown)}. "
                f"{connector_id} has: {', '.join(sorted(manifest.tools))}"
            )

    for connector_id, manifest in MANIFESTS.items():
        if manifest.id != connector_id:
            raise ValueError(
                f"Manifest registered as {connector_id!r} has id {manifest.id!r}"
            )
        unknown = sorted(set(manifest.credentials) - set(Settings.model_fields))
        if unknown:
            raise ValueError(
                f"{connector_id} manifest lists credential(s) that get_secret() "
                f"doesn't know: {', '.join(unknown)}"
            )
        if manifest.kind == "native":
            missing = sorted(
                set(manifest.tools) - set(NATIVE_TOOLS.get(connector_id, {}))
            )
            if missing:
                raise ValueError(
                    f"{connector_id} manifest lists tool(s) with no function "
                    f"in NATIVE_TOOLS: {', '.join(missing)}"
                )
            missing = sorted(
                set(manifest.triggers) - set(NATIVE_TRIGGERS.get(connector_id, {}))
            )
            if missing:
                raise ValueError(
                    f"{connector_id} manifest lists trigger(s) with no function "
                    f"in NATIVE_TRIGGERS: {', '.join(missing)}"
                )


check(agent)
