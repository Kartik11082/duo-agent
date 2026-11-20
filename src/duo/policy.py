"""Decides whether a tool runs on its own ("auto") or waits for a human ("approve").

The rule: only an exact "auto" skips approval. A missing entry, a typo, or any
other value means "approve", the safe default. (agent.yaml and the manifests
are validated at startup, so a typo there already stops duo; this rule is the
second line of defense.)

Where a tool's policy comes from, first match wins:
1. agent.yaml: connectors.<id>.policy.<tool>
2. the connector's manifest: tools.<tool>
"""

from typing import Literal

from duo.config import agent
from duo.connectors.registry import MANIFESTS


def decide(default: str | None, override: str | None) -> Literal["auto", "approve"]:
    """The override if there is one, else the default; "auto" only if exactly "auto"."""
    chosen = override if override is not None else default
    return "auto" if chosen == "auto" else "approve"


def policy_for(connector: str, tool: str) -> Literal["auto", "approve"]:
    """The policy for one connector's tool, from its manifest and agent.yaml."""
    manifest = MANIFESTS.get(connector)
    default = manifest.tools.get(tool) if manifest else None
    settings = agent.connectors.get(connector)
    override = settings.policy.get(tool) if settings else None
    return decide(default, override)
