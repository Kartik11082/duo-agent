"""Connector manifests, and the startup check of agent.yaml against them."""

import pytest

from duo.config import AgentConfig
from duo.connectors import registry


def config_with(connectors):
    return AgentConfig.model_validate(
        {"model": {"provider": "ollama", "name": "m"}, "connectors": connectors}
    )


def test_github_manifest_matches_the_code():
    manifest = registry.MANIFESTS["github"]
    assert manifest.tools == {
        "list_issues": "auto",
        "get_issue": "auto",
        "add_label": "approve",
        "post_comment": "approve",
    }
    assert manifest.triggers == ["issues.opened"]
    assert set(manifest.tools) == set(registry.NATIVE_TOOLS["github"])


def test_repo_agent_yaml_passes_the_check():
    registry.check(registry.agent)  # the config loaded from agent.yaml


def test_unknown_connector_fails():
    with pytest.raises(ValueError, match="unknown connector 'githb'"):
        registry.check(config_with({"githb": {}}))


def test_unknown_tool_in_policy_fails():
    config = config_with({"github": {"policy": {"post_coment": "auto"}}})
    with pytest.raises(ValueError, match=r"unknown tool.*post_coment"):
        registry.check(config)


def test_manifest_tool_without_a_function_fails(monkeypatch):
    monkeypatch.setitem(registry.NATIVE_TOOLS, "github", {"add_label": print})
    with pytest.raises(ValueError, match=r"no function.*post_comment"):
        registry.check(config_with({}))


def test_bad_manifest_fails(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text("id: x\nkind: plugin\nauth: none\ntools: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="kind"):
        registry.load_manifest(path)


def test_manifest_trigger_without_a_job_builder_fails(monkeypatch):
    monkeypatch.setitem(registry.NATIVE_TRIGGERS, "github", {})
    with pytest.raises(ValueError, match=r"no function.*issues\.opened"):
        registry.check(config_with({}))


def test_enabled_github_gives_its_tools_and_triggers():
    tools = {"list_issues", "get_issue", "add_label", "post_comment"}
    assert set(registry.tools_for("github")) == tools
    assert registry.triggers_for("github") == ["issues.opened"]


def test_disabled_or_unlisted_connector_gives_nothing(monkeypatch):
    monkeypatch.setattr(registry.agent.connectors["github"], "enabled", False)
    assert registry.tools_for("github") == {}
    assert registry.triggers_for("github") == []
    monkeypatch.delitem(registry.agent.connectors, "github")
    assert registry.tools_for("github") == {}


def test_manifest_credential_get_secret_does_not_know_fails(monkeypatch):
    manifest = registry.MANIFESTS["github"]
    monkeypatch.setattr(manifest, "credentials", ["github_app_key"])
    with pytest.raises(ValueError, match=r"credential.*github_app_key"):
        registry.check(config_with({}))
