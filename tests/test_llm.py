"""llm.enable_tracing: LangSmith is on only when LANGSMITH_API_KEY is set, and
traces to an agent address or a project, never both."""

import os

import pytest

from duo.config import agent, settings
from duo.llm import enable_tracing, tracing_destination

TRACING_VARS = ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT")
EXTRA_VARS = ("LANGSMITH_ENDPOINT", "LANGSMITH_AGENT_ID", "LANGSMITH_AGENT_ENVIRONMENT")


@pytest.fixture
def tracing_env(monkeypatch):
    """Blank the tracing env vars; monkeypatch puts back what was there after."""
    for name in TRACING_VARS + EXTRA_VARS:
        monkeypatch.setenv(name, "")


def test_tracing_stays_off_without_a_key(tracing_env):  # noqa: ARG001
    assert enable_tracing() is False
    assert [os.environ[name] for name in TRACING_VARS] == ["", "", ""]


def test_tracing_turns_on_with_a_key(tracing_env, monkeypatch):  # noqa: ARG001
    monkeypatch.setattr(settings, "langsmith_api_key", "test-key")
    monkeypatch.setattr(agent.tracing, "project", "duo-test")
    assert enable_tracing() is True
    assert [os.environ[name] for name in TRACING_VARS] == [
        "true",
        "test-key",
        "duo-test",
    ]


def test_repo_agent_yaml_names_the_project():
    assert agent.tracing.project == "duo"


def test_tracing_false_keeps_it_off_even_with_a_key(tracing_env, monkeypatch):  # noqa: ARG001
    monkeypatch.setattr(settings, "langsmith_api_key", "test-key")
    monkeypatch.setattr(settings, "langsmith_tracing", False)
    assert enable_tracing() is False
    assert os.environ["LANGSMITH_TRACING"] == ""


def test_agent_address_replaces_the_project(tracing_env, monkeypatch):  # noqa: ARG001
    # LangSmith refuses a run whose environment names both a project and an agent.
    monkeypatch.setattr(settings, "langsmith_api_key", "test-key")
    monkeypatch.setattr(settings, "langsmith_agent_id", "duo")
    monkeypatch.setattr(settings, "langsmith_agent_environment", "local")
    assert enable_tracing() is True
    assert os.environ["LANGSMITH_AGENT_ID"] == "duo"
    assert os.environ["LANGSMITH_AGENT_ENVIRONMENT"] == "local"
    assert "LANGSMITH_PROJECT" not in os.environ
    assert tracing_destination() == "agent duo (local)"


def test_half_an_agent_address_fails(tracing_env, monkeypatch):  # noqa: ARG001
    monkeypatch.setattr(settings, "langsmith_api_key", "test-key")
    monkeypatch.setattr(settings, "langsmith_agent_id", "duo")
    with pytest.raises(ValueError, match="both LANGSMITH_AGENT_ID and"):
        enable_tracing()


def test_endpoint_is_passed_on_only_when_set(tracing_env, monkeypatch):  # noqa: ARG001
    monkeypatch.setattr(settings, "langsmith_api_key", "test-key")
    assert enable_tracing() is True
    assert os.environ["LANGSMITH_ENDPOINT"] == ""  # left as the fixture blanked it
    assert tracing_destination() == "project duo"
    monkeypatch.setattr(settings, "langsmith_endpoint", "https://eu.example")
    enable_tracing()
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://eu.example"
