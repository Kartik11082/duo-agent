"""llm.enable_tracing: LangSmith is on only when LANGSMITH_API_KEY is set."""

import os

import pytest

from duo.config import agent, settings
from duo.llm import enable_tracing

TRACING_VARS = ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT")


@pytest.fixture
def tracing_env(monkeypatch):
    """Blank the tracing env vars; monkeypatch puts back what was there after."""
    for name in TRACING_VARS:
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
