"""The triage graph, end to end, on the sample issue (no repo, so a dry run).

The LLM is stubbed (fake_llm in conftest.py) and state is kept in memory.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from duo.agents import triage

CONFIG = {"configurable": {"thread_id": "test-issue"}}


@pytest.fixture
def graph(fake_llm):  # noqa: ARG001 (requested for its side effect)
    return triage.build_graph(InMemorySaver())


def test_pauses_with_the_proposed_actions(graph):
    result = graph.invoke(triage.SAMPLE_ISSUE, CONFIG)
    proposal = result["__interrupt__"][0].value
    assert [a["name"] for a in proposal["actions"]] == ["add_label", "post_comment"]
    assert proposal["actions"][0]["args"] == {"label": "bug"}


def test_resume_yes_gives_dry_run_results(graph):
    graph.invoke(triage.SAMPLE_ISSUE, CONFIG)
    result = graph.invoke(Command(resume="yes"), CONFIG)
    assert result["approved"] is True
    assert len(result["results"]) == 2
    assert all(line.startswith("(dry run)") for line in result["results"])


def test_resume_no_is_rejected(graph):
    graph.invoke(triage.SAMPLE_ISSUE, CONFIG)
    result = graph.invoke(Command(resume="no"), CONFIG)
    assert result["approved"] is False
    assert result["results"] == []


def test_all_auto_runs_without_pausing(graph, monkeypatch):
    policy = {"add_label": "auto", "post_comment": "auto"}
    monkeypatch.setattr(triage, "PERMISSION_POLICY", policy)
    result = graph.invoke(triage.SAMPLE_ISSUE, CONFIG)
    assert "__interrupt__" not in result
    assert all(line.startswith("(dry run)") for line in result["results"])
