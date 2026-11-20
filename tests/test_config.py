"""agent.yaml: the repo's file loads, and mistakes fail at load with the key named."""

from pathlib import Path

import pytest

from duo import llm
from duo.config import agent, load_agent_config

GOOD = """
model: {provider: ollama, name: "qwen2.5:7b"}
connectors:
  github: {enabled: true, policy: {post_comment: auto}}
"""


def write(tmp_path, text):
    path = tmp_path / "agent.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_repo_agent_yaml_loads():
    config = load_agent_config(Path("agent.yaml"))
    assert config.model.provider == "ollama"
    assert config.connectors["github"].enabled is True


def test_good_config_loads_with_defaults(tmp_path):
    config = load_agent_config(write(tmp_path, GOOD))
    assert config.model.temperature == 0.0
    assert config.model.num_ctx is None
    assert config.connectors["github"].policy == {"post_comment": "auto"}


def test_misspelled_policy_value_fails(tmp_path):
    bad = GOOD.replace("post_comment: auto", "post_comment: Auto")
    with pytest.raises(ValueError, match=r"connectors\.github\.policy\.post_comment"):
        load_agent_config(write(tmp_path, bad))


def test_unknown_key_fails(tmp_path):
    bad = GOOD.replace("enabled: true", "enabeld: true")
    with pytest.raises(ValueError, match=r"connectors\.github\.enabeld"):
        load_agent_config(write(tmp_path, bad))


def test_missing_model_fails(tmp_path):
    with pytest.raises(ValueError, match="model"):
        load_agent_config(write(tmp_path, "connectors: {}\n"))


def test_num_ctx_is_left_to_ollama_unless_set(monkeypatch):
    assert llm.get_llm().num_ctx is None
    monkeypatch.setattr(agent.model, "num_ctx", 8192)
    assert llm.get_llm().num_ctx == 8192
