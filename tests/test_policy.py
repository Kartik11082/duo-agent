"""route_by_policy: an action runs on its own only if its policy is exactly "auto"."""

from duo.agents import triage


def state_with(*names):
    return {"actions": [{"name": name, "args": {}} for name in names]}


def test_default_policy_asks_for_approval():
    assert triage.route_by_policy(state_with("add_label", "post_comment")) == "approve"


def test_all_auto_executes(monkeypatch):
    policy = {"add_label": "auto", "post_comment": "auto"}
    monkeypatch.setattr(triage, "PERMISSION_POLICY", policy)
    assert triage.route_by_policy(state_with("add_label", "post_comment")) == "execute"


def test_one_unlisted_action_asks_for_approval(monkeypatch):
    monkeypatch.setattr(triage, "PERMISSION_POLICY", {"add_label": "auto"})
    assert triage.route_by_policy(state_with("add_label", "close_issue")) == "approve"


def test_mistyped_value_asks_for_approval(monkeypatch):
    monkeypatch.setattr(triage, "PERMISSION_POLICY", {"add_label": "Auto"})
    assert triage.route_by_policy(state_with("add_label")) == "approve"
