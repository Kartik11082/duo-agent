"""channels/telegram.py: send a proposal, then resume the run from button taps.

The bot and the taps are fakes, the LLM is stubbed, the GitHub actions are
replaced, and the graph keeps state in memory. Nothing leaves the process.

Don't import duo.channels.telegram at the top of any test file: importing it
builds the real graph, which connects to Postgres. The tg fixture imports it
only after swapping in an in-memory build_graph.
"""

import asyncio
import importlib
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from telegram import Chat, InaccessibleMessage, Message

from duo.agents import triage
from duo.config import settings

ISSUE = {"repo": "owner/repo", "number": 7, "title": "Crash", "body": "It crashes."}
THREAD_ID = "owner/repo#7"


@pytest.fixture
def tg(monkeypatch, fake_llm):  # noqa: ARG001 (requested for its side effect)
    """The telegram module, using one in-memory graph for the whole test."""
    graph = triage.build_graph(InMemorySaver())
    monkeypatch.setattr(triage, "build_graph", lambda: graph)
    module = importlib.import_module("duo.channels.telegram")
    # After the first test the module is cached and not imported again,
    # so also set its graph directly.
    monkeypatch.setattr(module, "graph", graph)
    return module


@pytest.fixture
def github(monkeypatch):
    """Fake GitHub actions that record their calls. Replaces all of ACTIONS."""
    calls = []

    def add_label(repo, number, label):
        calls.append(("add_label", repo, number, label))
        return f"Labeled #{number} '{label}'"

    def post_comment(repo, number, text):
        calls.append(("post_comment", repo, number, text))
        return f"Commented on #{number}"

    actions = {"add_label": add_label, "post_comment": post_comment}
    monkeypatch.setattr(triage, "ACTIONS", actions)
    return calls


class FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text, reply_markup=None):
        self.sent.append(
            SimpleNamespace(chat_id=chat_id, text=text, markup=reply_markup)
        )


def send_proposal(tg):
    """Run triage_and_ask; return the sent message and its two button payloads."""
    bot = FakeBot()
    asyncio.run(tg.triage_and_ask(bot, THREAD_ID, ISSUE, "Heading"))
    [sent] = bot.sent
    approve, reject = sent.markup.inline_keyboard[0]
    return sent, approve.callback_data, reject.callback_data


def tap(tg, data, text, chat_id=None, inaccessible=False):
    """Tap a button: call on_button with a fake update. Returns the query.

    inaccessible=True taps a message older than 48 hours, which Telegram
    sends as an InaccessibleMessage.
    """
    chat_id = chat_id or settings.telegram_chat_id
    chat = Chat(id=chat_id, type="private")
    # Real message objects, because on_button checks isinstance(..., Message).
    if inaccessible:
        message = InaccessibleMessage(chat=chat, message_id=1)
    else:
        message = Message(message_id=1, date=datetime.now(UTC), chat=chat, text=text)
    query = SimpleNamespace(data=data, message=message, answers=[], edits=[])

    async def answer(text=None):
        query.answers.append(text)

    async def edit_message_text(text):
        query.edits.append(text)

    query.answer = answer
    query.edit_message_text = edit_message_text
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id), callback_query=query
    )
    asyncio.run(tg.on_button(update, None))
    return query


def test_proposal_is_sent_with_buttons(tg):
    sent, approve, reject = send_proposal(tg)
    assert sent.chat_id == settings.telegram_chat_id
    assert "add_label" in sent.text and "label: bug" in sent.text
    assert approve == f"yes:{THREAD_ID}"
    assert reject == f"no:{THREAD_ID}"


def test_approve_runs_the_actions(tg, github):
    sent, approve, _ = send_proposal(tg)
    query = tap(tg, approve, sent.text)
    assert query.edits[-1].endswith(
        "✅ Approved:\n• Labeled #7 'bug'\n• Commented on #7"
    )
    assert [call[0] for call in github] == ["add_label", "post_comment"]


def test_reject_runs_nothing(tg, github):
    sent, _, reject = send_proposal(tg)
    query = tap(tg, reject, sent.text)
    assert query.edits[-1].endswith("❌ Rejected, nothing done.")
    assert github == []


def test_failing_action_reports_failed(tg, monkeypatch):
    def add_label(repo, number, label):  # noqa: ARG001
        raise RuntimeError("GitHub is down")

    monkeypatch.setattr(triage, "ACTIONS", {"add_label": add_label})
    sent, approve, _ = send_proposal(tg)
    query = tap(tg, approve, sent.text)
    assert query.edits[-1].endswith("⚠️ Failed: GitHub is down")


def test_double_tap_is_already_handled(tg, github):
    sent, approve, _ = send_proposal(tg)
    tap(tg, approve, sent.text)
    second = tap(tg, approve, sent.text)
    assert second.edits == [f"{sent.text}\n\nAlready handled."]
    assert len(github) == 2  # the actions ran once, not twice


def test_other_chat_is_not_allowed(tg, github):
    sent, approve, _ = send_proposal(tg)
    query = tap(tg, approve, sent.text, chat_id=999)
    assert query.answers == ["Not allowed."]
    assert query.edits == []
    assert github == []


def test_tap_on_a_message_older_than_48_hours_is_expired(tg, github):
    sent, approve, _ = send_proposal(tg)
    query = tap(tg, approve, sent.text, inaccessible=True)
    assert query.answers == ["⌛ Expired, nothing done."]
    assert query.edits == []  # an inaccessible message can't be edited
    assert github == []
    config = {"configurable": {"thread_id": THREAD_ID}}
    assert tg.graph.get_state(config).next == ("approve",)  # still paused
