"""channels/telegram_chat.py: messages, approval buttons, taps and /new.

The bot, updates and taps are fakes; the agent has a scripted fake model and
an in-memory checkpointer; chat_threads is a dict. Nothing leaves the process.
"""

import asyncio
import importlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from chat_fakes import call, fake_github, make_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from telegram import Chat, Message

from duo import storage
from duo.agents import triage
from duo.config import settings

LABEL_7 = call("add_label", repo="o/r", number=7, label="documentation")


@pytest.fixture
def tgc(monkeypatch):
    """The telegram_chat module, with chat_threads kept in a dict.

    It imports channels/telegram.py, which builds the triage graph at import;
    an in-memory build_graph keeps that off Postgres.
    """
    build_graph = triage.build_graph
    monkeypatch.setattr(triage, "build_graph", lambda: build_graph(InMemorySaver()))
    threads = {}
    monkeypatch.setattr(storage, "get_chat_thread", threads.get)
    monkeypatch.setattr(storage, "set_chat_thread", threads.__setitem__)
    module = importlib.import_module("duo.channels.telegram_chat")
    module.threads = threads  # for the tests to look at
    return module


@pytest.fixture
def github(monkeypatch):
    return fake_github(monkeypatch)


class FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text, reply_markup=None):
        self.sent.append(
            SimpleNamespace(chat_id=chat_id, text=text, markup=reply_markup)
        )


def context_for(agent):
    return SimpleNamespace(bot=FakeBot(), bot_data={"chat_agent": agent})


def message_update(text, chat_id=None):
    """A fake update for a text message (or command) in the chat."""
    chat_id = chat_id or settings.telegram_chat_id
    replies = []

    async def reply_text(reply):
        replies.append(reply)

    message = SimpleNamespace(
        chat_id=chat_id, text=text, reply_text=reply_text, replies=replies
    )
    return SimpleNamespace(effective_chat=SimpleNamespace(id=chat_id), message=message)


def send(tgc, context, text, chat_id=None):
    """Send a text message; returns the fake message (its .replies)."""
    update = message_update(text, chat_id)
    asyncio.run(tgc.on_message(update, context))
    return update.message


def tap(tgc, context, data, text, sent_at=None):
    """Tap a button on a message; returns the fake query."""
    chat = Chat(id=settings.telegram_chat_id, type="private")
    message = Message(
        message_id=1, date=sent_at or datetime.now(UTC), chat=chat, text=text
    )
    query = SimpleNamespace(data=data, message=message, answers=[], edits=[])

    async def answer(text=None):
        query.answers.append(text)

    async def edit_message_text(text):
        query.edits.append(text)

    query.answer = answer
    query.edit_message_text = edit_message_text
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=settings.telegram_chat_id),
        callback_query=query,
    )
    asyncio.run(tgc.on_chat_button(update, context))
    return query


def buttons(sent):
    approve, reject = sent.markup.inline_keyboard[0]
    return approve.callback_data, reject.callback_data


def test_message_gets_the_agents_reply(tgc, monkeypatch, github):
    agent, _ = make_agent(
        monkeypatch, call("list_issues", repo="o/r"), AIMessage("One open: #7.")
    )
    context = context_for(agent)
    send(tgc, context, "What's open on o/r?")
    assert [m.text for m in context.bot.sent] == ["One open: #7."]
    assert github == [("list_issues", "o/r", "open")]


def test_write_tool_sends_the_exact_call_with_buttons(tgc, monkeypatch, github):
    agent, _ = make_agent(monkeypatch, LABEL_7)
    context = context_for(agent)
    send(tgc, context, "Label #7 as documentation")
    [sent] = context.bot.sent
    assert "add_label" in sent.text and "label: documentation" in sent.text
    approve, reject = buttons(sent)
    thread_id = tgc.threads[settings.telegram_chat_id]
    assert approve.startswith(f"chat:y:{thread_id}:")
    assert reject.startswith(f"chat:n:{thread_id}:")
    assert len(approve.encode()) <= 64  # Telegram's callback_data limit
    assert github == []


def test_approve_runs_the_tool_and_replies(tgc, monkeypatch, github):
    agent, _ = make_agent(monkeypatch, LABEL_7, AIMessage("Labeled #7."))
    context = context_for(agent)
    send(tgc, context, "Label #7 as documentation")
    [sent] = context.bot.sent
    query = tap(tgc, context, buttons(sent)[0], sent.text)
    assert query.edits == [f"{sent.text}\n\n✅ Approved"]
    assert github == [("add_label", "o/r", 7, "documentation")]
    assert context.bot.sent[-1].text == "Labeled #7."


def test_reject_runs_nothing(tgc, monkeypatch, github):
    agent, _ = make_agent(monkeypatch, LABEL_7, AIMessage("OK, left it."))
    context = context_for(agent)
    send(tgc, context, "Label #7 as documentation")
    [sent] = context.bot.sent
    query = tap(tgc, context, buttons(sent)[1], sent.text)
    assert query.edits == [f"{sent.text}\n\n❌ Rejected"]
    assert github == []
    assert context.bot.sent[-1].text == "OK, left it."


def test_double_tap_is_already_handled(tgc, monkeypatch, github):
    agent, _ = make_agent(monkeypatch, LABEL_7, AIMessage("Labeled #7."))
    context = context_for(agent)
    send(tgc, context, "Label #7 as documentation")
    [sent] = context.bot.sent
    tap(tgc, context, buttons(sent)[0], sent.text)
    second = tap(tgc, context, buttons(sent)[0], sent.text)
    assert second.edits == [f"{sent.text}\n\nAlready handled."]
    assert len(github) == 1


def test_old_button_does_not_answer_a_newer_pause(tgc, monkeypatch, github):
    # Approving the first label makes the model ask for a second one.
    label_8 = call("add_label", repo="o/r", number=8, label="bug")
    agent, _ = make_agent(monkeypatch, LABEL_7, label_8)
    context = context_for(agent)
    send(tgc, context, "Label #7 and #8")
    first = context.bot.sent[0]
    tap(tgc, context, buttons(first)[0], first.text)
    assert "number: 8" in context.bot.sent[-1].text  # the second pause

    stale = tap(tgc, context, buttons(first)[0], first.text)
    assert stale.edits == [f"{first.text}\n\nAlready handled."]
    assert github == [("add_label", "o/r", 7, "documentation")]  # #8 still waits


def test_expired_tap_rejects(tgc, monkeypatch, github):
    agent, _ = make_agent(monkeypatch, LABEL_7, AIMessage("OK."))
    context = context_for(agent)
    send(tgc, context, "Label #7 as documentation")
    [sent] = context.bot.sent
    long_ago = datetime.now(UTC) - timedelta(hours=settings.approval_timeout_hours + 1)
    query = tap(tgc, context, buttons(sent)[0], sent.text, sent_at=long_ago)
    assert query.edits == [f"{sent.text}\n\n⌛ Expired, rejected."]
    assert github == []


def test_message_while_paused_asks_for_a_tap_first(tgc, monkeypatch, github):
    agent, model = make_agent(monkeypatch, LABEL_7)
    context = context_for(agent)
    send(tgc, context, "Label #7 as documentation")
    message = send(tgc, context, "Hello?")
    assert message.replies == [
        "Approve or reject the action above first, or send /new to start over."
    ]
    assert len(model.seen) == 1  # the model wasn't called again
    assert github == []


def test_new_starts_a_fresh_thread(tgc, monkeypatch):
    agent, model = make_agent(monkeypatch, AIMessage("Hi."), AIMessage("Hi again."))
    context = context_for(agent)
    send(tgc, context, "First chat")
    old_thread = tgc.threads[settings.telegram_chat_id]

    update = message_update("/new")
    asyncio.run(tgc.new_chat(update, context))
    assert update.message.replies == ["New conversation started."]
    assert tgc.threads[settings.telegram_chat_id] != old_thread

    send(tgc, context, "Second chat")
    sent_to_model = [m.content for m in model.seen[-1]][1:]  # [0]: system prompt
    assert sent_to_model == ["Second chat"]  # the first chat isn't in this thread


def test_other_chat_is_ignored(tgc, monkeypatch):
    agent, model = make_agent(monkeypatch, AIMessage("Hi."))
    context = context_for(agent)
    send(tgc, context, "Hello", chat_id=999)
    assert context.bot.sent == []
    assert model.seen == []


def test_failed_run_is_reported(tgc, monkeypatch):
    agent, model = make_agent(monkeypatch)

    def ollama_down(*_args, **_kwargs):
        raise RuntimeError("Ollama is down")

    monkeypatch.setattr(type(model), "_generate", ollama_down)
    context = context_for(agent)
    send(tgc, context, "Hello")
    assert [m.text for m in context.bot.sent] == ["⚠️ Failed: Ollama is down"]
