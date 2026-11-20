"""Telegram chat: text messages go to the chat agent, and taps answer its pauses.

- on_message: run the agent on the chat's current thread and send its reply.
  If it pauses for approval, send the exact tool calls with Approve/Reject.
- new_chat (/new): start a fresh thread for this chat.
- on_chat_button: resume the paused agent with "approve" or "reject".

apps/worker.py builds the agent and puts it in bot_data["chat_agent"].

Button data is "chat:<y|n>:<thread_id>:<pause>", at most 64 bytes (Telegram's
limit). The "chat:" prefix keeps these taps away from triage's buttons.
<pause> is the start of the pause's interrupt id: a thread can pause many
times, and an old button must not answer a newer pause you haven't seen.
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Interrupt
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.ext import ContextTypes

from duo import storage
from duo.channels.telegram import describe_actions, is_allowed
from duo.config import settings

log = logging.getLogger("duo.worker")

BUTTON_PREFIX = "chat"
PAUSE_ID_CHARS = 8
TELEGRAM_MAX_CHARS = 4096  # longer messages are refused


def new_thread(chat_id: int) -> str:
    """Start a fresh thread for the chat and make it the current one."""
    thread_id = f"chat-{chat_id}-{uuid.uuid4().hex[:8]}"
    storage.set_chat_thread(chat_id, thread_id)
    return thread_id


def current_thread(chat_id: int) -> str:
    return storage.get_chat_thread(chat_id) or new_thread(chat_id)


def thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


async def pending_pause(agent, thread_id: str) -> Interrupt | None:
    """The pause on this thread that's waiting for a tap, or None."""
    interrupts = (await agent.aget_state(thread_config(thread_id))).interrupts
    return interrupts[0] if interrupts else None


def approval_buttons(thread_id: str, pause: Interrupt) -> InlineKeyboardMarkup:
    pause_id = pause.id[:PAUSE_ID_CHARS]
    approve = f"{BUTTON_PREFIX}:y:{thread_id}:{pause_id}"
    reject = f"{BUTTON_PREFIX}:n:{thread_id}:{pause_id}"
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ Approve", callback_data=approve),
                InlineKeyboardButton("❌ Reject", callback_data=reject),
            ]
        ]
    )


async def run_and_reply(bot: Bot, agent, thread_id: str, agent_input) -> None:
    """Run the agent until it replies or pauses, and send that to the chat.

    agent_input is the new user message, or a Command that resumes a pause.
    """
    chat_id = settings.telegram_chat_id
    try:
        # Async, so the bot stays responsive while the LLM runs (agents/chat.py).
        result = await agent.ainvoke(agent_input, thread_config(thread_id))
    except Exception as error:
        # E.g. Ollama down. Other GitHub errors already went back to the model.
        log.exception("Chat run on %s failed", thread_id)
        await bot.send_message(chat_id=chat_id, text=f"⚠️ Failed: {error}")
        return

    if "__interrupt__" in result:
        pause = result["__interrupt__"][0]
        actions = pause.value["action_requests"]  # [{"name", "args", ...}]
        await bot.send_message(
            chat_id=chat_id,
            text=f"Approve this?\n{describe_actions(actions)}",
            reply_markup=approval_buttons(thread_id, pause),
        )
        log.info("Chat %s paused with %d action(s)", thread_id, len(actions))
        return

    reply = result["messages"][-1].text or "(no reply)"
    await bot.send_message(chat_id=chat_id, text=reply[:TELEGRAM_MAX_CHARS])


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A text message: the next user turn in this chat's thread."""
    if not is_allowed(update):
        return
    assert update.message is not None and update.message.text is not None
    agent = context.bot_data["chat_agent"]
    thread_id = current_thread(update.message.chat_id)

    # A new message now would leave the paused tool call without an answer.
    if await pending_pause(agent, thread_id) is not None:
        await update.message.reply_text(
            "Approve or reject the action above first, or send /new to start over."
        )
        return

    user_turn = {"messages": [{"role": "user", "content": update.message.text}]}
    await run_and_reply(context.bot, agent, thread_id, user_turn)


async def new_chat(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """/new: forget the conversation. The old thread stays in the checkpointer."""
    if not is_allowed(update):
        return
    assert update.message is not None  # always set for a command
    new_thread(update.message.chat_id)
    await update.message.reply_text("New conversation started.")


async def on_chat_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Approve/Reject tap: resume the paused agent named in the button."""
    query = update.callback_query
    assert query is not None  # always set for a button tap
    if not is_allowed(update):
        await query.answer("Not allowed.")
        return
    # Older than 48 hours: Telegram can't edit it any more. The run stays paused.
    if not isinstance(query.message, Message):
        await query.answer("⌛ Expired, nothing done.")
        return
    await query.answer()  # stops the loading spinner on the button

    assert query.data is not None  # our buttons always carry data
    _, answer, thread_id, pause_id = query.data.split(":")
    agent = context.bot_data["chat_agent"]
    original = query.message.text

    # A double tap, or a button from an earlier pause on this thread.
    pause = await pending_pause(agent, thread_id)
    if pause is None or not pause.id.startswith(pause_id):
        await query.edit_message_text(f"{original}\n\nAlready handled.")
        return

    # Lazy expiry, as for triage: checked only when someone taps.
    age = datetime.now(UTC) - query.message.date
    if age > timedelta(hours=settings.approval_timeout_hours):
        decision, outcome = "reject", "⌛ Expired, rejected."
    elif answer == "y":
        decision, outcome = "approve", "✅ Approved"
    else:
        decision, outcome = "reject", "❌ Rejected"

    # Editing first removes the buttons, so a second tap is less likely.
    await query.edit_message_text(f"{original}\n\n{outcome}")
    # One button answers every tool call in this pause.
    count = len(pause.value["action_requests"])
    resume = Command(resume={"decisions": [{"type": decision}] * count})
    await run_and_reply(context.bot, agent, thread_id, resume)
    log.info("Chat %s resumed with %s", thread_id, decision)
