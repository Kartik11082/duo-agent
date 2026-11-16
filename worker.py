"""Telegram bot that starts graph runs and resumes them from button taps.

Milestone 4: /demo triages the sample issue and sends Approve/Reject buttons.
Tapping a button resumes that paused run from duo.db, even after a restart.

Run:  uv run python worker.py
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone

from langgraph.types import Command
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from config import settings
from graph import SAMPLE_ISSUE, build_graph

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
# httpx logs every request URL at INFO, and Telegram URLs contain the bot token.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("duo.worker")

graph = build_graph()


def is_allowed(update: Update) -> bool:
    """Only the configured chat may use the bot."""
    chat = update.effective_chat
    return chat is not None and chat.id == settings.telegram_chat_id


async def demo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/demo: triage the sample issue, pause, and ask for approval."""
    if not is_allowed(update):
        return
    thread_id = f"demo-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    await update.message.reply_text("Triaging the sample issue...")
    # graph.invoke blocks while the LLM runs; a thread keeps the bot responsive.
    result = await asyncio.to_thread(graph.invoke, SAMPLE_ISSUE, config)
    proposal = result["__interrupt__"][0].value

    buttons = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ Approve", callback_data=f"yes:{thread_id}"),
                InlineKeyboardButton("❌ Reject", callback_data=f"no:{thread_id}"),
            ]
        ]
    )
    await update.message.reply_text(
        f"Issue: {SAMPLE_ISSUE['title']}\n\n"
        f"Proposed label: {proposal['label']}\n"
        f"Reason: {proposal['reasoning']}",
        reply_markup=buttons,
    )
    log.info("Paused %s, proposed label %s", thread_id, proposal["label"])


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Approve/Reject tap: resume the paused run named in the button."""
    query = update.callback_query
    if not is_allowed(update):
        await query.answer("Not allowed.")
        return
    await query.answer()  # stops the loading spinner on the button

    answer, thread_id = query.data.split(":", 1)
    config = {"configurable": {"thread_id": thread_id}}
    original = query.message.text

    # Nothing waiting to run means this run already finished (e.g. a double tap).
    if not graph.get_state(config).next:
        await query.edit_message_text(f"{original}\n\nAlready handled.")
        return

    # Lazy expiry: checked only when someone taps. An expired run is closed as "no".
    age = datetime.now(timezone.utc) - query.message.date
    if age > timedelta(hours=settings.approval_timeout_hours):
        answer = "no"
        outcome = "⌛ Expired, nothing done."
    else:
        outcome = None

    # Editing first removes the buttons, so a second tap is less likely.
    await query.edit_message_text(f"{original}\n\nWorking...")
    result = await asyncio.to_thread(graph.invoke, Command(resume=answer), config)

    if outcome is None:
        if result["approved"]:
            outcome = f"✅ Approved: label '{result['label']}' applied (simulated)."
        else:
            outcome = "❌ Rejected, nothing done."
    await query.edit_message_text(f"{original}\n\n{outcome}")
    log.info("Resumed %s with %s", thread_id, answer)


def main() -> None:
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env first.")

    app = Application.builder().token(settings.telegram_bot_token).build()
    app.add_handler(CommandHandler("demo", demo))
    app.add_handler(CallbackQueryHandler(on_button))
    log.info("Worker started. Send /demo to your bot.")
    app.run_polling()  # long polling: no public URL needed


if __name__ == "__main__":
    main()
