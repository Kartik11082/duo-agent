"""Telegram: starts triage runs, sends proposals, and resumes runs from button taps.

- triage_and_ask: run the graph for an issue and send the proposal with
  Approve/Reject buttons (apps/worker.py calls it for each queued job).
- /demo: the same flow for a hardcoded sample issue.
- Tapping a button resumes that paused run from Postgres, even after a restart.
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta

from langgraph.types import Command
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from duo.agents.triage import SAMPLE_ISSUE, build_graph
from duo.config import settings

log = logging.getLogger("duo.worker")

graph = build_graph()


def is_allowed(update: Update) -> bool:
    """Only the configured chat may use the bot."""
    chat = update.effective_chat
    return chat is not None and chat.id == settings.telegram_chat_id


def describe_actions(actions: list[dict]) -> str:
    """List actions with their arguments, so you see exactly what will run."""
    lines = []
    for action in actions:
        lines.append(f"• {action['name']}")
        for key, value in action["args"].items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def describe_results(results: list[str]) -> str:
    return "\n".join(f"• {line}" for line in results)


async def triage_and_ask(bot: Bot, thread_id: str, issue: dict, heading: str) -> None:
    """Run the graph until it pauses for approval, then send the proposal.

    If the policy marks every action "auto", the graph doesn't pause: it runs
    the actions and we just report the results.
    """
    config = {"configurable": {"thread_id": thread_id}}
    graph_input = {
        "repo": issue["repo"],  # empty for the sample issue: dry run
        "number": issue["number"],
        "title": issue["title"],
        "body": issue["body"],
    }
    # graph.invoke blocks while the LLM runs; a thread keeps the bot responsive.
    result = await asyncio.to_thread(graph.invoke, graph_input, config)

    if "__interrupt__" not in result:
        await bot.send_message(
            chat_id=settings.telegram_chat_id,
            text=f"{heading}\n\n⚡ Auto-executed:\n{describe_results(result['results'])}",
        )
        log.info("Auto-executed %s", thread_id)
        return

    proposal = result["__interrupt__"][0].value
    buttons = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ Approve", callback_data=f"yes:{thread_id}"),
                InlineKeyboardButton("❌ Reject", callback_data=f"no:{thread_id}"),
            ]
        ]
    )
    await bot.send_message(
        chat_id=settings.telegram_chat_id,
        text=f"{heading}\n\nProposed actions:\n{describe_actions(proposal['actions'])}",
        reply_markup=buttons,
    )
    log.info("Paused %s with %d action(s)", thread_id, len(proposal["actions"]))


async def demo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/demo: triage the sample issue, pause, and ask for approval."""
    if not is_allowed(update):
        return
    await update.message.reply_text("Triaging the sample issue...")
    thread_id = f"demo-{uuid.uuid4().hex[:8]}"
    heading = f"Issue (dry run): {SAMPLE_ISSUE['title']}"
    await triage_and_ask(context.bot, thread_id, SAMPLE_ISSUE, heading)


async def on_button(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
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
    age = datetime.now(UTC) - query.message.date
    if age > timedelta(hours=settings.approval_timeout_hours):
        answer = "no"
        outcome = "⌛ Expired, nothing done."
    else:
        outcome = None

    # Editing first removes the buttons, so a second tap is less likely.
    await query.edit_message_text(f"{original}\n\nWorking...")
    try:
        result = await asyncio.to_thread(graph.invoke, Command(resume=answer), config)
    except Exception as error:
        # E.g. GitHub down mid-way. Safe to retry later: both actions are idempotent.
        log.exception("Resuming %s failed", thread_id)
        await query.edit_message_text(f"{original}\n\n⚠️ Failed: {error}")
        return

    if outcome is None:
        if result["approved"]:
            outcome = f"✅ Approved:\n{describe_results(result['results'])}"
        else:
            outcome = "❌ Rejected, nothing done."
    await query.edit_message_text(f"{original}\n\n{outcome}")
    log.info("Resumed %s with %s", thread_id, answer)
