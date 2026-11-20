"""The worker: the Telegram bot plus a loop that runs queued jobs.

- Job loop: every 2 seconds, claim a queued GitHub job (from apps/server.py), triage the issue, and send the proposal with Approve/Reject buttons.
- The bot's handlers (/demo, button taps) live in channels/telegram.py.
- Chat: text messages, /new and chat button taps go to the chat agent
  (channels/telegram_chat.py).

Run:  uv run duo-worker
"""

import asyncio
import logging

from telegram import Bot
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from duo import storage
from duo.agents import chat
from duo.channels.telegram import demo, on_button, triage_and_ask
from duo.channels.telegram_chat import new_chat, on_chat_button, on_message
from duo.config import agent, get_secret, settings
from duo.llm import enable_tracing

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
# httpx logs every request URL at INFO, and Telegram URLs contain the bot token.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("duo.worker")


async def process_jobs(bot: Bot) -> None:
    """Background loop: run queued jobs one at a time, forever."""
    while True:
        job = storage.claim_next_job()
        if job is None:
            await asyncio.sleep(2)
            continue

        try:
            issue = job["payload"]
            heading = f"{job['thread_id']}: {issue['title']}\n{issue['url']}"
            await triage_and_ask(bot, job["thread_id"], issue, heading)
            storage.finish_job(job["id"], "done")  # done = proposal sent
        except Exception as error:
            # Catch everything so one bad job can't stop the loop.
            log.exception("Job %s failed", job["id"])
            storage.finish_job(job["id"], "failed", str(error))


async def start_job_loop(app: Application) -> None:
    """Runs once when the bot starts (post_init)."""
    reset = storage.reset_running_jobs()
    if reset:
        log.info("Reset %d job(s) left running by a previous crash", reset)
    # Keep a reference, or Python may garbage-collect the running task.
    app.bot_data["job_loop"] = asyncio.create_task(process_jobs(app.bot))


def main() -> None:
    token = get_secret("telegram_bot_token")
    if not token or not settings.telegram_chat_id:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env first.")

    if enable_tracing():
        log.info("LangSmith tracing on, project %s", agent.tracing.project)
    storage.setup()
    app = Application.builder().token(token).post_init(start_job_loop).build()
    app.bot_data["chat_agent"] = chat.build_agent()
    app.add_handler(CommandHandler("demo", demo))
    app.add_handler(CommandHandler("new", new_chat))
    # Each tap goes to the handler whose pattern matches its button data.
    app.add_handler(CallbackQueryHandler(on_button, pattern=r"^(yes|no):"))
    app.add_handler(CallbackQueryHandler(on_chat_button, pattern=r"^chat:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    log.info("Worker started. Waiting for jobs and messages; /demo and /new work.")
    app.run_polling()  # long polling: no public URL needed


if __name__ == "__main__":
    main()
