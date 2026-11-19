"""The worker: the Telegram bot plus a loop that runs queued jobs.

- Job loop: every 2 seconds, claim a queued GitHub job (from apps/server.py), triage the issue, and send the proposal with Approve/Reject buttons.
- The bot's handlers (/demo, button taps) live in channels/telegram.py.

Run:  uv run duo-worker
"""

import asyncio
import logging

from telegram import Bot
from telegram.ext import Application, CallbackQueryHandler, CommandHandler

from duo import storage
from duo.channels.telegram import demo, on_button, triage_and_ask
from duo.config import settings

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
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env first.")

    storage.setup()
    app = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(start_job_loop)
        .build()
    )
    app.add_handler(CommandHandler("demo", demo))
    app.add_handler(CallbackQueryHandler(on_button))
    log.info("Worker started. Waiting for jobs; /demo also works.")
    app.run_polling()  # long polling: no public URL needed


if __name__ == "__main__":
    main()
