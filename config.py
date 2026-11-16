"""All settings for duo, in one place.

Values come from environment variables or a local .env file. Anything that's
missing falls back to the defaults below, so the project runs with no .env
at all.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Which LLM to use. Switching to Bedrock later is a .env change, e.g.
    # MODEL_PROVIDER=bedrock_converse and MODEL_NAME=<a Bedrock model id>.
    model_provider: str = "ollama"
    model_name: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"

    # 0 means the same input gives (nearly) the same output, which is what
    # triage and evals need.
    temperature: float = 0.0

    # Telegram approvals (worker.py). Empty defaults so the rest of the
    # project still runs without a bot; worker.py checks they are set.
    telegram_bot_token: str = ""
    telegram_chat_id: int = 0  # only this chat may approve or reject
    approval_timeout_hours: float = 24.0

    # SQLite file shared by the LangGraph checkpointer and the jobs table.
    db_path: str = "duo.db"

    # GitHub webhooks (server.py). Must match the secret set on the webhook.
    github_webhook_secret: str = ""

    # GitHub App (tools.py): the agent acts on GitHub as its own bot account.
    github_app_id: int = 0
    # Keep the .pem outside this folder: OneDrive would sync it to the cloud.
    github_private_key_path: str = ""


# Import this everywhere: `from config import settings`
settings = Settings()
