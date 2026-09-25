import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# Model names starting "gemini" go to Google (AI Studio free tier); anything else goes to Groq (free tier).
# `or`, not a get() default: CI passes an unset repo variable as an empty string.
LLM_MODEL = os.environ.get("LLM_MODEL") or "gemini-3.5-flash-lite"
# Tried in order when LLM_MODEL is overloaded or failing (comma-separated). Gemini ones run
# with thinking off, which "-lite" models don't accept, so don't list those here.
# A model is skipped when its provider's API key isn't set.
LLM_FALLBACK_MODELS = os.environ.get("LLM_FALLBACK_MODELS", "gemini-3.5-flash,openai/gpt-oss-120b").split(",")
# Characters of each search snippet shown to the model (0 = all). Groq's free tier rejects
# requests over ~8k tokens, which untrimmed search results reach within a few searches.
SEARCH_SNIPPET_CHARS = int(os.environ.get("SEARCH_SNIPPET_CHARS",
                                          "0" if LLM_MODEL.startswith("gemini") else "350"))

# Model requests per UTC day. Scheduled checks stop at their share, so the rest
# stays free for the Telegram bot's /check and questions. The defaults sit above
# the ~240/day the old 4x-daily schedule used on the free tier without hitting a
# quota; lower them if Google's free-tier limits for your models are tighter.
DAILY_REQUEST_LIMIT = int(os.environ.get("DAILY_REQUEST_LIMIT") or "300")  # `or`: CI passes unset as ""
SCHEDULED_REQUEST_SHARE = int(os.environ.get("SCHEDULED_REQUEST_SHARE", "250"))

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

DB_PATH = Path(os.environ.get("WATCHAGENT_DB")
               or Path(__file__).resolve().parent.parent / "watches.db")


def require_api_keys() -> None:
    """Validate API keys are present. Call before doing anything that needs them."""
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set in .env")
    if not TAVILY_API_KEY:
        raise RuntimeError(
            "TAVILY_API_KEY is not set in .env. Get a free key at https://tavily.com"
        )


def require_telegram_config() -> None:
    """Validate Telegram alert config is present. Call before sending alerts."""
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")
    if not TELEGRAM_CHAT_ID:
        raise RuntimeError(
            "TELEGRAM_CHAT_ID is not set in .env. Run `watchagent telegram-setup` "
            "after messaging your bot once to find it."
        )
