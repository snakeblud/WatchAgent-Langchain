import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

DB_PATH = Path(__file__).resolve().parent.parent / "watches.db"


def require_api_keys() -> None:
    """Validate API keys are present. Call before doing anything that needs them."""
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set in .env")
    if not TAVILY_API_KEY:
        raise RuntimeError(
            "TAVILY_API_KEY is not set in .env. Get a free key at https://tavily.com"
        )
