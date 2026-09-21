import json
import urllib.error
import urllib.request

from watchagent.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, require_telegram_config

MAX_MESSAGE_CHARS = 4000  # Telegram's hard limit is 4096


def _api(method: str, payload: dict | None = None) -> dict:
    """Call a Telegram Bot API method and return its `result`."""
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload or {}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as resp:
            data = json.load(resp)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to reach Telegram API ({method}): {e}") from e
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error ({method}): {data.get('description', data)}")
    return data["result"]


def send_telegram(text: str, buttons: list[tuple[str, str]] | None = None) -> None:
    """Send `text` to the configured chat, optionally with one row of inline buttons.

    `buttons` is a list of (label, callback_data) pairs.
    """
    require_telegram_config()
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text[:MAX_MESSAGE_CHARS]}
    if buttons:
        payload["reply_markup"] = {
            "inline_keyboard": [[{"text": label, "callback_data": data} for label, data in buttons]]
        }
    try:
        _api("sendMessage", payload)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to send Telegram alert: {e}") from e


def get_updates(offset: int = 0) -> list[dict]:
    """Fetch pending updates (messages and button taps) starting at `offset`."""
    return _api("getUpdates", {
        "offset": offset,
        "timeout": 0,
        "allowed_updates": ["message", "callback_query"],
    })


def answer_callback(callback_query_id: str, text: str = "") -> None:
    """Dismiss the spinner on a tapped inline button."""
    _api("answerCallbackQuery", {"callback_query_id": callback_query_id, "text": text})


def discover_chat_ids() -> dict[int, str]:
    """Look up chat ids that have messaged this bot, via getUpdates.

    Used by `watchagent telegram-setup` so the user never has to paste their
    bot token or chat id into a chat session to configure alerts.
    """
    chat_ids: dict[int, str] = {}
    for update in _api("getUpdates"):
        message = update.get("message") or update.get("channel_post")
        if not message:
            continue
        chat = message["chat"]
        label = chat.get("username") or chat.get("first_name") or chat.get("title") or ""
        chat_ids[chat["id"]] = label
    return chat_ids
