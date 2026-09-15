import json
import urllib.error
import urllib.request

from watchagent.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, require_telegram_config


def send_telegram(text: str) -> None:
    require_telegram_config()
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": text}).encode()
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            resp.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to send Telegram alert: {e}") from e


def discover_chat_ids() -> dict[int, str]:
    """Look up chat ids that have messaged this bot, via getUpdates.

    Used by `watchagent telegram-setup` so the user never has to paste their
    bot token or chat id into a chat session to configure alerts.
    """
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.load(resp)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to reach Telegram API: {e}") from e

    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data.get('description', data)}")

    chat_ids: dict[int, str] = {}
    for update in data.get("result", []):
        message = update.get("message") or update.get("channel_post")
        if not message:
            continue
        chat = message["chat"]
        label = chat.get("username") or chat.get("first_name") or chat.get("title") or ""
        chat_ids[chat["id"]] = label
    return chat_ids
