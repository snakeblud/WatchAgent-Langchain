"""Telegram front end: read pending messages once, act on them, reply.

Runs as a short job (`watchagent bot`) rather than a long-lived listener, so it
fits on a scheduled GitHub Actions workflow. Only the chat in
TELEGRAM_CHAT_ID is ever served.
"""
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from langchain_core.tools import tool

from watchagent import db
from watchagent.config import GEMINI_MODEL, TELEGRAM_CHAT_ID, require_telegram_config
from watchagent.verdict import describe_trend

OFFSET_KEY = "telegram_offset"
PENDING_TTL = timedelta(hours=24)
HISTORY_LIMIT = 10

HELP = """WatchAgent commands:
/list - watches with their latest verdict
/status <id> - latest price and verdict for a watch
/history <id> - recent checks
/check <id> - check the price now (about a minute)
/add brand | model | ref | target [| currency]
/target <id> <price> - change a target price
/remove <id> - stop tracking a watch
Or just ask a question, e.g. "is the Aquanaut worth it yet?"
Adding, retargeting and removing ask you to confirm first."""

QA_PROMPT = """You are the assistant for a personal luxury-watch price tracker.
Answer using ONLY what the tools return, quoting the numbers. You cannot change
the watchlist: if asked to add, remove or retarget a watch, tell the user to use
/add, /target or /remove. Keep answers to a few sentences. If there is no data
yet, say so."""


@dataclass
class Services:
    """What the bot needs from the outside world, so tests can swap it out."""
    send: Callable[..., None]  # send(text, buttons=None)
    answer_callback: Callable[[str, str], None]
    check: Callable[..., object]  # check(watch_row) -> CheckResult
    answer: Callable[[str], str]  # free-text question -> reply


# --- formatting ---------------------------------------------------------------

def _label(row) -> str:
    return f"{row['brand']} {row['model']} ({row['reference_no'] or 'no ref'})"


def _check_line(check) -> str:
    if check is None:
        return "not checked yet"
    return f"{check['verdict']} at {check['price']:,.0f} {check['currency']} ({check['pct_vs_target']:+.1f}% vs target)"


def _parse_id(text: str) -> int | None:
    text = text.strip()
    return int(text) if text.isdigit() else None


def _parse_price(text: str) -> float | None:
    try:
        value = float(text.strip().replace(",", ""))
    except ValueError:
        return None
    return value if value > 0 else None


def _find_watch(args: str, services: Services):
    watch_id = _parse_id(args.split()[0]) if args.split() else None
    if watch_id is None:
        services.send("Give a watch id, e.g. /status 1 (see /list).")
        return None
    row = db.get_watch(watch_id)
    if row is None:
        services.send(f"No watch with id {watch_id}. See /list.")
    return row


def _confirm(services: Services, action: str, payload: dict, question: str) -> None:
    pending_id = db.create_pending(action, payload)
    services.send(question, [("Confirm", f"ok:{pending_id}"), ("Cancel", f"no:{pending_id}")])


# --- commands -----------------------------------------------------------------

def cmd_help(_args: str, services: Services) -> None:
    services.send(HELP)


def cmd_list(_args: str, services: Services) -> None:
    rows = db.list_watches()
    if not rows:
        services.send("Watchlist is empty. Add one with /add brand | model | ref | target")
        return
    services.send("\n".join(
        f"[{r['id']}] {_label(r)}\n    target {r['target_price']:,.0f} {r['currency']} - "
        f"{_check_line(db.latest_check(r['id']))}"
        for r in rows
    ))


def cmd_status(args: str, services: Services) -> None:
    row = _find_watch(args, services)
    if row is None:
        return
    check = db.latest_check(row["id"])
    if check is None:
        services.send(f"[{row['id']}] {_label(row)}\nNot checked yet. Try /check {row['id']}.")
        return
    services.send(
        f"[{row['id']}] {_label(row)}\n"
        f"{check['verdict']}: {check['price']:,.2f} {check['currency']} "
        f"(target {row['target_price']:,.2f}, {check['pct_vs_target']:+.1f}% vs target)\n"
        f"Trend: {describe_trend(check['trend_pct'])}\n"
        f"{check['confidence']} ({check['sources']}) at {check['checked_at'][:16].replace('T', ' ')} UTC\n"
        f"{check['reasoning']}"
    )


def cmd_history(args: str, services: Services) -> None:
    row = _find_watch(args, services)
    if row is None:
        return
    checks = db.recent_checks(row["id"], HISTORY_LIMIT)
    if not checks:
        services.send(f"[{row['id']}] {_label(row)}\nNo checks recorded yet.")
        return
    lines = [
        f"{c['checked_at'][:16].replace('T', ' ')}  {c['price']:>10,.0f} {c['currency']}  "
        f"{c['verdict']} ({c['pct_vs_target']:+.1f}%)"
        for c in checks
    ]
    services.send(f"[{row['id']}] {_label(row)}\n" + "\n".join(lines))


def cmd_check(args: str, services: Services) -> None:
    row = _find_watch(args, services)
    if row is None:
        return
    services.send(f"Checking {_label(row)}... this takes about a minute.")
    try:
        result = services.check(row)
    except Exception as e:
        services.send(f"Check failed: {e}")
        return
    services.send(result.message())


def cmd_add(args: str, services: Services) -> None:
    parts = [p.strip() for p in args.split("|")]
    if len(parts) not in (4, 5) or not parts[0] or not parts[1]:
        services.send("Usage: /add brand | model | ref | target [| currency]\n"
                      "e.g. /add Rolex | Submariner | 126610LN | 14000")
        return
    brand, model, ref, target_text = parts[:4]
    currency = (parts[4] if len(parts) == 5 and parts[4] else "USD").upper()
    target = _parse_price(target_text)
    if target is None:
        services.send(f"'{target_text}' isn't a valid target price.")
        return
    for existing in db.list_watches():
        if (existing["brand"].lower(), existing["model"].lower(), (existing["reference_no"] or "").lower()) \
                == (brand.lower(), model.lower(), ref.lower()):
            services.send(f"Already tracking {_label(existing)} as [{existing['id']}].")
            return
    _confirm(
        services, "add",
        {"brand": brand, "model": model, "ref": ref, "target": target, "currency": currency},
        f"Add {brand} {model} ({ref or 'no ref'}) with target {target:,.0f} {currency}?",
    )


def cmd_target(args: str, services: Services) -> None:
    words = args.split()
    watch_id = _parse_id(words[0]) if words else None
    price = _parse_price(words[1]) if len(words) == 2 else None
    if watch_id is None or price is None:
        services.send("Usage: /target <id> <price>, e.g. /target 1 13500")
        return
    row = db.get_watch(watch_id)
    if row is None:
        services.send(f"No watch with id {watch_id}. See /list.")
        return
    _confirm(
        services, "target", {"id": watch_id, "price": price},
        f"Change the target for {_label(row)} from {row['target_price']:,.0f} to {price:,.0f} {row['currency']}?",
    )


def cmd_remove(args: str, services: Services) -> None:
    row = _find_watch(args, services)
    if row is None:
        return
    _confirm(
        services, "remove", {"id": row["id"]},
        f"Stop tracking {_label(row)}? This also deletes its price history and checks.",
    )


COMMANDS: dict[str, Callable[[str, Services], None]] = {
    "/start": cmd_help,
    "/help": cmd_help,
    "/list": cmd_list,
    "/status": cmd_status,
    "/history": cmd_history,
    "/check": cmd_check,
    "/add": cmd_add,
    "/target": cmd_target,
    "/remove": cmd_remove,
}


# --- confirmations ------------------------------------------------------------

def _execute(action: str, payload: dict) -> str:
    if action == "add":
        watch_id = db.add_watch(payload["brand"], payload["model"], payload["target"],
                                payload["ref"], payload["currency"])
        return f"Added [{watch_id}] {payload['brand']} {payload['model']}. Try /check {watch_id}."
    if action == "target":
        if not db.update_target(payload["id"], payload["price"]):
            raise ValueError(f"no watch with id {payload['id']}")
        return f"Target for [{payload['id']}] is now {payload['price']:,.0f}."
    if action == "remove":
        if not db.remove_watch(payload["id"]):
            raise ValueError(f"no watch with id {payload['id']}")
        return f"Removed [{payload['id']}]."
    raise ValueError(f"unknown action {action}")


def handle_callback(callback: dict, services: Services, now: datetime | None = None) -> None:
    verb, _, raw_id = callback.get("data", "").partition(":")
    if verb not in ("ok", "no") or not raw_id.isdigit():
        services.answer_callback(callback["id"], "Unknown action")
        return
    resolved = db.resolve_pending(int(raw_id), "done" if verb == "ok" else "cancelled")
    if resolved is None:
        services.answer_callback(callback["id"], "Already handled or expired")
        return
    if verb == "no":
        services.answer_callback(callback["id"], "Cancelled")
        services.send("Cancelled.")
        return
    action, payload, created_at = resolved
    if (now or datetime.now(timezone.utc)) - datetime.fromisoformat(created_at) > PENDING_TTL:
        services.answer_callback(callback["id"], "Expired")
        services.send("That request expired. Send the command again.")
        return
    try:
        message = _execute(action, payload)
    except ValueError as e:
        message = f"Not done: {e}"
    services.answer_callback(callback["id"], "Done")
    services.send(message)


# --- dispatch -----------------------------------------------------------------

def handle_text(text: str, services: Services) -> None:
    if not text.startswith("/"):
        services.send(services.answer(text))
        return
    command, _, rest = text.partition(" ")
    handler = COMMANDS.get(command.split("@")[0].lower())
    if handler is None:
        services.send("Unknown command.\n" + HELP)
        return
    handler(rest.strip(), services)


def handle_update(update: dict, services: Services, allowed_chat_id: int) -> None:
    callback = update.get("callback_query")
    message = callback.get("message") if callback else update.get("message")
    chat_id = ((message or {}).get("chat") or {}).get("id")
    if chat_id != allowed_chat_id:
        return  # anyone else who finds the bot gets no reply and no processing
    if callback:
        handle_callback(callback, services)
    elif message.get("text"):
        handle_text(message["text"].strip(), services)


# --- free-text questions ------------------------------------------------------

@tool
def get_watchlist_status() -> str:
    """List every tracked watch with its target price and latest check result."""
    rows = db.list_watches()
    if not rows:
        return "The watchlist is empty."
    return "\n".join(
        f"id={r['id']} {_label(r)} target={r['target_price']:,.0f} {r['currency']} - "
        f"{_check_line(db.latest_check(r['id']))}"
        for r in rows
    )


@tool
def get_check_history(watch_id: int) -> str:
    """Recent checks (date, price, verdict) for one watch, newest first.

    Args:
        watch_id: The watch id from get_watchlist_status
    """
    checks = db.recent_checks(watch_id, HISTORY_LIMIT)
    if not checks:
        return "No checks recorded for that watch."
    return "\n".join(
        f"{c['checked_at'][:16]}: {c['price']:,.0f} {c['currency']} {c['verdict']} "
        f"({c['pct_vs_target']:+.1f}% vs target; {describe_trend(c['trend_pct'])}; {c['confidence']})"
        for c in checks
    )


def answer_question(question: str) -> str:
    """One-shot, read-only Q&A over the stored checks. It has no tool that can write."""
    from langchain.agents import create_agent
    from langchain_google_genai import ChatGoogleGenerativeAI

    from watchagent.config import require_api_keys

    require_api_keys()
    agent = create_agent(
        model=ChatGoogleGenerativeAI(model=GEMINI_MODEL),
        tools=[get_watchlist_status, get_check_history],
        system_prompt=QA_PROMPT,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]},
                          config={"recursion_limit": 12})
    return result["messages"][-1].text or "I couldn't work out an answer."


# --- entry point --------------------------------------------------------------

def default_services() -> Services:
    from watchagent import notify

    agent_cache: list = []

    def check(row):
        from watchagent.agent import build_agent, check_watch
        if not agent_cache:
            agent_cache.append(build_agent())
        return check_watch(agent_cache[0], row)

    return Services(send=notify.send_telegram, answer_callback=notify.answer_callback,
                    check=check, answer=answer_question)


def run_once(services: Services | None = None) -> int:
    """Process every pending Telegram update once. Returns how many there were."""
    from watchagent import notify

    require_telegram_config()
    services = services or default_services()
    allowed_chat_id = int(TELEGRAM_CHAT_ID)
    offset = int(db.get_state(OFFSET_KEY) or 0)
    updates = notify.get_updates(offset)
    for update in updates:
        try:
            handle_update(update, services, allowed_chat_id)
        except Exception as e:
            print(f"update {update.get('update_id')} failed: {e}", file=sys.stderr)
            try:
                services.send(f"Sorry, that failed: {e}")
            except Exception:
                pass
        # Advance after every update, even a failed one, so a bad message can't
        # be replayed forever.
        db.set_state(OFFSET_KEY, str(update["update_id"] + 1))
    return len(updates)
