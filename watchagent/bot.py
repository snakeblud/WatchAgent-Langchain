"""Telegram front end: read pending messages once, act on them, reply.

Runs as a short job (`watchagent bot`) rather than a long-lived listener, so it
fits on a scheduled GitHub Actions workflow. Only the chat in
TELEGRAM_CHAT_ID is ever served.
"""
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Callable

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from watchagent import db
from watchagent.config import LLM_MODEL, TELEGRAM_CHAT_ID, require_api_keys, require_telegram_config
from watchagent.verdict import describe_trend

OFFSET_KEY = "telegram_offset"
PENDING_TTL = timedelta(hours=24)
HISTORY_LIMIT = 10
# Past Telegram messages (user + assistant) the assistant sees.
CHAT_MEMORY = 12

HELP = """WatchAgent commands:
/list - watches with their latest verdict
/status <id> - latest price and verdict for a watch
/history <id> - recent checks
/check <id> - check the price now (about a minute)
/add brand | model | ref | target [| currency]
/target <id> <price> - change a target price
/remove <id> - stop tracking a watch
Or just write, e.g. "add the Black Bay 58 at 3.5k and check its price".
Adding, retargeting and removing ask you to confirm first."""

ASSISTANT_PROMPT = """You are the assistant for a personal luxury-watch price tracker,
chatting on Telegram. Use your tools; don't guess numbers.
- To answer questions, read the watchlist and check history, and quote the numbers.
- For a fresh price, use research_price (it takes about a minute).
- To add, retarget or remove a watch, use the propose_* tools. They only send the
  user a Confirm button; nothing changes until the user taps it, so say that.
Keep replies to a few sentences."""


@dataclass
class Services:
    """What the bot needs from the outside world, so tests can swap it out."""
    send: Callable[..., None]  # send(text, buttons=None)
    answer_callback: Callable[[str, str], None]
    check: Callable[..., object]  # check(watch_row) -> CheckResult
    answer: Callable[[str], str]  # free-text question -> reply


# --- formatting and parsing ---------------------------------------------------

def _label(row) -> str:
    return f"{row['brand']} {row['model']} ({row['reference_no'] or 'no ref'})"


def _check_line(check) -> str:
    if check is None:
        return "not checked yet"
    return f"{check['verdict']} at {check['price']:,.0f} {check['currency']} ({check['pct_vs_target']:+.1f}% vs target)"


def _first_id(args: str) -> int | None:
    """The watch id a command's arguments start with, e.g. 3 in "3 13500"."""
    words = args.split()
    return int(words[0]) if words and words[0].isdigit() else None


def _parse_price(text: str) -> float | None:
    try:
        value = float(text.strip().replace(",", ""))
    except ValueError:
        return None
    return value if value > 0 else None


def _find_watch(args: str, services: Services):
    watch_id = _first_id(args)
    if watch_id is None:
        services.send("Give a watch id, e.g. /status 1 (see /list).")
        return None
    row = db.get_watch(watch_id)
    if row is None:
        services.send(f"No watch with id {watch_id}. See /list.")
    return row


# --- read-only commands -------------------------------------------------------

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


# --- changes: staged as a pending action, run only after a Confirm tap --------
# The slash commands and the assistant's propose_* tools both go through these.

def _confirm(services: Services, action: str, payload: dict, question: str) -> None:
    pending_id = db.create_pending(action, payload)
    services.send(question, [("Confirm", f"ok:{pending_id}"), ("Cancel", f"no:{pending_id}")])


def propose_add(brand: str, model: str, ref: str, target: float, currency: str,
                services: Services) -> str | None:
    """Ask the user to confirm adding a watch. Returns why not, or None once the button is sent."""
    if not brand.strip() or not model.strip():
        return "Brand and model are required."
    if target <= 0:
        return "The target price must be positive."
    wanted = (brand.lower(), model.lower(), ref.lower())
    for existing in db.list_watches():
        if (existing["brand"].lower(), existing["model"].lower(), (existing["reference_no"] or "").lower()) == wanted:
            return f"Already tracking {_label(existing)} as [{existing['id']}]."
    _confirm(
        services, "add",
        {"brand": brand, "model": model, "ref": ref, "target": target, "currency": currency},
        f"Add {brand} {model} ({ref or 'no ref'}) with target {target:,.0f} {currency}?",
    )
    return None


def propose_target(watch_id: int, price: float, services: Services) -> str | None:
    """Ask the user to confirm a new target. Returns why not, or None once the button is sent."""
    row = db.get_watch(watch_id)
    if row is None:
        return f"No watch with id {watch_id}. See /list."
    if price <= 0:
        return "The target price must be positive."
    _confirm(
        services, "target", {"id": watch_id, "price": price},
        f"Change the target for {_label(row)} from {row['target_price']:,.0f} to {price:,.0f} {row['currency']}?",
    )
    return None


def propose_remove(watch_id: int, services: Services) -> str | None:
    """Ask the user to confirm removing a watch. Returns why not, or None once the button is sent."""
    row = db.get_watch(watch_id)
    if row is None:
        return f"No watch with id {watch_id}. See /list."
    _confirm(
        services, "remove", {"id": row["id"]},
        f"Stop tracking {_label(row)}? This also deletes its price history, checks and notes.",
    )
    return None


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
    if error := propose_add(brand, model, ref, target, currency, services):
        services.send(error)


def cmd_target(args: str, services: Services) -> None:
    words = args.split()
    watch_id = _first_id(args)
    price = _parse_price(words[1]) if len(words) == 2 else None
    if watch_id is None or price is None:
        services.send("Usage: /target <id> <price>, e.g. /target 1 13500")
        return
    if error := propose_target(watch_id, price, services):
        services.send(error)


def cmd_remove(args: str, services: Services) -> None:
    watch_id = _first_id(args)
    if watch_id is None:
        services.send("Usage: /remove <id>, e.g. /remove 1 (see /list)")
        return
    if error := propose_remove(watch_id, services):
        services.send(error)


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


# --- confirm / cancel button taps ---------------------------------------------

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


# --- free-text messages: the conversational assistant ------------------------

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


_PROPOSED = "Sent the user a Confirm button. Nothing changes until they tap it."


@tool
def propose_add_watch(brand: str, model: str, reference: str, target_price: float,
                      runtime: ToolRuntime[Services], currency: str = "USD") -> str:
    """Ask the user to confirm adding a watch to the watchlist.

    Args:
        brand: e.g. "Tudor"
        model: e.g. "Black Bay 58"
        reference: Reference number, e.g. "M79030N-0001"; "" if unknown
        target_price: The price the user would buy at
        currency: ISO code, default USD
    """
    return propose_add(brand, model, reference, target_price, currency.upper(), runtime.context) \
        or _PROPOSED


@tool
def propose_new_target(watch_id: int, target_price: float, runtime: ToolRuntime[Services]) -> str:
    """Ask the user to confirm changing a watch's target price.

    Args:
        watch_id: The watch id from get_watchlist_status
        target_price: The new target price
    """
    return propose_target(watch_id, target_price, runtime.context) or _PROPOSED


@tool
def propose_removal(watch_id: int, runtime: ToolRuntime[Services]) -> str:
    """Ask the user to confirm removing a watch from the watchlist.

    Args:
        watch_id: The watch id from get_watchlist_status
    """
    return propose_remove(watch_id, runtime.context) or _PROPOSED


@tool
def research_price(watch_id: int, runtime: ToolRuntime[Services]) -> str:
    """Run a fresh price check on a watch (about a minute) and return the result with its verdict.

    Args:
        watch_id: The watch id from get_watchlist_status
    """
    row = db.get_watch(watch_id)
    if row is None:
        return f"No watch with id {watch_id}."
    try:
        return runtime.context.check(row).message()
    except Exception as e:
        return f"The check failed: {e}"


ASSISTANT_TOOLS = [get_watchlist_status, get_check_history, research_price,
                   propose_add_watch, propose_new_target, propose_removal]


def run_assistant(text: str, services: Services) -> str:
    """Answer a free-text message. The assistant can read and research, but only *propose* changes."""
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware, ModelRetryMiddleware

    from watchagent.agent import make_model, request_budget, retry_on
    from watchagent.config import DAILY_REQUEST_LIMIT

    require_api_keys()
    agent = create_agent(
        model=make_model(LLM_MODEL),
        tools=ASSISTANT_TOOLS,
        system_prompt=ASSISTANT_PROMPT,
        context_schema=Services,
        middleware=[ModelRetryMiddleware(max_retries=3, initial_delay=5, on_failure="error", retry_on=retry_on),
                    ModelCallLimitMiddleware(run_limit=10, exit_behavior="end"),
                    request_budget(DAILY_REQUEST_LIMIT)],
    )
    history = [{"role": role, "content": content} for role, content in db.recent_chat(CHAT_MEMORY)]
    result = agent.invoke({"messages": history + [{"role": "user", "content": text}]},
                          config={"recursion_limit": 30}, context=services)
    reply = result["messages"][-1].text or "I couldn't work out an answer."
    db.add_chat("user", text)
    db.add_chat("assistant", reply)
    return reply


# --- entry point --------------------------------------------------------------

@lru_cache(maxsize=1)
def _research_agent():
    """Built on the first /check only, then reused."""
    from watchagent.agent import build_agent
    return build_agent()


def default_services() -> Services:
    from watchagent import notify

    from watchagent.agent import check_watch

    services = Services(send=notify.send_telegram, answer_callback=notify.answer_callback,
                        check=lambda row: check_watch(_research_agent(), row), answer=lambda text: "")
    # The assistant needs the finished Services (to send Confirm buttons), so it is wired in last.
    services.answer = lambda text: run_assistant(text, services)
    return services


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
