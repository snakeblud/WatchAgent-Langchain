import argparse
import sys
import time
from datetime import datetime, timedelta, timezone

from watchagent import db

DEFAULT_ALERT_COOLDOWN_HOURS = 12.0


def cmd_add(args: argparse.Namespace) -> None:
    watch_id = db.add_watch(
        brand=args.brand,
        model=args.model,
        target_price=args.target_price,
        reference_no=args.ref or "",
        currency=args.currency,
        notes=args.notes or "",
    )
    print(f"Added watch id={watch_id}: {args.brand} {args.model} (target {args.target_price} {args.currency})")


def cmd_list(_args: argparse.Namespace) -> None:
    rows = db.list_watches()
    if not rows:
        print("Watchlist is empty. Add one with: watchagent add ...")
        return
    for r in rows:
        print(f"[{r['id']}] {r['brand']} {r['model']} (ref {r['reference_no']}) "
              f"- target {r['target_price']} {r['currency']}")


def _maybe_notify(row, verdict, cooldown_hours: float) -> None:
    from watchagent.notify import send_telegram

    if verdict.verdict != "BUY":
        return
    last = row["last_alerted_at"]
    if last:
        last_dt = datetime.fromisoformat(last)
        if datetime.now(timezone.utc) - last_dt < timedelta(hours=cooldown_hours):
            return
    send_telegram(
        f"BUY: {verdict.watch}\n"
        f"Current: {verdict.current_price:.2f} {verdict.currency} "
        f"(target {verdict.target_price:.2f})\n"
        f"Trend: {verdict.trend}\n"
        f"{verdict.reasoning}"
    )
    db.mark_alerted(row["id"])


def _run_checks(agent, rows, notify: bool, cooldown_hours: float) -> int:
    """Runs a check round. Returns the number of watches that failed to check."""
    from watchagent.agent import check_watch

    failures = 0
    print(f"{'WATCH':30} {'CURRENT':>12} {'TARGET':>10} {'VERDICT':>11}  TREND / REASONING")
    print("-" * 110)
    for row in rows:
        try:
            verdict = check_watch(agent, row)
        except Exception as e:
            print(f"{row['brand']} {row['model']}: check failed: {e}", file=sys.stderr)
            failures += 1
            continue
        print(f"{verdict.watch:30} {verdict.current_price:>8.2f} {verdict.currency:<3} "
              f"{verdict.target_price:>10.2f} {verdict.verdict:>11}  "
              f"{verdict.trend} - {verdict.reasoning}")
        if notify:
            try:
                _maybe_notify(row, verdict, cooldown_hours)
            except Exception as e:
                print(f"  (alert failed: {e})", file=sys.stderr)
    return failures


def cmd_check(args: argparse.Namespace) -> None:
    # Imported lazily so `add`/`list` work without API keys configured.
    from watchagent.agent import build_agent

    rows = db.list_watches()
    if not rows:
        print("Watchlist is empty. Add a watch first with: watchagent add ...")
        return

    agent = build_agent()
    failures = _run_checks(agent, rows, notify=args.notify, cooldown_hours=args.alert_cooldown_hours)
    if failures:
        sys.exit(1)


def cmd_schedule(args: argparse.Namespace) -> None:
    from watchagent.agent import build_agent

    agent = build_agent()
    interval_seconds = args.interval_minutes * 60
    print(f"Checking every {args.interval_minutes} minute(s)"
          f"{' with Telegram alerts on BUY' if args.notify else ''}. Ctrl+C to stop.")
    try:
        while True:
            rows = db.list_watches()
            if not rows:
                print("Watchlist is empty. Add a watch first with: watchagent add ...")
            else:
                _run_checks(agent, rows, notify=args.notify,
                            cooldown_hours=args.alert_cooldown_hours)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nStopped.")


def cmd_telegram_setup(_args: argparse.Namespace) -> None:
    from watchagent.notify import discover_chat_ids

    chat_ids = discover_chat_ids()
    if not chat_ids:
        print("No chats found yet. Send any message to your bot on Telegram, "
              "then run `watchagent telegram-setup` again.")
        return
    print("Found chat id(s):")
    for chat_id, label in chat_ids.items():
        print(f"  {chat_id}  {label}")
    print("Add TELEGRAM_CHAT_ID=<id> to your .env (pick yours if there are several).")


def cmd_telegram_test(_args: argparse.Namespace) -> None:
    from watchagent.notify import send_telegram

    send_telegram("WatchAgent alerts are working.")
    print("Sent a test message to your configured Telegram chat.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="watchagent", description="Luxury watch price tracker & buy advisor")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add a watch to the watchlist")
    p_add.add_argument("--brand", required=True)
    p_add.add_argument("--model", required=True)
    p_add.add_argument("--ref", help="Reference number")
    p_add.add_argument("--target-price", type=float, required=True)
    p_add.add_argument("--currency", default="USD")
    p_add.add_argument("--notes")
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="List tracked watches")
    p_list.set_defaults(func=cmd_list)

    p_check = sub.add_parser("check", help="Check current prices and get buy advice")
    p_check.add_argument("--notify", action="store_true",
                          help="Send a Telegram alert for any BUY verdict")
    p_check.add_argument("--alert-cooldown-hours", type=float,
                          default=DEFAULT_ALERT_COOLDOWN_HOURS,
                          help="Don't re-alert the same watch within this many hours "
                               f"(default {DEFAULT_ALERT_COOLDOWN_HOURS})")
    p_check.set_defaults(func=cmd_check)

    p_schedule = sub.add_parser("schedule", help="Run checks repeatedly on an interval")
    p_schedule.add_argument("--interval-minutes", type=float, default=60.0,
                             help="Minutes between check rounds (default 60)")
    p_schedule.add_argument("--notify", action="store_true",
                             help="Send a Telegram alert for any BUY verdict")
    p_schedule.add_argument("--alert-cooldown-hours", type=float,
                             default=DEFAULT_ALERT_COOLDOWN_HOURS,
                             help="Don't re-alert the same watch within this many hours "
                                  f"(default {DEFAULT_ALERT_COOLDOWN_HOURS})")
    p_schedule.set_defaults(func=cmd_schedule)

    p_tg_setup = sub.add_parser("telegram-setup",
                                 help="Find your Telegram chat id (message your bot first)")
    p_tg_setup.set_defaults(func=cmd_telegram_setup)

    p_tg_test = sub.add_parser("telegram-test", help="Send a test Telegram alert")
    p_tg_test.set_defaults(func=cmd_telegram_test)

    args = parser.parse_args()
    db.init_db()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
