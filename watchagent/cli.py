import argparse
import sys

from watchagent import db


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


def cmd_check(_args: argparse.Namespace) -> None:
    # Imported lazily so `add`/`list` work without API keys configured.
    from watchagent.agent import build_agent, check_watch

    rows = db.list_watches()
    if not rows:
        print("Watchlist is empty. Add a watch first with: watchagent add ...")
        return

    agent = build_agent()
    print(f"{'WATCH':30} {'CURRENT':>12} {'TARGET':>10} {'VERDICT':>11}  TREND / REASONING")
    print("-" * 110)
    for row in rows:
        verdict = check_watch(agent, row)
        print(f"{verdict.watch:30} {verdict.current_price:>8.2f} {verdict.currency:<3} "
              f"{verdict.target_price:>10.2f} {verdict.verdict:>11}  "
              f"{verdict.trend} - {verdict.reasoning}")


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
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args()
    db.init_db()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
