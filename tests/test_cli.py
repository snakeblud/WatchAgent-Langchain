from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from watchagent.cli import ALERT_IMPROVEMENT_PCT, build_parser, should_alert

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)


def result(verdict="BUY", price=10_000.0, listings=3):
    return SimpleNamespace(verdict=SimpleNamespace(verdict=verdict), price=price, listings=listings)


def row(hours_ago=None, price=None):
    last = None if hours_ago is None else (NOW - timedelta(hours=hours_ago)).isoformat()
    return {"last_alerted_at": last, "last_alerted_price": price}


def test_parser_builds_and_parses_every_command():
    parser = build_parser()
    assert parser.parse_args(["check", "--id", "3", "--notify"]).id == 3
    assert parser.parse_args(["check"]).id is None
    assert parser.parse_args(["schedule", "--interval-minutes", "30"]).interval_minutes == 30
    for command in ("add --brand A --model B --target-price 1", "list", "telegram-setup", "telegram-test"):
        parser.parse_args(command.split())


@pytest.mark.parametrize("command", ["add", "list", "check", "schedule", "telegram-setup", "telegram-test"])
def test_every_command_can_render_help(command):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([command, "--help"])
    assert exit_info.value.code == 0


def test_only_buy_verdicts_alert():
    assert not should_alert(row(), result("WAIT"), 48, NOW)
    assert not should_alert(row(), result("OVERPRICED"), 48, NOW)
    assert should_alert(row(), result("BUY"), 48, NOW)


def test_never_alerted_before_alerts():
    assert should_alert(row(hours_ago=None), result(), 48, NOW)


def test_recent_alert_at_same_price_is_suppressed():
    assert not should_alert(row(hours_ago=6, price=10_000), result(price=10_000), 48, NOW)


def test_alert_after_cooldown_even_at_same_price():
    assert should_alert(row(hours_ago=49, price=10_000), result(price=10_000), 48, NOW)


def test_clear_price_drop_alerts_inside_the_cooldown():
    dropped = 10_000 * (1 - ALERT_IMPROVEMENT_PCT / 100)
    assert should_alert(row(hours_ago=6, price=10_000), result(price=dropped), 48, NOW)


def test_small_price_drop_inside_the_cooldown_is_suppressed():
    assert not should_alert(row(hours_ago=6, price=10_000), result(price=9_900), 48, NOW)


def test_alert_history_without_a_price_falls_back_to_the_cooldown():
    assert not should_alert(row(hours_ago=6, price=None), result(price=1), 48, NOW)


def test_a_buy_backed_by_too_few_listings_does_not_alert():
    assert not should_alert(row(), result(listings=2), 48, NOW)


# --- run summary ------------------------------------------------------------

def checked(verdict):
    return SimpleNamespace(verdict=SimpleNamespace(verdict=verdict))


def watch(i, brand="Rolex", model="Sub"):
    return {"id": i, "brand": brand, "model": model}


def test_summary_counts_verdicts_and_names_failures_and_skips():
    from watchagent.agent import DailyBudgetExceeded
    from watchagent.cli import run_summary

    rows = [watch(1), watch(2), watch(3), watch(4, "Omega", "Speedmaster"), watch(5, "Grand Seiko", "Snowflake")]
    outcomes = [checked("WAIT"), checked("OVERPRICED"), checked("OVERPRICED"),
                RuntimeError("Error calling model (RESOURCE_EXHAUSTED): 429 quota"),
                DailyBudgetExceeded("used up")]
    assert run_summary(rows, outcomes) == (
        "WatchAgent check: 3 of 5 watches checked\n"
        "WAIT 1 · OVERPRICED 2\n"
        "Failed (1):\n"
        "- Omega Speedmaster: AI quota used up\n"
        "Skipped, daily AI budget used up (1): Grand Seiko Snowflake"
    )


def test_summary_when_every_check_fails():
    from watchagent.cli import run_summary

    text = run_summary([watch(1)], [RuntimeError("503 UNAVAILABLE")])
    assert text == "WatchAgent check: 0 of 1 watches checked\nFailed (1):\n- Rolex Sub: AI service busy"


@pytest.mark.parametrize("message,reason", [
    ("Error calling model 'gemini-3.5-flash' (RESOURCE_EXHAUSTED): 429", "AI quota used up"),
    ("503 UNAVAILABLE. This model is currently experiencing high demand", "AI service busy"),
    ("the agent finished without any listing that passed the checks", "no usable listing found"),
    ("something else\nwith detail", "something else"),
])
def test_failure_reasons_are_plain(message, reason):
    from watchagent.cli import _failure_reason

    assert _failure_reason(RuntimeError(message)) == reason


def test_summary_is_sent_only_when_notifying(monkeypatch):
    from watchagent import agent as agent_mod, cli, notify

    sent = []
    monkeypatch.setattr(agent_mod, "check_watch", lambda _agent, row: SimpleNamespace(
        watch="Rolex Sub", price=1.0, currency="USD", target_price=1.0, listings=3,
        verdict=SimpleNamespace(verdict="WAIT", trend_pct=None), confidence_label="3 listings", explanation="x"))
    monkeypatch.setattr(notify, "send_telegram", sent.append)
    cli._run_checks(None, [watch(1)], notify=False, cooldown_hours=48)
    assert sent == []
    cli._run_checks(None, [watch(1)], notify=True, cooldown_hours=48)
    assert sent == ["WatchAgent check: 1 of 1 watches checked\nWAIT 1"]
