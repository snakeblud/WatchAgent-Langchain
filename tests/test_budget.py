from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from watchagent import agent as agent_mod, cli, db
from watchagent.agent import DailyBudgetExceeded, request_budget, retry_on


@pytest.fixture(autouse=True)
def database(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", Path(tmp_path / "budget.db"))
    db.init_db()


def fake_agent(daily_limit):
    model = GenericFakeChatModel(messages=iter(AIMessage(content="ok") for _ in range(10)))
    return create_agent(model=model, tools=[], middleware=[request_budget(daily_limit)])


def ask(agent):
    return agent.invoke({"messages": [{"role": "user", "content": "hi"}]})


def test_counter_starts_at_zero_and_counts():
    assert db.llm_requests_today() == 0
    db.count_llm_request()
    db.count_llm_request()
    assert db.llm_requests_today() == 2


def test_every_model_request_is_counted_and_refused_past_the_limit():
    agent = fake_agent(daily_limit=2)
    ask(agent)
    ask(agent)
    assert db.llm_requests_today() == 2
    with pytest.raises(DailyBudgetExceeded):
        ask(agent)
    assert db.llm_requests_today() == 2  # the refused request was never sent or counted


def test_scheduled_share_leaves_the_rest_for_the_bot():
    scheduled, bot = fake_agent(daily_limit=1), fake_agent(daily_limit=3)
    ask(scheduled)
    with pytest.raises(DailyBudgetExceeded):
        ask(scheduled)
    ask(bot)
    ask(bot)
    assert db.llm_requests_today() == 3


def test_budget_refusals_are_not_retried():
    assert not retry_on(DailyBudgetExceeded("used up"))
    assert retry_on(TimeoutError())


def test_stalest_watches_are_checked_first():
    a = db.add_watch("A", "a", 1)
    b = db.add_watch("B", "b", 1)
    c = db.add_watch("C", "c", 1)
    db.record_check(a, "r1", 1.0, "USD", "3 listings", "x", "BUY", 0, None, "")
    db.record_check(c, "r2", 1.0, "USD", "3 listings", "x", "BUY", 0, None, "")
    order = [r["id"] for r in cli.stalest_first(db.list_watches())]
    assert order == [b, a, c]  # never checked, then oldest check first


def test_budget_skips_are_not_failures(monkeypatch, capsys):
    def check_watch(_agent, row):
        if row["id"] == 1:
            raise RuntimeError("real failure")
        raise DailyBudgetExceeded("used up")

    monkeypatch.setattr(agent_mod, "check_watch", check_watch)
    rows = [{"id": i, "brand": "B", "model": "M"} for i in (1, 2, 3)]
    assert cli._run_checks(None, rows, notify=False, cooldown_hours=48) == 1
    assert "2 watch(es) skipped" in capsys.readouterr().out


def test_model_must_research_until_a_price_is_recorded():
    from types import SimpleNamespace

    from watchagent.agent import research_before_answering

    def request(recorded):
        ns = SimpleNamespace(runtime=SimpleNamespace(context=SimpleNamespace(recorded=recorded)),
                             tool_choice=None, response_format="answer tool")
        ns.override = lambda **kw: SimpleNamespace(**{**vars(ns), **kw})
        return ns

    seen = []
    research_before_answering.wrap_model_call(request(None), seen.append)
    research_before_answering.wrap_model_call(request("a price"), seen.append)
    assert (seen[0].tool_choice, seen[0].response_format) == ("required", None)
    assert (seen[1].tool_choice, seen[1].response_format) == (None, "answer tool")
