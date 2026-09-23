from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from watchagent import bot, db, notify

CHAT = 42


class Recorder:
    """Stands in for Telegram and the agent, and remembers what the bot did."""

    def __init__(self):
        self.sent: list[tuple[str, list | None]] = []
        self.answered: list[tuple[str, str]] = []
        self.checked: list[int] = []
        self.check_error: Exception | None = None
        self.questions: list[str] = []
        self.send_hook = None

    def services(self) -> bot.Services:
        return bot.Services(send=self.send, answer_callback=self.answer_callback,
                            check=self.check, answer=self.answer)

    def send(self, text, buttons=None):
        if self.send_hook:
            self.send_hook(text)
        self.sent.append((text, buttons))

    def answer_callback(self, callback_id, text=""):
        self.answered.append((callback_id, text))

    def check(self, row):
        self.checked.append(row["id"])
        if self.check_error:
            raise self.check_error
        return SimpleNamespace(message=lambda: "WAIT: fake result")

    def answer(self, question):
        self.questions.append(question)
        return "fake answer"

    @property
    def texts(self):
        return [t for t, _ in self.sent]


@pytest.fixture
def rec():
    return Recorder()


@pytest.fixture(autouse=True)
def database(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", Path(tmp_path / "bot.db"))
    db.init_db()
    db.add_watch("Rolex", "Submariner", 14_000, "126610LN")
    db.add_watch("Patek Philippe", "Aquanaut", 55_000, "5167A")


def message(text, chat=CHAT, update_id=1):
    return {"update_id": update_id, "message": {"chat": {"id": chat}, "text": text}}


def tap(data, chat=CHAT, update_id=1):
    return {"update_id": update_id,
            "callback_query": {"id": "cb1", "data": data, "message": {"chat": {"id": chat}}}}


def send(rec, update):
    bot.handle_update(update, rec.services(), CHAT)


def confirm_id(rec):
    _, buttons = rec.sent[-1]
    return int(buttons[0][1].split(":")[1])


# --- who is allowed -----------------------------------------------------------

def test_messages_from_other_chats_are_ignored(rec):
    send(rec, message("/remove 1", chat=999))
    send(rec, message("/list", chat=999))
    send(rec, message("what is my watchlist?", chat=999))
    assert rec.sent == [] and rec.questions == []
    assert len(db.list_watches()) == 2


def test_button_taps_from_other_chats_are_ignored(rec):
    send(rec, message("/remove 1"))
    pending = confirm_id(rec)
    send(rec, tap(f"ok:{pending}", chat=999))
    assert db.get_watch(1) is not None
    assert rec.answered == []


def test_updates_without_text_are_ignored(rec):
    send(rec, {"update_id": 1, "message": {"chat": {"id": CHAT}, "photo": []}})
    assert rec.sent == []


# --- read commands ------------------------------------------------------------

def test_list_shows_latest_verdict_or_not_checked(rec):
    db.record_check(1, "r1", 14_950.0, "USD", "SINGLE-SOURCE", "chrono24", "WAIT", 6.8, None, "x")
    send(rec, message("/list"))
    text = rec.texts[-1]
    assert "[1] Rolex Submariner (126610LN)" in text and "WAIT at 14,950 USD (+6.8% vs target)" in text
    assert "[2] Patek Philippe Aquanaut" in text and "not checked yet" in text


def test_status_and_history(rec):
    db.record_check(1, "r1", 15_000.0, "USD", "CONFIRMED", "chrono24 + watchcharts", "WAIT", 7.1, None, "first")
    db.record_check(1, "r2", 13_900.0, "USD", "SINGLE-SOURCE", "chrono24", "BUY", -0.7, -7.3, "second")
    send(rec, message("/status 1"))
    assert "BUY: 13,900.00 USD" in rec.texts[-1] and "second" in rec.texts[-1]
    send(rec, message("/history 1"))
    assert rec.texts[-1].index("13,900") < rec.texts[-1].index("15,000")  # newest first


def test_id_errors_are_explained(rec):
    send(rec, message("/status"))
    send(rec, message("/status abc"))
    send(rec, message("/status 99"))
    assert "Give a watch id" in rec.texts[0] and "Give a watch id" in rec.texts[1]
    assert "No watch with id 99" in rec.texts[2]


def test_unknown_command_shows_help(rec):
    send(rec, message("/wat"))
    assert "Unknown command" in rec.texts[-1] and "/list" in rec.texts[-1]


def test_command_with_bot_suffix_is_recognised(rec):
    send(rec, message("/list@MyWatchBot"))
    assert "Rolex" in rec.texts[-1]


# --- free text and /check -----------------------------------------------------

def test_free_text_goes_to_the_read_only_answerer(rec):
    send(rec, message("is the Aquanaut worth it?"))
    assert rec.questions == ["is the Aquanaut worth it?"] and rec.texts == ["fake answer"]


def assistant_call(tool, rec, **kwargs):
    return tool.func(runtime=SimpleNamespace(context=rec.services()), **kwargs)


def test_assistant_proposals_only_stage_a_confirmation(rec):
    out = assistant_call(bot.propose_add_watch, rec, brand="Tudor", model="Black Bay 58",
                         reference="79030N", target_price=3500)
    assert out == bot._PROPOSED and len(db.list_watches()) == 2
    assert rec.sent[-1][1][0][0] == "Confirm"
    assistant_call(bot.propose_removal, rec, watch_id=1)
    assistant_call(bot.propose_new_target, rec, watch_id=1, target_price=1)
    assert db.get_watch(1)["target_price"] == 14_000  # nothing changed yet
    send(rec, tap(f"ok:{confirm_id(rec)}"))
    assert db.get_watch(1)["target_price"] == 1


def test_assistant_proposal_errors_go_back_to_the_model(rec):
    assert "No watch with id 99" in assistant_call(bot.propose_removal, rec, watch_id=99)
    assert "must be positive" in assistant_call(bot.propose_new_target, rec, watch_id=1, target_price=-5)
    assert "Already tracking" in assistant_call(bot.propose_add_watch, rec, brand="rolex",
                                                model="submariner", reference="126610LN", target_price=1)
    assert rec.sent == []


def test_assistant_research_delegates_to_the_check(rec):
    assert assistant_call(bot.research_price, rec, watch_id=2) == "WAIT: fake result"
    rec.check_error = RuntimeError("no listings")
    assert assistant_call(bot.research_price, rec, watch_id=2) == "The check failed: no listings"


def test_assistant_has_no_tool_that_writes_directly():
    names = {t.name for t in bot.ASSISTANT_TOOLS}
    assert names == {"get_watchlist_status", "get_check_history", "research_price",
                     "propose_add_watch", "propose_new_target", "propose_removal"}


def test_chat_memory_keeps_the_latest_messages_in_order():
    for i in range(15):
        db.add_chat("user", f"m{i}")
    assert [c for _, c in db.recent_chat(3)] == ["m12", "m13", "m14"]


def test_check_runs_the_agent_and_replies(rec):
    send(rec, message("/check 2"))
    assert rec.checked == [2]
    assert "Checking Patek Philippe Aquanaut" in rec.texts[0] and rec.texts[-1] == "WAIT: fake result"


def test_failed_check_is_reported(rec):
    rec.check_error = RuntimeError("no price")
    send(rec, message("/check 1"))
    assert rec.texts[-1] == "Check failed: no price"


# --- confirm-before-write -----------------------------------------------------

def test_add_needs_a_confirm_tap(rec):
    send(rec, message("/add Omega | Speedmaster | 310.30 | 6000"))
    assert len(db.list_watches()) == 2  # nothing yet
    assert "Add Omega Speedmaster (310.30) with target 6,000 USD?" in rec.texts[-1]
    send(rec, tap(f"ok:{confirm_id(rec)}"))
    added = db.list_watches()[-1]
    assert (added["brand"], added["model"], added["target_price"], added["currency"]) == \
        ("Omega", "Speedmaster", 6000, "USD")
    assert rec.answered[-1] == ("cb1", "Done") and rec.texts[-1].startswith("Added [3]")


def test_add_with_currency(rec):
    send(rec, message("/add Omega | Speedmaster | 310.30 | 6,000 | sgd"))
    send(rec, tap(f"ok:{confirm_id(rec)}"))
    assert db.list_watches()[-1]["currency"] == "SGD"


def test_cancel_changes_nothing(rec):
    send(rec, message("/remove 1"))
    send(rec, tap(f"no:{confirm_id(rec)}"))
    assert db.get_watch(1) is not None and rec.texts[-1] == "Cancelled."


def test_a_confirm_can_only_run_once(rec):
    send(rec, message("/add Omega | Speedmaster | 310.30 | 6000"))
    pending = confirm_id(rec)
    send(rec, tap(f"ok:{pending}"))
    send(rec, tap(f"ok:{pending}"))
    assert len(db.list_watches()) == 3
    assert rec.answered[-1][1] == "Already handled or expired"


def test_expired_confirmation_does_nothing(rec):
    send(rec, message("/remove 1"))
    later = datetime.now(timezone.utc) + bot.PENDING_TTL + timedelta(minutes=1)
    bot.handle_callback(tap(f"ok:{confirm_id(rec)}")["callback_query"], rec.services(), now=later)
    assert db.get_watch(1) is not None and "expired" in rec.texts[-1]


def test_target_change_after_confirm(rec):
    send(rec, message("/target 1 13,500"))
    assert "from 14,000 to 13,500 USD" in rec.texts[-1]
    assert db.get_watch(1)["target_price"] == 14_000
    send(rec, tap(f"ok:{confirm_id(rec)}"))
    assert db.get_watch(1)["target_price"] == 13_500


def test_remove_deletes_the_watch_and_its_data(rec):
    db.record_check(1, "r1", 14_000.0, "USD", "CONFIRMED", "x", "BUY", 0, None, "x")
    db.record_price(1, 14_000.0, "USD", "u", "s", run_id="r1")
    send(rec, message("/remove 1"))
    send(rec, tap(f"ok:{confirm_id(rec)}"))
    assert db.get_watch(1) is None
    assert db.latest_check(1) is None and db.get_price_history(1) == []


@pytest.mark.parametrize("text,fragment", [
    ("/add Omega | Speedmaster", "Usage: /add"),
    ("/add Omega | Speedmaster | 310 | free", "isn't a valid target price"),
    ("/add Omega | Speedmaster | 310 | -5", "isn't a valid target price"),
    ("/add rolex | SUBMARINER | 126610ln | 1", "Already tracking"),
    ("/target 1", "Usage: /target"),
    ("/target 1 0", "Usage: /target"),
    ("/target 99 100", "No watch with id 99"),
    ("/remove", "Usage: /remove"),
    ("/remove 99", "No watch with id 99"),
])
def test_bad_input_is_rejected_without_staging_anything(rec, text, fragment):
    send(rec, message(text))
    assert fragment in rec.texts[-1] and rec.sent[-1][1] is None


def test_garbage_callback_data_is_rejected(rec):
    send(rec, tap("rm -rf"))
    send(rec, tap("ok:abc"))
    assert [t for _, t in rec.answered] == ["Unknown action", "Unknown action"]


# --- polling loop -------------------------------------------------------------

@pytest.fixture
def polling(monkeypatch):
    monkeypatch.setattr(bot, "TELEGRAM_CHAT_ID", str(CHAT))
    monkeypatch.setattr(bot, "require_telegram_config", lambda: None)
    state = SimpleNamespace(updates=[], fetched=[])

    def get_updates(offset):
        state.fetched.append(offset)
        return state.updates

    monkeypatch.setattr(notify, "get_updates", get_updates)
    return state


def test_run_once_advances_the_offset(rec, polling):
    polling.updates = [message("/list", update_id=10), message("/list", update_id=11)]
    assert bot.run_once(rec.services()) == 2
    assert db.get_state(bot.OFFSET_KEY) == "12"
    bot.run_once(rec.services())
    assert polling.fetched == [0, 12]


def test_idle_poll_leaves_the_database_unchanged(rec, polling):
    assert bot.run_once(rec.services()) == 0
    assert db.get_state(bot.OFFSET_KEY) is None


def test_a_failing_update_is_reported_and_does_not_block_the_rest(rec, polling):
    def flaky(text):
        if text.startswith("Checking"):
            raise RuntimeError("telegram down")

    rec.send_hook = flaky
    polling.updates = [message("/check 1", update_id=1), message("/list", update_id=2)]
    assert bot.run_once(rec.services()) == 2
    assert any("Sorry, that failed: telegram down" in t for t in rec.texts)
    assert any("Rolex" in t for t in rec.texts)  # the /list still went out
    assert db.get_state(bot.OFFSET_KEY) == "3"
