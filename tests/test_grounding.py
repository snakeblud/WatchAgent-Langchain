from pathlib import Path
from types import SimpleNamespace

import pytest

from watchagent import db, tools
from watchagent.tools import (
    PriceObservation,
    WatchContext,
    _agreeing_cluster,
    _amounts_in_text,
    _price_is_grounded,
    _source_from_url,
)


def obs(price, url="https://www.chrono24.com/a"):
    return PriceObservation(price=price, source_url=url, snippet="")


# --- amounts -----------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Pre-owned, $15,200", [15200]),
    ("Price US$ 5,570.50 today", [5570.5]),
    ("now 16.8K USD", [16800]),
    ("was $16.8K", [16800]),
    ("15,200 USD", [15200]),
    ("$ 5,570 + $ 65 for shipping", [5570, 65]),
])
def test_usd_amounts(text, expected):
    assert sorted(_amounts_in_text(text, "USD")) == sorted(expected)


def test_bare_numbers_are_not_prices():
    text = "Rolex 126610LN, 2021 model, up 10.6% (4.1%) in 30 days, 41 mm"
    assert _amounts_in_text(text, "USD") == []


def test_year_before_a_dollar_sign_is_not_a_price():
    assert _amounts_in_text("Bought 2020 $15,200", "USD") == [15200]


def test_european_number_format():
    assert _amounts_in_text("14.500,00 €", "EUR") == [14500]
    assert _amounts_in_text("€ 14.500", "EUR") == [14500]


def test_amounts_are_matched_per_currency():
    text = "$15,200 or S$ 20,000 or EUR 14,000"
    assert _amounts_in_text(text, "USD") == [15200]
    assert _amounts_in_text(text, "SGD") == [20000]
    assert _amounts_in_text(text, "EUR") == [14000]


# --- grounding ---------------------------------------------------------------

def test_price_must_match_an_amount_not_any_number():
    snippet = "Submariner 126610LN, $ 5,570 + $ 65 for shipping, up 10.6%"
    assert _price_is_grounded(5570, snippet, "USD")
    assert not _price_is_grounded(126610, snippet, "USD")
    assert not _price_is_grounded(10.6, snippet, "USD")


def test_grounding_tolerance_is_one_percent():
    assert _price_is_grounded(15_100, "$15,200", "USD")      # 0.66% off
    assert not _price_is_grounded(15_000, "$15,200", "USD")  # 1.3% off


def test_wrong_currency_is_not_grounded():
    assert not _price_is_grounded(14_000, "EUR 14,000", "USD")


# --- clustering and sources --------------------------------------------------

def test_cluster_is_mutual_not_anchor_relative():
    # 100 and 105 are within 5%; 110 is within 5% of 105 but not of 100.
    cluster = _agreeing_cluster([obs(100), obs(105), obs(110)])
    assert sorted(o.price for o in cluster) == [100, 105]


def test_cluster_never_divides_by_zero():
    assert _agreeing_cluster([obs(0), obs(100)]) is not None


@pytest.mark.parametrize("url,source", [
    ("https://www.chrono24.com/rolex/x.htm", "chrono24"),
    ("https://chrono24.com/x", "chrono24"),
    ("https://watchcharts.com/watch_model/1", "watchcharts"),
    ("https://www.hodinkee.com/shop/a", "hodinkee.com"),
    ("https://notchrono24.com/x", "notchrono24.com"),
])
def test_source_comes_from_the_url_domain(url, source):
    assert _source_from_url(url) == source


# --- record tools ------------------------------------------------------------

CHRONO_URL = "https://www.chrono24.com/rolex/sub.htm"
CHARTS_URL = "https://watchcharts.com/watch_model/126610ln"
CHRONO_LINE = f"- Rolex Submariner ({CHRONO_URL}): Pre-owned $15,200 incl. box and papers"
CHARTS_LINE = f"- Rolex Submariner 126610LN ({CHARTS_URL}): Market price $15,600"


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", Path(tmp_path / "test.db"))
    db.init_db()
    watch_id = db.add_watch("Rolex", "Submariner", 15_000, "126610LN")
    return WatchContext(
        watch_id=watch_id, currency="USD", run_id="run1",
        search_results={CHRONO_URL: CHRONO_LINE, CHARTS_URL: CHARTS_LINE},
    )


def call(tool, ctx, **kwargs):
    return tool.func(runtime=SimpleNamespace(context=ctx), **kwargs)


def rows(watch_id):
    return db.get_price_history(watch_id)


def test_record_price_accepts_a_real_listing(ctx):
    out = call(tools.record_price, ctx, price=15200, source_url=CHRONO_URL,
               snippet="Pre-owned $15,200 incl. box")
    assert out.startswith("Recorded")
    assert ctx.recorded.confidence == "SINGLE-SOURCE" and ctx.recorded.sources == ["chrono24"]
    (row,) = rows(ctx.watch_id)
    assert (row["run_id"], row["source"], row["price"]) == ("run1", "chrono24", 15200)


def test_record_price_rejects_a_url_search_never_returned(ctx):
    out = call(tools.record_price, ctx, price=15200, source_url="https://evil.example/x",
               snippet="$15,200")
    assert out.startswith("ERROR") and "not in the search" in out
    assert rows(ctx.watch_id) == [] and ctx.recorded is None


def test_record_price_rejects_an_invented_snippet(ctx):
    out = call(tools.record_price, ctx, price=9000, source_url=CHRONO_URL,
               snippet="Brand new $9,000 with warranty")
    assert out.startswith("ERROR") and "not text from that search result" in out
    assert rows(ctx.watch_id) == []


def test_record_price_rejects_a_price_not_in_the_snippet(ctx):
    out = call(tools.record_price, ctx, price=14000, source_url=CHRONO_URL,
               snippet="Pre-owned $15,200 incl. box")
    assert out.startswith("ERROR") and "not stated" in out


def test_only_one_price_is_recorded_per_check(ctx):
    call(tools.record_price, ctx, price=15200, source_url=CHRONO_URL, snippet="$15,200")
    out = call(tools.record_price, ctx, price=15600, source_url=CHARTS_URL, snippet="$15,600")
    assert out.startswith("ERROR: a price was already recorded")
    assert len(rows(ctx.watch_id)) == 1


def test_confirmed_records_the_median_of_agreeing_sites(ctx):
    out = call(tools.record_price_confirmed, ctx, observations=[
        PriceObservation(price=15200, source_url=CHRONO_URL, snippet="Pre-owned $15,200"),
        PriceObservation(price=15600, source_url=CHARTS_URL, snippet="Market price $15,600"),
    ])
    assert out.startswith("Recorded 15400")
    assert ctx.recorded.confidence == "CONFIRMED"
    assert ctx.recorded.sources == ["chrono24", "watchcharts"]
    (row,) = rows(ctx.watch_id)
    assert row["source"] == "chrono24 + watchcharts" and row["price"] == 15400


def test_confirmed_rejects_sites_that_disagree(ctx):
    ctx.search_results[CHARTS_URL] = f"- Rolex ({CHARTS_URL}): Market price $19,000"
    out = call(tools.record_price_confirmed, ctx, observations=[
        PriceObservation(price=15200, source_url=CHRONO_URL, snippet="$15,200"),
        PriceObservation(price=19000, source_url=CHARTS_URL, snippet="$19,000"),
    ])
    assert out.startswith("ERROR") and "do not agree" in out
    assert rows(ctx.watch_id) == [] and ctx.recorded is None


def test_confirmed_needs_two_different_sites(ctx):
    other = "https://www.chrono24.com/rolex/other.htm"
    ctx.search_results[other] = f"- Rolex ({other}): $15,300"
    out = call(tools.record_price_confirmed, ctx, observations=[
        PriceObservation(price=15200, source_url=CHRONO_URL, snippet="$15,200"),
        PriceObservation(price=15300, source_url=other, snippet="$15,300"),
    ])
    assert out.startswith("ERROR") and "2 different sites" in out


def test_confirmed_rejects_one_bad_observation_and_names_it(ctx):
    out = call(tools.record_price_confirmed, ctx, observations=[
        PriceObservation(price=15200, source_url=CHRONO_URL, snippet="$15,200"),
        PriceObservation(price=15600, source_url="https://nope.example/x", snippet="$15,600"),
    ])
    assert out.startswith("ERROR") and "nope.example" in out
    assert rows(ctx.watch_id) == []
