from pathlib import Path
from types import SimpleNamespace

import pytest

from watchagent import db, tools
from watchagent.tools import (
    Listing,
    WatchContext,
    _amounts_in_text,
    _price_is_grounded,
    _ref_key,
    _same_reference,
    _source_from_url,
)


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


# --- references and sources -------------------------------------------------

@pytest.mark.parametrize("ref,key", [
    ("5167A-001", "5167A"), ("126610LN", "126610LN"), ("200V/701", "200V"),
    ("sbga211", "SBGA211"), ("", ""),
])
def test_reference_key(ref, key):
    assert _ref_key(ref) == key


@pytest.mark.parametrize("seen,wanted,same", [
    ("126610", "126610LN", True),       # listing omits the suffix
    ("126610LN-0001", "126610LN", True),
    ("5167A-001", "5167A", True),
    ("126610LV", "126610LN", False),    # Starbucks, a different watch
    ("5968A", "5167A", False),
])
def test_same_reference(seen, wanted, same):
    assert _same_reference(seen, wanted) is same


@pytest.mark.parametrize("url,source", [
    ("https://www.chrono24.com/rolex/x.htm", "chrono24"),
    ("https://chrono24.com/x", "chrono24"),
    ("https://watchcharts.com/watch_model/1", "watchcharts"),
    ("https://www.hodinkee.com/shop/a", "hodinkee.com"),
    ("https://notchrono24.com/x", "notchrono24.com"),
])
def test_source_comes_from_the_url_domain(url, source):
    assert _source_from_url(url) == source


# --- submit_listings ---------------------------------------------------------

URLS = [f"https://www.chrono24.com/rolex/sub-{i}.htm" for i in range(5)]
DEALER = "https://www.bobswatches.com/rolex-submariner-126610ln"
CHARTS = "https://watchcharts.com/watch_model/126610"
RESULTS = {
    URLS[0]: f"- Rolex Submariner 126610LN ({URLS[0]}): Pre-owned $15,200 box and papers",
    URLS[1]: f"- Rolex Submariner 126610LN ({URLS[1]}): Unworn $15,600",
    URLS[2]: f"- Rolex Submariner Date 126610LN ({URLS[2]}): $14,900",
    URLS[3]: f"- Rolex Submariner 126610LV Starbucks ({URLS[3]}): $17,500",
    URLS[4]: f"- Rolex Submariner 126610LN ({URLS[4]}): Parts watch $4,000",
    DEALER: f"- Submariner 126610LN at Bob's ({DEALER}): $15,450",
    CHARTS: f"- Submariner Date 126610LN ({CHARTS}): Market Price $13,706",
}


def listing(url, price, snippet=None, ref="126610LN", kind="asking"):
    return Listing(price=price, source_url=url, snippet=snippet or f"${price:,.0f}",
                   reference_seen=ref, kind=kind)


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", Path(tmp_path / "test.db"))
    db.init_db()
    watch_id = db.add_watch("Rolex", "Submariner", 15_000, "126610LN")
    return WatchContext(watch_id=watch_id, currency="USD", run_id="run1",
                        reference_no="126610LN", search_results=dict(RESULTS))


def submit(ctx, *listings):
    return tools.submit_listings.func(listings=list(listings), runtime=SimpleNamespace(context=ctx))


def test_three_good_listings_record_their_median(ctx):
    out = submit(ctx, listing(URLS[0], 15200), listing(URLS[1], 15600), listing(DEALER, 15450))
    assert "Recorded 15,450 USD" in out
    assert ctx.recorded.listings == 3 and ctx.recorded.sources == ["bobswatches.com", "chrono24"]
    (row,) = db.get_price_history(ctx.watch_id)
    assert (row["price"], row["run_id"]) == (15450, "run1")


def test_listings_add_up_across_calls_and_feedback_says_what_is_missing(ctx):
    out = submit(ctx, listing(URLS[0], 15200))
    assert "Accepted 1" in out and "Need 2 more" in out and ctx.recorded is None
    out = submit(ctx, listing(URLS[1], 15600), listing(URLS[0], 15200))
    assert "already submitted" in out and "Need 1 more" in out
    assert "Recorded" in submit(ctx, listing(URLS[2], 14900))


def test_wrong_reference_is_rejected_with_the_reason(ctx):
    out = submit(ctx, listing(URLS[3], 17500, ref="126610LV"))
    assert "reference 126610LV is not 126610LN" in out and ctx.accepted == []


def test_reference_must_appear_on_the_page_even_if_the_model_claims_it(ctx):
    ctx.search_results[URLS[3]] = f"- Rolex Submariner Starbucks ({URLS[3]}): $17,500"
    out = submit(ctx, listing(URLS[3], 17500, ref="126610LN"))
    assert "does not appear" in out and ctx.accepted == []


def test_invented_url_snippet_or_price_is_rejected(ctx):
    out = submit(ctx,
                 listing("https://evil.example/x", 15200),
                 listing(URLS[0], 9000, snippet="Brand new $9,000"),
                 listing(URLS[1], 14000, snippet="Unworn $15,600"))
    assert "not in the search_watch_price results" in out
    assert "not text from that search result" in out
    assert "not stated as a USD amount" in out
    assert ctx.accepted == []


def test_estimates_do_not_count_as_listings(ctx):
    out = submit(ctx, listing(CHARTS, 13706, snippet="Market Price $13,706", ref="126610", kind="estimate"))
    assert "market estimates don't count" in out and ctx.accepted == []


def test_outlier_is_dropped_before_the_median(ctx):
    out = submit(ctx, listing(URLS[0], 15200), listing(URLS[1], 15600),
                 listing(URLS[2], 14900), listing(URLS[4], 4000))
    assert "more than 25% from the other listings" in out
    assert ctx.recorded.listings == 3 and ctx.recorded.price == 15200


def test_nothing_is_recorded_twice(ctx):
    submit(ctx, listing(URLS[0], 15200), listing(URLS[1], 15600), listing(DEALER, 15450))
    assert "already recorded" in submit(ctx, listing(URLS[2], 14900))
    assert len(db.get_price_history(ctx.watch_id)) == 1


def test_finalize_records_fewer_listings_as_lower_confidence(ctx):
    submit(ctx, listing(URLS[0], 15200), listing(URLS[1], 15600))
    recorded = tools.finalize(ctx)
    assert recorded.listings == 2 and recorded.price == 15400


def test_finalize_with_nothing_accepted_records_nothing(ctx):
    assert tools.finalize(ctx) is None and db.get_price_history(ctx.watch_id) == []


def test_open_listing_only_opens_search_results(ctx, monkeypatch):
    runtime = SimpleNamespace(context=ctx)
    assert tools.open_listing.func(url="https://evil.example/x", runtime=runtime).startswith("ERROR")
    page = "Rolex Submariner 126610LN\nListing A $15,300\nShipping info\nListing B $15,350"
    fake = SimpleNamespace(invoke=lambda _: {"results": [{"raw_content": page}]})
    monkeypatch.setattr(tools, "_extractor", lambda: fake)
    out = tools.open_listing.func(url=URLS[0], runtime=runtime)
    assert out == "Listing A $15,300\nListing B $15,350"
    # Prices read from the opened page can now be submitted.
    assert "Accepted 1" in submit(ctx, listing(URLS[0], 15350, snippet="Listing B $15,350"))


def test_save_note_is_bound_to_the_watch(ctx):
    tools.save_note.func(note="  search the full ref  ", runtime=SimpleNamespace(context=ctx))
    assert db.recent_notes(ctx.watch_id) == ["search the full ref"]


def test_several_listings_on_one_page_count_separately(ctx):
    page = URLS[0]
    ctx.search_results[page] += " also $15,300 and $15,450"
    out = submit(ctx, listing(page, 15200), listing(page, 15300), listing(page, 15450))
    assert "Recorded 15,300 USD" in out
