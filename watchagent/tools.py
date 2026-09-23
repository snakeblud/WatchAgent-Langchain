"""The research agent's tools, and the deterministic checks behind them.

The model searches, opens pages and submits listings; code then decides which
listings to trust. A listing counts only if its price is written in real search
text (grounding), it is for the right reference, and it is not an outlier. Once
MIN_LISTINGS agree, their median is recorded as the watch's price, once per check.
"""
import re
import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langchain_tavily import TavilyExtract, TavilySearch
from pydantic import BaseModel, Field

from watchagent import db

# A check records a price once this many listings of the right reference agree.
MIN_LISTINGS = 3
# Accepted listings further than this from their median are treated as outliers.
OUTLIER_PCT = 25.0
# A submitted price must be within this fraction of an amount written in its snippet.
GROUNDING_TOLERANCE = 0.01
# How many price lines of an opened page the model gets to see.
MAX_PAGE_LINES = 60

_SOURCE_DOMAINS: dict[str, list[str] | None] = {
    "chrono24": ["chrono24.com"],
    "watchcharts": ["watchcharts.com"],
    "general": None,
}

# How a currency is written next to an amount. Anything not listed is matched
# by its ISO code alone.
_CURRENCY_MARKERS: dict[str, list[str]] = {
    "USD": ["US$", "USD", "$"],
    "EUR": ["EUR", "€"],
    "GBP": ["GBP", "£"],
    "CHF": ["CHF"],
    "SGD": ["S$", "SGD"],
    "HKD": ["HK$", "HKD"],
    "JPY": ["JPY", "¥"],
}

# No-break and narrow no-break space: thousands separators in e.g. "14 500 €".
_SPACES = "  "
# "15,200", "14.500,00", "14 500" or a plain "15200" / "99.5".
_NUMBER = rf"\d{{1,3}}(?:[,.{_SPACES}]\d{{3}})+(?:[.,]\d{{1,2}})?|\d+(?:[.,]\d{{1,2}})?"


# --- search clients -----------------------------------------------------------
# Built on first use so importing this module doesn't need TAVILY_API_KEY.

@lru_cache(maxsize=1)
def _tavily() -> TavilySearch:
    return TavilySearch(max_results=5, topic="general")


@lru_cache(maxsize=1)
def _extractor() -> TavilyExtract:
    return TavilyExtract()


# --- data ---------------------------------------------------------------------

@dataclass
class RecordedPrice:
    price: float
    currency: str
    listings: int
    sources: list[str]


class Listing(BaseModel):
    price: float = Field(description="The price this listing states")
    source_url: str = Field(description="URL of the listing, exactly as returned by search_watch_price")
    snippet: str = Field(description="Exact text from that search result or opened page that states this price")
    reference_seen: str = Field(description='Reference number the listing is for, as written on it, e.g. "5167A-001"; "" if not shown')
    kind: Literal["asking", "estimate"] = Field(
        description='"asking" for a watch for sale, "estimate" for a market-price estimate such as WatchCharts'
    )


@dataclass
class WatchContext:
    """Per-check state the tools read and write. The model never sees or sets it."""
    watch_id: int
    currency: str
    run_id: str
    reference_no: str = ""
    # url -> every text the model was shown for it (search result, then any opened page)
    search_results: dict[str, str] = field(default_factory=dict)
    accepted: list[Listing] = field(default_factory=list)
    recorded: RecordedPrice | None = None


# --- reading prices out of text -----------------------------------------------

def _marker_alternatives(markers: list[str]) -> str:
    """A regex alternation of currency markers that won't match inside a word ("CHF" in "XCHFY")."""
    parts = []
    # Longest first, so "US$" wins over "$".
    for marker in sorted(markers, key=len, reverse=True):
        tail = r"(?![A-Za-z])" if marker[-1].isalpha() else ""
        parts.append(rf"(?<![A-Za-z]){re.escape(marker)}{tail}")
    return "|".join(parts)


@lru_cache(maxsize=None)
def _amount_patterns(currency: str) -> tuple[re.Pattern, ...]:
    markers = _CURRENCY_MARKERS.get(currency.upper(), [currency.upper()])
    gap = rf"[\s{_SPACES}]?"
    thousand = rf"(?:{gap}(?P<k>[kK])\b)?"
    patterns = [re.compile(rf"(?:{_marker_alternatives(markers)}){gap}(?P<num>{_NUMBER}){thousand}")]
    # "1234 $" is far more often a year next to a price than a price, so only
    # codes and the euro sign are accepted after the number.
    suffix_markers = [m for m in markers if m[-1].isalpha() or m == "€"]
    if suffix_markers:
        patterns.append(re.compile(
            rf"(?P<num>{_NUMBER}){thousand}{gap}(?:{_marker_alternatives(suffix_markers)})"
        ))
    return tuple(patterns)


def _parse_amount(raw: str) -> float:
    """Turn "15,200", "14.500,00" or "14 500" into a float, guessing which separator is the decimal."""
    s = raw.replace(" ", "").replace(" ", "")
    has_comma, has_dot = "," in s, "." in s
    decimal = None
    if has_comma and has_dot:
        decimal = "," if s.rfind(",") > s.rfind(".") else "."
    elif has_comma or has_dot:
        sep = "," if has_comma else "."
        parts = s.split(sep)
        # A lone separator followed by exactly 3 digits is a thousands separator.
        decimal = sep if len(parts) == 2 and len(parts[1]) != 3 else None
    if decimal:
        thousands = "." if decimal == "," else ","
        s = s.replace(thousands, "").replace(decimal, ".")
    else:
        s = s.replace(",", "").replace(".", "")
    return float(s)


def _amounts_in_text(text: str, currency: str) -> list[float]:
    """Amounts written as a price in `currency`, e.g. "$15,200", "16.8K USD" or "14.500,00 €".

    Bare numbers (years, percentages, reference numbers) are deliberately ignored.
    """
    amounts = []
    for pattern in _amount_patterns(currency):
        for match in pattern.finditer(text):
            value = _parse_amount(match.group("num"))
            amounts.append(value * 1000 if match.group("k") else value)
    return amounts


def _price_is_grounded(price: float, snippet: str, currency: str) -> bool:
    return any(
        amount > 0 and abs(amount - price) / amount <= GROUNDING_TOLERANCE
        for amount in _amounts_in_text(snippet, currency)
    )


# --- checking a listing -------------------------------------------------------

def _source_from_url(url: str) -> str:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    for name, domains in _SOURCE_DOMAINS.items():
        if domains and any(host == d or host.endswith("." + d) for d in domains):
            return name
    return host or "unknown"


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def _ref_key(ref: str) -> str:
    """A reference's identifying part, upper-case without punctuation: 5167A-001 -> 5167A, 200V/701 -> 200V."""
    return re.sub(r"[^A-Z0-9]", "", re.split(r"[-/ ]", ref.strip().upper())[0])


def _same_reference(seen: str, wanted: str) -> bool:
    """126610 or 126610LN-0001 match 126610LN; 126610LV (a different watch) does not."""
    seen, wanted = _ref_key(seen), _ref_key(wanted)
    return seen.startswith(wanted) or wanted.startswith(seen)


def _listing_error(ctx: WatchContext, listing: Listing) -> str | None:
    """Why this listing can't be trusted, or None if it checks out."""
    if listing.price <= 0:
        return "price must be positive"
    text = ctx.search_results.get(listing.source_url)
    if text is None:
        known = ", ".join(list(ctx.search_results)[:8])
        return f"source_url was not in the search_watch_price results (copy one exactly: {known})"
    if not listing.snippet.strip() or _normalize(listing.snippet) not in _normalize(text):
        return "snippet is not text from that search result or page"
    if not _price_is_grounded(listing.price, listing.snippet, ctx.currency):
        return f"price is not stated as a {ctx.currency} amount in the snippet"
    wanted = _ref_key(ctx.reference_no)
    if wanted and listing.reference_seen and not _same_reference(listing.reference_seen, wanted):
        return f"reference {listing.reference_seen} is not {ctx.reference_no}"
    if wanted and wanted not in re.sub(r"[^A-Z0-9]", "", text.upper()):
        return f"reference {ctx.reference_no} does not appear in that result or page"
    return None


def _without_outliers(listings: list[Listing]) -> tuple[list[Listing], list[Listing]]:
    """Split into (kept, outliers). Needs 3+ listings to tell which one is odd."""
    if len(listings) < 3:
        return listings, []
    median = statistics.median(l.price for l in listings)
    kept = [l for l in listings if abs(l.price - median) / median * 100 <= OUTLIER_PCT]
    return kept, [l for l in listings if l not in kept]


# --- recording the price ------------------------------------------------------

def finalize(ctx: WatchContext) -> RecordedPrice | None:
    """Record the median of the accepted listings as this check's price, if there are any.

    Safe to call again: a check records at most one price.
    """
    if ctx.recorded or not ctx.accepted:
        return ctx.recorded
    kept, _ = _without_outliers(ctx.accepted)
    price = statistics.median(l.price for l in kept)
    sources = sorted({_source_from_url(l.source_url) for l in kept})
    citation = "; ".join(f"{_source_from_url(l.source_url)}: {l.price} ({l.source_url})" for l in kept)
    db.record_price(ctx.watch_id, price, ctx.currency, source_url=kept[0].source_url,
                    snippet=f"Median of {len(kept)} listing(s) - {citation}",
                    run_id=ctx.run_id, source=" + ".join(sources))
    ctx.recorded = RecordedPrice(price, ctx.currency, len(kept), sources)
    return ctx.recorded


# --- tools --------------------------------------------------------------------

@tool
def search_watch_price(query: str, runtime: ToolRuntime[WatchContext], source: str = "general") -> str:
    """Search the web for current listings and prices of a watch.

    Args:
        query: Search query, e.g. "Rolex Submariner 126610LN price"
        source: "chrono24", "watchcharts", or "general" (broad web search
            across dealers and marketplaces)
    """
    domains = _SOURCE_DOMAINS.get(source)
    params = {"query": query}
    if domains:
        params["include_domains"] = domains
    results = _tavily().invoke(params)
    items = results.get("results", []) if isinstance(results, dict) else []
    lines = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url = item.get("url", "")
        line = f"- {item.get('title', '')} ({url}): {item.get('content', '')}"
        # Remembered so submitted listings can be checked against what was really shown.
        runtime.context.search_results[url] = line
        lines.append(line)
    return "\n".join(lines) if lines else f"No results found for source={source}."


@tool
def open_listing(url: str, runtime: ToolRuntime[WatchContext]) -> str:
    """Open a page from the search results to read its details and prices.

    Use this when a search snippet doesn't show the price or reference clearly,
    or to read a marketplace page that lists several watches. Returns the
    page's lines that mention a price.

    Args:
        url: A URL exactly as returned by search_watch_price
    """
    ctx = runtime.context
    if url not in ctx.search_results:
        return "ERROR: only URLs returned by search_watch_price can be opened."
    results = _extractor().invoke({"urls": [url]}).get("results") or []
    page = (results[0].get("raw_content") or "") if results else ""
    if not page:
        return "ERROR: could not read that page. Try another result."
    ctx.search_results[url] += "\n" + page
    price_lines = [line.strip() for line in page.splitlines() if _amounts_in_text(line, ctx.currency)]
    if not price_lines:
        return f"The page has no {ctx.currency} prices."
    return "\n".join(price_lines[:MAX_PAGE_LINES])


@tool
def submit_listings(listings: list[Listing], runtime: ToolRuntime[WatchContext]) -> str:
    """Submit listings you found for this watch. Code checks each one and says what it still needs.

    Submit every usable listing, one per watch for sale (and market-price
    estimates marked kind="estimate"). You can call this several times: accepted
    listings add up. Once MIN_LISTINGS asking prices of the right reference are
    accepted, the median is recorded as the watch's price and you are done.

    Args:
        listings: The listings, each with price, source_url, snippet,
            reference_seen and kind
    """
    ctx = runtime.context
    if ctx.recorded:
        return "The price is already recorded. Give your short explanation."

    rejected = []
    for listing in listings:
        # A marketplace page can list several watches, so a listing is its URL *and* price.
        known = {(l.source_url, l.price) for l in ctx.accepted}
        if (listing.source_url, listing.price) in known:
            rejected.append(f"{listing.source_url}: already submitted")
        elif listing.kind == "estimate":
            rejected.append(f"{listing.source_url}: market estimates don't count as listings")
        elif error := _listing_error(ctx, listing):
            rejected.append(f"{listing.source_url} ({listing.price}): {error}")
        else:
            ctx.accepted.append(listing)

    kept, outliers = _without_outliers(ctx.accepted)
    rejected += [f"{l.source_url} ({l.price}): more than {OUTLIER_PCT:g}% from the other listings"
                 for l in outliers]
    report = f"Accepted {len(kept)} asking-price listing(s)"
    if kept:
        report += f", median {statistics.median(l.price for l in kept):,.0f} {ctx.currency}"
    if rejected:
        report += ". Rejected: " + "; ".join(rejected)

    if len(kept) >= MIN_LISTINGS:
        recorded = finalize(ctx)
        return f"{report}. Recorded {recorded.price:,.0f} {ctx.currency}. Now give your short explanation."
    return (f"{report}. Need {MIN_LISTINGS - len(kept)} more: search with a more specific query "
            "(e.g. the full reference), try another source, or open a page that lists several watches.")


@tool
def save_note(note: str, runtime: ToolRuntime[WatchContext]) -> str:
    """Save a short tip for the next time this watch is checked.

    E.g. which query found good listings, or which kind of listing is a trap
    ("generic Aquanaut pages mix in the 5968"). Keep it to one sentence.

    Args:
        note: The tip, one sentence
    """
    db.add_note(runtime.context.watch_id, note.strip()[:300])
    return "Saved."


ALL_TOOLS = [search_watch_price, open_listing, submit_listings, save_note]
