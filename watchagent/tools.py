import re
import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import urlparse

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from watchagent import db

@lru_cache(maxsize=1)
def _tavily() -> TavilySearch:
    # Built on first use so importing this module doesn't need TAVILY_API_KEY.
    return TavilySearch(max_results=5, topic="general")

_SOURCE_DOMAINS: dict[str, list[str] | None] = {
    "chrono24": ["chrono24.com"],
    "watchcharts": ["watchcharts.com"],
    "general": None,
}

CONSENSUS_TOLERANCE = 0.05
GROUNDING_TOLERANCE = 0.01

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

_SPACES = "  "
_NUMBER = rf"\d{{1,3}}(?:[,.{_SPACES}]\d{{3}})+(?:[.,]\d{{1,2}})?|\d+(?:[.,]\d{{1,2}})?"

_ALREADY_RECORDED = (
    "ERROR: a price was already recorded for this check. Only one price is recorded "
    "per check, so stop calling record tools and give your short explanation."
)


@dataclass
class RecordedPrice:
    price: float
    currency: str
    confidence: str  # "CONFIRMED" or "SINGLE-SOURCE"
    sources: list[str]


@dataclass
class WatchContext:
    """Per-check state the tools read and write. The model never sees or sets it."""
    watch_id: int
    currency: str
    run_id: str
    search_results: dict[str, str] = field(default_factory=dict)  # url -> result line shown to the model
    recorded: RecordedPrice | None = None


class PriceObservation(BaseModel):
    price: float = Field(description="The price stated by this listing")
    source_url: str = Field(description="URL of the listing, exactly as returned by search_watch_price")
    snippet: str = Field(description="Exact text from that search result that states this price")


def _marker_alternatives(markers: list[str]) -> str:
    parts = []
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


def _price_is_grounded(price: float, snippet: str, currency: str,
                       tolerance: float = GROUNDING_TOLERANCE) -> bool:
    return any(
        amount > 0 and abs(amount - price) / amount <= tolerance
        for amount in _amounts_in_text(snippet, currency)
    )


def _source_from_url(url: str) -> str:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    for name, domains in _SOURCE_DOMAINS.items():
        if domains and any(host == d or host.endswith("." + d) for d in domains):
            return name
    return host or "unknown"


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def _agreeing_cluster(observations: list[PriceObservation],
                      tolerance: float = CONSENSUS_TOLERANCE) -> list[PriceObservation]:
    """Largest subset whose prices all lie within `tolerance` of the subset's lowest price."""
    best: list[PriceObservation] = []
    for low in observations:
        cluster = [o for o in observations if low.price <= o.price <= low.price * (1 + tolerance)]
        if len(cluster) > len(best):
            best = cluster
    return best


def _observation_error(ctx: WatchContext, observation: PriceObservation) -> str | None:
    """Why this observation can't be trusted, or None if it checks out."""
    if observation.price <= 0:
        return "price must be positive"
    line = ctx.search_results.get(observation.source_url)
    if line is None:
        known = ", ".join(list(ctx.search_results)[:8])
        return f"source_url was not in the search_watch_price results (copy one exactly: {known})"
    snippet = observation.snippet
    if not snippet.strip() or _normalize(snippet) not in _normalize(line):
        return "snippet is not text from that search result"
    if not _price_is_grounded(observation.price, snippet, ctx.currency):
        return f"price is not stated as a {ctx.currency} amount in the snippet"
    return None


@tool
def search_watch_price(query: str, runtime: ToolRuntime[WatchContext], source: str = "general") -> str:
    """Search the web for current luxury watch listings and prices.

    Use this to find what a specific watch (brand, model, reference number)
    is currently selling for on marketplaces, dealer sites, or price-tracking
    sites like Chrono24 or WatchCharts.

    Call this at least twice with different `source` values (e.g. "chrono24"
    then "watchcharts") to gather independent listings before recording a
    price with record_price_confirmed — a single source is not enough to
    confirm a price.

    Args:
        query: Search query, e.g. "Rolex Submariner 126610LN price"
        source: Which source to search: "chrono24", "watchcharts", or
            "general" (broad web search across dealers/marketplaces)
    """
    domains = _SOURCE_DOMAINS.get(source, None)
    params = {"query": query}
    if domains:
        params["include_domains"] = domains
    results = _tavily().invoke(params)
    items = results.get("results", []) if isinstance(results, dict) else []
    lines = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")
        line = f"- {title} ({url}): {content}"
        runtime.context.search_results[url] = line
        lines.append(line)
    return "\n".join(lines) if lines else f"No results found for source={source}."


@tool
def record_price(price: float, source_url: str, snippet: str,
                 runtime: ToolRuntime[WatchContext]) -> str:
    """Record the watch's current price from ONE listing (lower confidence).

    Use this only if record_price_confirmed was rejected or you found just one
    usable listing. The call is rejected unless `source_url` is a URL returned
    by search_watch_price, `snippet` is text from that same result, and `price`
    is written as an amount in that snippet.

    Args:
        price: The listing's price as a number, e.g. 15200
        source_url: URL of the listing, exactly as returned by search_watch_price
        snippet: Exact text from that search result that states this price
    """
    ctx = runtime.context
    if ctx.recorded:
        return _ALREADY_RECORDED
    error = _observation_error(ctx, PriceObservation(price=price, source_url=source_url, snippet=snippet))
    if error:
        return (f"ERROR: {error}. Nothing was recorded. Re-read the search_watch_price "
                "results and try again with exact text from them, or search again.")
    source = _source_from_url(source_url)
    db.record_price(ctx.watch_id, price, ctx.currency, source_url, snippet,
                    run_id=ctx.run_id, source=source)
    ctx.recorded = RecordedPrice(price, ctx.currency, "SINGLE-SOURCE", [source])
    return f"Recorded {price} {ctx.currency} from {source} (single source). Now give your explanation."


@tool
def record_price_confirmed(observations: list[PriceObservation],
                           runtime: ToolRuntime[WatchContext]) -> str:
    """Record the watch's price once 2+ different sites agree on it (preferred).

    Provide one PriceObservation per listing, from different sites (e.g. one
    from chrono24.com, one from watchcharts.com). This is rejected unless:
      - each observation's source_url is a URL returned by search_watch_price,
        its snippet is text from that result, and its price is written as an
        amount in that snippet, AND
      - at least 2 observations from different sites agree within 5%.

    On success, records the median of the agreeing observations. If the sites
    disagree, or you only have one usable listing, this returns an error: fall
    back to record_price with your single best listing.

    Args:
        observations: One PriceObservation per listing (need 2+ different sites)
    """
    ctx = runtime.context
    if ctx.recorded:
        return _ALREADY_RECORDED
    if len(observations) < 2:
        return "ERROR: need listings from at least 2 different sites to confirm a price."

    problems = [
        f"{_source_from_url(o.source_url)} ({o.price}): {error}"
        for o in observations if (error := _observation_error(ctx, o))
    ]
    if problems:
        return "ERROR: not recorded. " + "; ".join(problems)

    sites = {_source_from_url(o.source_url) for o in observations}
    if len(sites) < 2:
        return f"ERROR: listings must come from at least 2 different sites (got only: {', '.join(sites)})."

    cluster = _agreeing_cluster(observations)
    cluster_sites = sorted({_source_from_url(o.source_url) for o in cluster})
    if len(cluster_sites) < 2:
        summary = ", ".join(f"{_source_from_url(o.source_url)}={o.price}" for o in observations)
        return (f"ERROR: sites do not agree within {CONSENSUS_TOLERANCE:.0%} ({summary}). Not recorded. "
                "Use record_price with your single best listing instead.")

    price = statistics.median(o.price for o in cluster)
    names = " + ".join(cluster_sites)
    citation = "; ".join(f"{_source_from_url(o.source_url)}: {o.price} ({o.source_url})" for o in cluster)
    db.record_price(ctx.watch_id, price, ctx.currency, source_url=cluster[0].source_url,
                    snippet=f"Confirmed across {names} - {citation}",
                    run_id=ctx.run_id, source=names)
    ctx.recorded = RecordedPrice(price, ctx.currency, "CONFIRMED", cluster_sites)
    return f"Recorded {price} {ctx.currency}, confirmed across {names}. Now give your explanation."


ALL_TOOLS = [search_watch_price, record_price, record_price_confirmed]
