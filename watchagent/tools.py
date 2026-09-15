import re
import statistics

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from watchagent import db

_tavily = TavilySearch(max_results=5, topic="general")

_SOURCE_DOMAINS: dict[str, list[str] | None] = {
    "chrono24": ["chrono24.com"],
    "watchcharts": ["watchcharts.com"],
    "general": None,
}

CONSENSUS_TOLERANCE = 0.05

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers_in_text(text: str) -> list[float]:
    numbers = []
    for match in _NUMBER_RE.findall(text):
        try:
            numbers.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return numbers


def _price_is_grounded(price: float, snippet: str, tolerance: float = 0.01) -> bool:
    for number in _numbers_in_text(snippet):
        if number == 0:
            continue
        if abs(number - price) / number <= tolerance:
            return True
    return False


class PriceObservation(BaseModel):
    source: str = Field(description='Which source this came from: "chrono24", "watchcharts", or "general"')
    price: float = Field(description="The price stated by this source")
    source_url: str = Field(description="URL of the listing")
    snippet: str = Field(description="Exact text from the search result that states this price")


def _agreeing_cluster(observations: list[PriceObservation],
                       tolerance: float = CONSENSUS_TOLERANCE) -> list[PriceObservation]:
    """Largest subset of observations whose prices are mutually within tolerance."""
    best: list[PriceObservation] = []
    for anchor in observations:
        cluster = [o for o in observations
                   if abs(o.price - anchor.price) / anchor.price <= tolerance]
        if len(cluster) > len(best):
            best = cluster
    return best


@tool
def search_watch_price(query: str, source: str = "general") -> str:
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
    results = _tavily.invoke(params)
    items = results.get("results", []) if isinstance(results, dict) else []
    lines = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")
        lines.append(f"- {title} ({url}): {content}")
    return "\n".join(lines) if lines else f"No results found for source={source}."


@tool
def record_price(watch_id: int, price: float, currency: str, source_url: str,
                  snippet: str) -> str:
    """Log an observed price for a watch to the price history database.

    Always call this after finding a current price, even if the price is not
    a good deal, so future checks can see the trend.

    `snippet` must be the exact text from search_watch_price results that
    states this price (e.g. "- Rolex Submariner 126610LN (chrono24.com):
    Pre-owned, $15,200"). The call is rejected if `price` does not actually
    appear in `snippet` — this prevents recording a number you inferred or
    misremembered instead of one an actual listing stated.

    Args:
        watch_id: The id of the watch (from list_watchlist)
        price: The observed price as a number, e.g. 15200
        currency: Currency code, e.g. "USD"
        source_url: URL where this price was observed (must be a URL from
            the search_watch_price results, not invented)
        snippet: Exact text from the search result that states this price
    """
    if not source_url or not snippet:
        return (
            "ERROR: source_url and snippet are required. Cite the exact "
            "search result (url and text) that states this price, then "
            "call record_price again."
        )
    if not _price_is_grounded(price, snippet):
        return (
            f"ERROR: price {price} was not found in the provided snippet, so it "
            "was NOT recorded. Re-read the search_watch_price results and call "
            "record_price again with a price that is actually stated in the "
            "snippet you cite, or search again if no listing gives a clear price."
        )
    db.record_price(watch_id, price, currency, source_url, snippet)
    return f"Recorded price {price} {currency} for watch {watch_id}."


@tool
def record_price_confirmed(watch_id: int, currency: str,
                            observations: list[PriceObservation]) -> str:
    """Record a price that's been cross-checked against 2+ independent sources.

    Provide one PriceObservation per source you searched with different
    `source` values in search_watch_price (e.g. one from "chrono24", one from
    "watchcharts"). This is rejected unless:
      - each observation's price actually appears in its own snippet, AND
      - at least 2 observations from different sources agree within 5% of
        each other.

    On success, records the median of the agreeing observations. If your
    sources disagree, or you only have one usable source, this tool returns
    an error — fall back to record_price with your single best grounded
    source instead, and note the lower confidence in your final reasoning.

    Args:
        watch_id: The id of the watch (from list_watchlist)
        currency: Currency code, e.g. "USD"
        observations: One PriceObservation per source (need 2+ distinct sources)
    """
    if len(observations) < 2:
        return "ERROR: need observations from at least 2 different sources to confirm a price."

    ungrounded = [o for o in observations if not _price_is_grounded(o.price, o.snippet)]
    if ungrounded:
        bad = ", ".join(f"{o.source} ({o.price})" for o in ungrounded)
        return f"ERROR: price not found in the cited snippet for: {bad}. Not recorded."

    distinct_sources = {o.source for o in observations}
    if len(distinct_sources) < 2:
        return (f"ERROR: observations must come from at least 2 different sources "
                f"(got only: {', '.join(distinct_sources)}).")

    cluster = _agreeing_cluster(observations)
    cluster_sources = {o.source for o in cluster}
    if len(cluster) < 2 or len(cluster_sources) < 2:
        summary = ", ".join(f"{o.source}={o.price}" for o in observations)
        return (
            f"ERROR: sources do not agree closely enough to confirm ({summary}). "
            "Not recorded. Use record_price with your best single grounded "
            "source instead, and mention the low confidence in your reasoning."
        )

    confirmed_price = statistics.median(o.price for o in cluster)
    citation = "; ".join(f"{o.source}: {o.price} ({o.source_url})" for o in cluster)
    db.record_price(
        watch_id, confirmed_price, currency,
        source_url=cluster[0].source_url,
        snippet=f"Confirmed across {len(cluster)} sources - {citation}",
    )
    return (f"Recorded confirmed price {confirmed_price} {currency} for watch {watch_id} "
            f"(agreement across {len(cluster)} sources: {cluster_sources}).")


@tool
def get_price_history(watch_id: int) -> str:
    """Get past recorded prices for a watch, most recent first.

    Use this to judge whether the current price is trending up or down,
    and how it compares to past observations.

    Args:
        watch_id: The id of the watch (from list_watchlist)
    """
    rows = db.get_price_history(watch_id)
    if not rows:
        return "No price history yet for this watch."
    return "\n".join(
        f"- {r['fetched_at']}: {r['price']} {r['currency']} ({r['source_url']})"
        for r in rows
    )


@tool
def list_watchlist() -> str:
    """List every watch being tracked, with its id and target price.

    Call this first to get the id of the watch you're checking.
    """
    rows = db.list_watches()
    if not rows:
        return "Watchlist is empty."
    return "\n".join(
        f"id={r['id']} | {r['brand']} {r['model']} ({r['reference_no']}) "
        f"| target={r['target_price']} {r['currency']} | notes={r['notes']}"
        for r in rows
    )


ALL_TOOLS = [search_watch_price, record_price, record_price_confirmed, get_price_history, list_watchlist]
