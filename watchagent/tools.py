import re

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from watchagent import db

_tavily = TavilySearch(max_results=5, topic="general")

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


@tool
def search_watch_price(query: str) -> str:
    """Search the web for current luxury watch listings and prices.

    Use this to find what a specific watch (brand, model, reference number)
    is currently selling for on marketplaces, dealer sites, or price-tracking
    sites like Chrono24 or WatchCharts.

    Args:
        query: Search query, e.g. "Rolex Submariner 126610LN price Chrono24"
    """
    results = _tavily.invoke({"query": query})
    items = results.get("results", []) if isinstance(results, dict) else []
    lines = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")
        lines.append(f"- {title} ({url}): {content}")
    return "\n".join(lines) if lines else "No results found."


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


ALL_TOOLS = [search_watch_price, record_price, get_price_history, list_watchlist]
