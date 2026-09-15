from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from watchagent import db

_tavily = TavilySearch(max_results=5, topic="general")


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
    items = results.get("results", []) if isinstance(results, dict) else results
    lines = []
    for item in items:
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")
        lines.append(f"- {title} ({url}): {content}")
    return "\n".join(lines) if lines else "No results found."


@tool
def record_price(watch_id: int, price: float, currency: str, source_url: str = "",
                  snippet: str = "") -> str:
    """Log an observed price for a watch to the price history database.

    Always call this after finding a current price, even if the price is not
    a good deal, so future checks can see the trend.

    Args:
        watch_id: The id of the watch (from list_watchlist)
        price: The observed price as a number, e.g. 15200
        currency: Currency code, e.g. "USD"
        source_url: URL where this price was observed
        snippet: Short text snippet supporting the price (listing title/condition/etc.)
    """
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
