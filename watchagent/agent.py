import sys
import time
import uuid
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.exceptions import ModelRateLimitError
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from watchagent import db
from watchagent.config import GEMINI_MODEL, require_api_keys
from watchagent.tools import ALL_TOOLS, WatchContext
from watchagent.verdict import TREND_WINDOW, Verdict, compute_verdict, describe_trend

MAX_RATE_LIMIT_RETRIES = 4
RATE_LIMIT_BASE_DELAY_SECONDS = 20

SYSTEM_PROMPT = """You are a luxury watch price-tracking analyst. Your job is to
find and record the current market price of ONE watch. The buy/wait verdict is
computed by code from the recorded price, so never give one.

For the watch described in the user message:
1. Call search_watch_price at least twice with different `source` values
   (e.g. "chrono24" then "watchcharts") to get independent listings. Only
   fall back to a single "general" search if both specific sources come up
   empty.
2. From each source's results, find the listing that states this exact
   watch's price (the right reference, not another model, strap or accessory)
   and note its exact source_url and the exact snippet text (never estimate,
   average, or round a price yourself).
3. Call record_price_confirmed with one PriceObservation per listing. This
   only succeeds if 2+ different sites agree within ~5% and records the
   consensus price for you. If it's rejected (sites disagree, or you only
   found one usable listing), call record_price with your single best listing.
   Tool results say why a call was rejected: fix it and retry. Only one price
   can be recorded per check.
4. Once a price is recorded, reply with a short explanation (1-3 sentences):
   name the site(s) and the specific listing, e.g. "chrono24 pre-owned,
   unworn full set at $16,350", plus any caveat such as condition or a
   listing that looked unreliable. Don't state a verdict or repeat the target.
"""


class PriceExplanation(BaseModel):
    explanation: str = Field(
        description="1-3 sentences on the recorded price: which listings it came from and any caveat. No verdict."
    )


@dataclass(frozen=True)
class CheckResult:
    watch: str
    price: float
    currency: str
    target_price: float
    confidence: str  # "CONFIRMED" or "SINGLE-SOURCE"
    sources: list[str]
    verdict: Verdict
    explanation: str

    @property
    def confidence_label(self) -> str:
        names = " + ".join(self.sources)
        if self.confidence == "CONFIRMED":
            return f"CONFIRMED ({names})"
        return f"SINGLE-SOURCE ({names}, lower confidence)"

    def message(self) -> str:
        return (
            f"{self.verdict.verdict}: {self.watch}\n"
            f"Current: {self.price:,.2f} {self.currency} "
            f"(target {self.target_price:,.2f}, {self.verdict.pct_vs_target:+.1f}% vs target)\n"
            f"Trend: {describe_trend(self.verdict.trend_pct)}\n"
            f"{self.confidence_label}: {self.explanation}"
        )


def build_agent():
    require_api_keys()
    model = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
    return create_agent(
        model=model,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        context_schema=WatchContext,
        response_format=PriceExplanation,
    )


def check_watch(agent, watch_row) -> CheckResult:
    label = f"{watch_row['brand']} {watch_row['model']}"
    if watch_row["reference_no"]:
        label += f" ({watch_row['reference_no']})"
    prompt = (
        f"Find the current price of: {label}. "
        f"Prices are in {watch_row['currency']}."
    )
    ctx = WatchContext(
        watch_id=watch_row["id"], currency=watch_row["currency"], run_id=uuid.uuid4().hex[:12],
    )
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"recursion_limit": 30},
                context=ctx,
            )
            break
        except ModelRateLimitError:
            if attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            delay = RATE_LIMIT_BASE_DELAY_SECONDS * (2 ** attempt)
            print(f"  Rate limited, retrying {watch_row['brand']} {watch_row['model']} "
                  f"in {delay}s (attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES})...",
                  file=sys.stderr)
            time.sleep(delay)

    recorded = ctx.recorded
    if recorded is None:
        raise RuntimeError("the agent finished without recording a valid price")

    explanation = result["structured_response"].explanation
    prior_prices = db.prior_check_prices(watch_row["id"], TREND_WINDOW)
    verdict = compute_verdict(recorded.price, watch_row["target_price"], prior_prices)
    db.record_check(
        watch_row["id"], ctx.run_id, recorded.price, recorded.currency, recorded.confidence,
        " + ".join(recorded.sources), verdict.verdict, verdict.pct_vs_target,
        verdict.trend_pct, explanation,
    )
    return CheckResult(
        watch=label, price=recorded.price, currency=recorded.currency,
        target_price=watch_row["target_price"], confidence=recorded.confidence,
        sources=recorded.sources, verdict=verdict, explanation=explanation,
    )
