"""The price-research agent: builds it and runs one check of one watch."""
import uuid
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    wrap_model_call,
)
from langchain.agents.middleware.model_retry import default_retry_on
from langchain.agents.structured_output import ToolStrategy
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from watchagent import db
from watchagent.config import DAILY_REQUEST_LIMIT, LLM_FALLBACK_MODELS, LLM_MODEL, require_api_keys
from watchagent.tools import ALL_TOOLS, MIN_LISTINGS, WatchContext, finalize
from watchagent.verdict import TREND_WINDOW, Verdict, compute_verdict, describe_trend

MAX_SEARCHES = 6
MAX_PAGE_OPENS = 4
# Kept low: on the free tier every model call comes out of a 50-a-day budget.
MAX_MODEL_CALLS = 10
MODEL_TIMEOUT_SECONDS = 90

SYSTEM_PROMPT = f"""You are a luxury watch price researcher. Your goal: find the
current market price of ONE watch by collecting at least {MIN_LISTINGS} listings
of its exact reference. The buy/wait verdict is computed by code afterwards, so
never give one.

How to work:
- Search (search_watch_price), starting with chrono24, then other sources.
- When a snippet doesn't clearly show the price or reference, or a page lists
  several watches, open it (open_listing) to read its prices.
- Submit what you found with submit_listings. Code checks every listing (the URL
  and snippet must be real, the price must be written in the snippet, the
  reference must match) and tells you what it accepted, what it rejected and
  why, and how many more it needs. Use that feedback: fix rejected listings, or
  search more specifically (e.g. the full reference number) and submit again.
- Copy snippets and URLs exactly; never estimate, average or round a price.
- Market-price estimates (e.g. WatchCharts "Market Price") are not listings:
  don't submit them.
- You have at most {MAX_SEARCHES} searches and {MAX_PAGE_OPENS} page opens.

Before you finish, if you learned something that would make the next check of
this watch faster (a query that worked, a kind of listing that was a trap), save
it with save_note. Then reply with 1-3 sentences: which listings the price came
from and any caveat. Don't state a verdict or repeat the target."""


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
    listings: int
    sources: list[str]
    verdict: Verdict
    explanation: str

    @property
    def confidence(self) -> str:
        return f"{self.listings} listing{'s' if self.listings != 1 else ''}"

    @property
    def confidence_label(self) -> str:
        label = f"{self.confidence} ({' + '.join(self.sources)})"
        return label if self.listings >= MIN_LISTINGS else f"{label}, lower confidence"

    def message(self) -> str:
        return (
            f"{self.verdict.verdict}: {self.watch}\n"
            f"Current: {self.price:,.2f} {self.currency} "
            f"(target {self.target_price:,.2f}, {self.verdict.pct_vs_target:+.1f}% vs target)\n"
            f"Trend: {describe_trend(self.verdict.trend_pct)}\n"
            f"{self.confidence_label}: {self.explanation}"
        )


def make_model(name: str, **settings) -> ChatGoogleGenerativeAI:
    """A Gemini chat model. Extra `settings` (e.g. thinking_budget=0) go to the client.

    Retries are left to ModelRetryMiddleware: the client's own defaults (no
    timeout, 6 retries) once hung a CI run for 40 minutes.
    """
    return ChatGoogleGenerativeAI(model=name, timeout=MODEL_TIMEOUT_SECONDS, max_retries=0, **settings)


class DailyBudgetExceeded(RuntimeError):
    """Today's share of model requests is used up. Nothing was sent."""


def request_budget(daily_limit: int):
    """Middleware that counts every model request (retries and fallbacks too) and refuses past `daily_limit`.

    Put it last in the middleware list so it sits closest to the model and sees each real request.
    """
    @wrap_model_call
    def count_and_limit(request, handler):
        # ponytail: check-then-count, so parallel checks can overshoot by a request or two.
        if db.llm_requests_today() >= daily_limit:
            raise DailyBudgetExceeded(f"today's {daily_limit} model requests are used up")
        db.count_llm_request()
        return handler(request)
    return count_and_limit


@wrap_model_call
def research_before_answering(request, handler):
    """Until a price is recorded, the model must call a research tool.

    Without this, a weak model can call the final-answer tool on its first turn
    and end the check without searching at all.
    """
    if request.runtime.context.recorded is None:
        request = request.override(tool_choice="required", response_format=None)
    return handler(request)


def retry_on(exc: Exception) -> bool:
    """Retry failed model calls, except when the daily budget says stop."""
    return not isinstance(exc, DailyBudgetExceeded) and default_retry_on(exc)


def build_agent(daily_limit: int = DAILY_REQUEST_LIMIT):
    require_api_keys()
    return create_agent(
        model=make_model(LLM_MODEL),
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        context_schema=WatchContext,
        response_format=ToolStrategy(PriceExplanation),
        middleware=[
            ModelRetryMiddleware(max_retries=3, initial_delay=10, max_delay=60, on_failure="error",
                                 retry_on=retry_on),
            # Thinking off: with it on, gemini-3.5-flash took 24-77s per call instead of ~2s.
            ModelFallbackMiddleware(*[make_model(m, thinking_budget=0) for m in LLM_FALLBACK_MODELS]),
            ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
            ToolCallLimitMiddleware(tool_name="search_watch_price", run_limit=MAX_SEARCHES),
            ToolCallLimitMiddleware(tool_name="open_listing", run_limit=MAX_PAGE_OPENS),
            research_before_answering,
            request_budget(daily_limit),
        ],
    )


def _prompt(watch_row, label: str) -> str:
    prompt = f"Find the current price of: {label}. Prices are in {watch_row['currency']}."
    notes = db.recent_notes(watch_row["id"])
    if notes:
        prompt += "\n\nYour notes from earlier checks of this watch:\n" + "\n".join(f"- {n}" for n in notes)
    return prompt


def check_watch(agent, watch_row) -> CheckResult:
    """Research one watch's price, compute its verdict in code, and store the check."""
    label = f"{watch_row['brand']} {watch_row['model']}"
    if watch_row["reference_no"]:
        label += f" ({watch_row['reference_no']})"
    ctx = WatchContext(
        watch_id=watch_row["id"], currency=watch_row["currency"], run_id=uuid.uuid4().hex[:12],
        reference_no=watch_row["reference_no"] or "",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": _prompt(watch_row, label)}]},
        # Each model call passes through several middleware steps; the model-call
        # limit, not this, is what ends a long run.
        config={"recursion_limit": 10 * MAX_MODEL_CALLS},
        context=ctx,
    )

    # If the agent ran out of budget before reaching MIN_LISTINGS, whatever it
    # did get accepted still becomes a (lower-confidence) price.
    recorded = finalize(ctx)
    if recorded is None:
        raise RuntimeError("the agent finished without any listing that passed the checks "
                           "(it may have skipped the search: see its tool calls)")

    structured = result.get("structured_response")
    explanation = structured.explanation if structured else "Research stopped at its budget."
    prior_prices = db.prior_check_prices(watch_row["id"], TREND_WINDOW)
    verdict = compute_verdict(recorded.price, watch_row["target_price"], prior_prices)
    check = CheckResult(
        watch=label, price=recorded.price, currency=recorded.currency,
        target_price=watch_row["target_price"], listings=recorded.listings,
        sources=recorded.sources, verdict=verdict, explanation=explanation,
    )
    db.record_check(
        watch_row["id"], ctx.run_id, check.price, check.currency, check.confidence,
        " + ".join(check.sources), verdict.verdict, verdict.pct_vs_target,
        verdict.trend_pct, explanation,
    )
    return check
