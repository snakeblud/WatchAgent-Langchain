import sys
import time
from typing import Literal

from langchain.agents import create_agent
from langchain_core.exceptions import ModelRateLimitError
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from watchagent.config import GEMINI_MODEL, require_api_keys
from watchagent.tools import ALL_TOOLS

MAX_RATE_LIMIT_RETRIES = 4
RATE_LIMIT_BASE_DELAY_SECONDS = 20

SYSTEM_PROMPT = """You are a luxury watch price-tracking analyst.

For the single watch described in the user message:
1. Call search_watch_price at least twice with different `source` values
   (e.g. "chrono24" then "watchcharts") to get independent listings. Only
   fall back to a single "general" search if both specific sources come up
   empty.
2. From each source's results, find the listing that states this exact
   watch's price and note its exact source_url and snippet text (never
   estimate, average, or round a price yourself).
3. Call record_price_confirmed with one PriceObservation per source. This
   only succeeds if 2+ sources agree within ~5% — it records the consensus
   price for you. If it's rejected (sources disagree, or you only found one
   usable source), instead call record_price with your single best grounded
   source. Whichever path you used, START your final reasoning with either
   "CONFIRMED (<source names, e.g. chrono24 + watchcharts>): " or
   "SINGLE-SOURCE (<source name>, lower confidence): " so the confidence
   level AND which source(s) it came from are always explicit.
4. Call get_price_history to see past observations for this watch.
5. Compare the current price against the target price AND the historical/
   market trend, then give a final verdict.

Verdict meanings:
- BUY: current price is at/below target AND not trending worse.
- WAIT: price is close but not good enough yet, or trending down further.
- OVERPRICED: price is well above target and/or historical norm.

Be concise and concrete in your reasoning (cite the actual numbers).
"""


class WatchVerdict(BaseModel):
    watch: str = Field(description="Brand, model, and reference of the watch")
    current_price: float = Field(description="Current observed price")
    currency: str = Field(description="Currency code of current_price")
    target_price: float = Field(description="The user's target price")
    trend: str = Field(description="Short description of the price trend, e.g. 'flat', 'down 5% over 3 checks'")
    verdict: Literal["BUY", "WAIT", "OVERPRICED"]
    reasoning: str = Field(description="1-3 sentence justification citing numbers")


def build_agent():
    require_api_keys()
    model = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
    return create_agent(
        model=model,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=WatchVerdict,
    )


def check_watch(agent, watch_row) -> WatchVerdict:
    prompt = (
        f"Check watch id={watch_row['id']}: {watch_row['brand']} {watch_row['model']} "
        f"(ref {watch_row['reference_no']}). Target price: "
        f"{watch_row['target_price']} {watch_row['currency']}."
    )
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"recursion_limit": 30},
            )
            return result["structured_response"]
        except ModelRateLimitError:
            if attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            delay = RATE_LIMIT_BASE_DELAY_SECONDS * (2 ** attempt)
            print(f"  Rate limited, retrying {watch_row['brand']} {watch_row['model']} "
                  f"in {delay}s (attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES})...",
                  file=sys.stderr)
            time.sleep(delay)
