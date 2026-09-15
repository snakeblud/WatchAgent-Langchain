from typing import Literal

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from watchagent.config import GEMINI_MODEL, require_api_keys
from watchagent.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are a luxury watch price-tracking analyst.

For the single watch described in the user message:
1. Call search_watch_price to find current listings/prices for it.
2. Call record_price to log the current price you found (do this even if the
   price is bad news).
3. Call get_price_history to see past observations for this watch.
4. Compare the current price against the target price AND the historical/
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
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"recursion_limit": 15},
    )
    return result["structured_response"]
