import statistics
from dataclasses import dataclass
from typing import Literal

# WAIT if the price is above target by no more than this many percent.
WAIT_BAND_PCT = 10.0
# At/below target but falling by at least this many percent versus recent
# checks: hold off, it is still getting cheaper.
STILL_FALLING_PCT = 3.0
# Number of prior per-check prices the trend is measured against.
TREND_WINDOW = 5
# Fewer prior points than this and there is no trend to speak of.
MIN_TREND_POINTS = 2

VerdictLabel = Literal["BUY", "WAIT", "OVERPRICED"]


@dataclass(frozen=True)
class Verdict:
    verdict: VerdictLabel
    pct_vs_target: float
    trend_pct: float | None


def compute_verdict(current: float, target: float, prior_prices: list[float]) -> Verdict:
    """Decide BUY / WAIT / OVERPRICED from numbers alone.

    `prior_prices` are the earlier per-check prices for this watch, newest
    first; the trend compares `current` with their median over the last
    TREND_WINDOW.
    """
    if current <= 0 or target <= 0:
        raise ValueError("current and target prices must be positive")

    pct_vs_target = (current - target) / target * 100
    trend_pct = _trend_pct(current, prior_prices)

    if pct_vs_target <= 0:
        falling = trend_pct is not None and trend_pct <= -STILL_FALLING_PCT
        label: VerdictLabel = "WAIT" if falling else "BUY"
    elif pct_vs_target <= WAIT_BAND_PCT:
        label = "WAIT"
    else:
        label = "OVERPRICED"
    return Verdict(label, pct_vs_target, trend_pct)


def describe_trend(trend_pct: float | None) -> str:
    if trend_pct is None:
        return "not enough history yet"
    if abs(trend_pct) < 0.5:
        return "flat vs recent checks"
    direction = "up" if trend_pct > 0 else "down"
    return f"{direction} {abs(trend_pct):.1f}% vs recent checks"


def _trend_pct(current: float, prior_prices: list[float]) -> float | None:
    recent = prior_prices[:TREND_WINDOW]
    if len(recent) < MIN_TREND_POINTS:
        return None
    baseline = statistics.median(recent)
    return (current - baseline) / baseline * 100
