import pytest

from watchagent.verdict import compute_verdict, describe_trend


@pytest.mark.parametrize("trend,text", [
    (None, "not enough history yet"),
    (0.2, "flat vs recent checks"),
    (-0.4, "flat vs recent checks"),
    (4.26, "up 4.3% vs recent checks"),
    (-12.0, "down 12.0% vs recent checks"),
])
def test_describe_trend(trend, text):
    assert describe_trend(trend) == text


def test_at_target_with_no_history_is_buy():
    v = compute_verdict(10_000, 10_000, [])
    assert v.verdict == "BUY"
    assert v.pct_vs_target == 0
    assert v.trend_pct is None


def test_below_target_and_flat_is_buy():
    assert compute_verdict(9_000, 10_000, [9_000, 9_100, 8_900]).verdict == "BUY"


def test_below_target_but_still_falling_is_wait():
    v = compute_verdict(9_000, 10_000, [10_000, 10_000, 10_000])
    assert v.trend_pct == pytest.approx(-10.0)
    assert v.verdict == "WAIT"


def test_below_target_and_rising_is_buy():
    assert compute_verdict(9_500, 10_000, [8_000, 8_000, 8_000]).verdict == "BUY"


def test_just_above_target_is_wait():
    assert compute_verdict(10_500, 10_000, []).verdict == "WAIT"


def test_wait_band_upper_edge_is_inclusive():
    assert compute_verdict(11_000, 10_000, []).verdict == "WAIT"


def test_well_above_target_is_overpriced():
    v = compute_verdict(11_001, 10_000, [])
    assert v.verdict == "OVERPRICED"
    assert v.pct_vs_target > 10


def test_single_prior_point_gives_no_trend():
    assert compute_verdict(9_000, 10_000, [12_000]).trend_pct is None


def test_trend_uses_only_the_most_recent_window():
    # Newest first: the five recent prices are 100, the older 1000 is ignored.
    v = compute_verdict(100, 100, [100, 100, 100, 100, 100, 1_000])
    assert v.trend_pct == 0


@pytest.mark.parametrize("current,target", [(0, 100), (-5, 100), (100, 0), (100, -1)])
def test_non_positive_prices_are_rejected(current, target):
    with pytest.raises(ValueError):
        compute_verdict(current, target, [])
