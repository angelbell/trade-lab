"""Leakage and look-ahead bias tests.

Three assertions:
1. EMA (adjust=False) has zero future-bar dependency.
2. All WFO windows have strictly non-overlapping IS/OOS timestamps.
3. Signals computed on IS-only data are identical to signals computed
   on IS+OOS data when restricted to IS timestamps.
"""

import numpy as np
import pandas as pd

from src.strategy import compute_signals_williams as compute_signals
from src.wfo import generate_windows

RNG = np.random.default_rng(42)


def _synthetic_ohlcv(n: int, start: str = "2020-01-01") -> tuple[pd.Series, pd.Series, pd.Series]:
    prices = 30_000 + RNG.standard_normal(n).cumsum() * 100
    idx    = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    close  = pd.Series(prices, index=idx, name="close")
    high   = (close * 1.001).rename("high")
    low    = (close * 0.999).rename("low")
    return close, high, low


# ---------------------------------------------------------------------------
# Test 1: EMA (adjust=False) is a pure recursive formula → no future dependency
# ---------------------------------------------------------------------------

def test_ema_no_future_dependency():
    close, _, _ = _synthetic_ohlcv(500)
    cutoff = 300

    for span in (9, 21):
        ema_trunc = close.iloc[:cutoff].ewm(span=span, adjust=False).mean()
        ema_full  = close.ewm(span=span, adjust=False).mean().iloc[:cutoff]
        pd.testing.assert_series_equal(
            ema_trunc, ema_full, check_exact=True,
            obj=f"EMA({span}) truncated vs full",
        )


# ---------------------------------------------------------------------------
# Test 2: IS and OOS index sets are disjoint for every generated window
# ---------------------------------------------------------------------------

def test_wfo_windows_no_index_overlap():
    close, _, _ = _synthetic_ohlcv(n=365 * 4 * 24, start="2020-01-01")
    windows = generate_windows(close.index, is_days=180, oos_days=30, step_days=30)

    assert len(windows) > 0, "No windows generated — check date range"

    for i, (is_idx, oos_idx) in enumerate(windows):
        overlap = is_idx.intersection(oos_idx)
        assert len(overlap) == 0, (
            f"Window {i}: {len(overlap)} overlapping timestamps — data leakage!"
        )

    oos_starts = [oos_idx[0] for _, oos_idx in windows]
    for a, b in zip(oos_starts, oos_starts[1:]):
        assert b > a, "OOS windows must advance in time"


# ---------------------------------------------------------------------------
# Test 3: Signals on IS data == signals on IS+OOS data (restricted to IS)
# ---------------------------------------------------------------------------

def test_signal_not_affected_by_oos_data():
    """Adding OOS bars must not change any IS signal (EMA, RSI, HTF EMA are all causal)."""
    is_bars    = 500
    extra_bars = 100

    close_is, high_is, low_is = _synthetic_ohlcv(is_bars)
    next_start = str(close_is.index[-1] + pd.Timedelta("15min"))
    close_ex, high_ex, low_ex = _synthetic_ohlcv(extra_bars, start=next_start)

    close_full = pd.concat([close_is, close_ex])
    high_full  = pd.concat([high_is,  high_ex])
    low_full   = pd.concat([low_is,   low_ex])

    params = dict(k=0.5, htf_timeframe="1h", htf_ema_period=50)


    le_is, lx_is, se_is, sx_is = compute_signals(close_is,   high_is,   low_is,   **params)
    le_fu, lx_fu, se_fu, sx_fu = compute_signals(close_full, high_full, low_full, **params)

    for name, sig_is, sig_full in [
        ("long_entries",  le_is, le_fu),
        ("long_exits",    lx_is, lx_fu),
        ("short_entries", se_is, se_fu),
        ("short_exits",   sx_is, sx_fu),
    ]:
        pd.testing.assert_series_equal(
            sig_is,
            sig_full.loc[close_is.index],
            check_exact=True,
            obj=f"{name}: IS-only vs IS prefix of IS+OOS",
        )
