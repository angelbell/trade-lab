"""Canonical argument templates for the engine's run() entry points.

BASE = the 15m-breakout family template every adopted 15m leg starts from
(gold15m / btc15m_L / btc15m_S override rr / gates / pullback_frac on top).
Copied verbatim from experiments/radar_gate_race.py (the frozen original);
invariants/book_tieback.py asserts the two stay equal.
"""

BASE = dict(pattern="B", sl_mode="line", sl_buf=0.25, swing="zigzag", zz_k=2.0,
            pivot_n=5, renko_k=2.0, mom_fast=12, mom_slow=26, trend_ema=80,
            bo_window=20, tp_mode="rr", rr=4.0, atr=14, cost=0.0, swap_pct=0.0,
            fwd=500, peryear=False, start=None, end=None, daily_sma=0,
            daily_slope_k=0, risk=0.01, gate_kama=0, tf="15min", csv="")
