"""squeeze_phase1.py -- Phase 1: regime SUB-GATE on the BTC 4h expansion breakout.

The leg's weakness is SIZE STABILITY (PBO=0.86), not return. A regime gate that supplies trend context
the breakout LACKS could cut the chop-year noise that makes size unstable. Test the proven transferable
gate (daily-KAMA-rising) and a simple HTF-trend-alignment, per-side (long only in up-regime, short only
in down-regime).

Falsifier (up front, STRICT): a gate is adopted ONLY if it (1) raises CAGR/DD, (2) LOWERS PBO (CSCV),
(3) beats the CAGR/DD random-DROP null (>=90%ile, i.e. not just n-trimming), and (4) plateaus across the
gate length. meanR-up with PBO flat/up = NON-improvement (the transfer-test trap). Else keep the clean
ungated leg. In-sample; live-forward arbitrates.
  .venv/bin/python research/squeeze_phase1.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import resample
from research.squeeze_breakout import run, atr
from research.portfolio_kama import cagr_dd
from research.regime_adaptive import kama
from research.overfit_audit import cscv

SPLIT = 2022


def regime_rising(d, kind="kama", n=14):
    """daily up-regime boolean on d's index (causal: yesterday's completed slope, ffilled)."""
    dc = d["close"].resample("1D").last().dropna()
    m = kama(dc, n) if kind == "kama" else dc.rolling(n).mean()
    rising = (pd.Series(m, index=dc.index) > pd.Series(m, index=dc.index).shift(1)).shift(1)
    return rising.reindex(d.index, method="ffill").fillna(False).astype(bool)


def at_times(series_bool, times):
    """look up a bool series (on a tz-aware index) at each trade entry time."""
    tt = pd.DatetimeIndex(times)
    if tt.tz is None and series_bool.index.tz is not None:
        tt = tt.tz_localize(series_bool.index.tz)
    idx = np.clip(series_bool.index.searchsorted(tt, side="right") - 1, 0, len(series_bool) - 1)
    return series_bool.values[idx]


def gated(d, rr, kind, n, q=0.25, don=30, L=120):
    """expansion trades with per-side regime gate: keep long if up-regime, short if down-regime."""
    t = run(d, rr=rr, hi_atr=True, no_overlap=True, sqz=q, don=don, L=L)
    if len(t) == 0:
        return t
    up = regime_rising(d, kind, n)
    a = at_times(up, t.time.values)
    keep = np.where(t.side.values == "L", a, ~a)
    return t[keep]


def cdd(t):
    return cagr_dd(t[["time", "R"]])[2] if len(t) else np.nan


def drop_null(t, keep_n, iters=3000, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([cagr_dd(t.iloc[np.sort(rng.choice(len(t), keep_n, replace=False))][["time", "R"]])[2]
                     for _ in range(iters)])


def pbo_of(d, kind=None, n=14):
    """PBO via CSCV over a (q,don,rr) grid, optionally with the regime gate applied."""
    monthly = {}
    for q in (0.20, 0.25, 0.30):
        for don in (20, 30, 40):
            for rr in (2.0, 3.0, 4.0):
                t = gated(d, rr, kind, n, q=q, don=don) if kind else run(d, rr=rr, hi_atr=True, no_overlap=True, sqz=q, don=don)
                if len(t) < 20:
                    continue
                monthly[(q, don, rr)] = t.set_index("time").R.resample("1ME").sum()
    months = sorted(set().union(*[set(s.index) for s in monthly.values()]))
    M = pd.DataFrame({str(k): s.reindex(months).fillna(0.0) for k, s in monthly.items()}, index=months)
    return cscv(M.values, S=16, max_combos=2000, seed=0)[0], M.shape[1]


def row(tag, t, base):
    is_ = t[t.time.dt.year < SPLIT].R.sum(); oos = t[t.time.dt.year >= SPLIT].R.sum()
    null = drop_null(base, len(t)); pct = (null < cdd(t)).mean() * 100
    print(f"  {tag:<26} n={len(t):>4} meanR={t.R.mean():+5.2f} totR={t.R.sum():>+6.1f} "
          f"IS={is_:+6.1f} OOS={oos:+6.1f} CAGR/DD={cdd(t):5.2f}  vs random-drop {pct:3.0f}%ile")


def main():
    d = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h")
    base = run(d, rr=3, hi_atr=True, no_overlap=True)
    print(f"== BTC 4h expansion breakout -- Phase 1 regime sub-gate (RR3 base) ==")
    print(f"  UNGATED base: n={len(base)} meanR={base.R.mean():+.2f} CAGR/DD={cdd(base):.2f}")

    print("\n  -- 1a. daily-KAMA-rising gate (per-side), plateau over KAMA n --")
    for n in (10, 14, 20):
        row(f"KAMA{n}", gated(d, 3, "kama", n), base)
    print("  -- 1b. daily-SMA-slope alignment (per-side), plateau over SMA n --")
    for n in (20, 50, 100):
        row(f"SMA{n}", gated(d, 3, "sma", n), base)

    print("\n  -- PBO (CSCV): does the gate LOWER it from the ungated 0.86? --")
    p0, c0 = pbo_of(d, None)
    print(f"     UNGATED         PBO={p0:.2f} ({c0} cfgs)")
    pk, ck = pbo_of(d, "kama", 14)
    print(f"     KAMA14-gated    PBO={pk:.2f} ({ck} cfgs)")
    ps, cs = pbo_of(d, "sma", 50)
    print(f"     SMA50-gated     PBO={ps:.2f} ({cs} cfgs)")

    print("\n  verdict: adopt a gate ONLY if CAGR/DD up AND PBO down AND >=90%ile vs random-drop AND plateau.")


if __name__ == "__main__":
    main()
