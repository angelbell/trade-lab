"""portfolio_kama.py -- combined book CAGR/DD with the KAMA gate on the BTC breakout leg.

The lever for CAGR/DD > a single strategy = DIVERSIFICATION (low-correlation legs cut DD).
This combines the validated legs -- gold 1H breakout (daily SMA gate), BTC 4H breakout
(+ KAMA gate, the 2026-06 improvement), BTC 4H EMA pullback (+ weekly cycle-phase gate) -- at a shared 1% risk and
reports standalone vs combined CAGR/DD. Answers "can the book clear ~1.0 CAGR/DD?".
"""
import os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run as run_bo, resample
from ema_pullback import run as run_pb
from research.regime_gate_lab import CFG, at
from research.regime_adaptive import kama

PB = dict(side="long", ema_fast=20, ema_slow=80, slope_k=6, filter="slope", er_period=14,
          swap_pct=0.0, daily_ema=0, exit_sma=0, exit_ma_type="sma", peryear=False,
          no_overlap=True, entry_trigger="close", fill_at_close=True, rr=3.0,
          min_stop_atr=0.5, atr=14, fwd=90, cost=0.001, trend_ma_type="sma", fast_ma_type="ema")


def cagr_dd(t, risk=0.01):
    t = t.sort_values("time")
    eq = (1 + risk * t.R).cumprod()
    dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100
    span = max((t.time.iloc[-1] - t.time.iloc[0]).days / 365.25, 0.5)
    cagr = (eq.iloc[-1] ** (1 / span) - 1) * 100
    return cagr, dd, cagr / max(dd, 1e-9), (eq.iloc[-1] - 1) * 100


def line(name, t):
    c, dd, cdd, ret = cagr_dd(t)
    print(f"  {name:<22} n={len(t):>4}  CAGR={c:+5.1f}%  maxDD={dd:4.1f}%  CAGR/DD={cdd:5.2f}  ret={ret:+5.0f}%")


def kama_gate_btc(t):
    d = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h")
    dc = d["close"].resample("1D").last().dropna()
    km = kama(dc, 14)
    return t[at((km > km.shift(1)).shift(1), t.time)]


def cycle_gate_pull(t, maxext=0.10, cyclelen=30):
    """Weekly cycle-phase gate (ADOPTED 2026-06-21): keep BTC pullback entries only when
    price is <= maxext above the weekly cyclelen-SMA (skip mature-bull chop; early-recovery
    phase is where the pullback edge lives). Causal -- the weekly value is known only at week
    close, so shift 1wk + ffill (cf. bear_short.weekly_bear, pine/btc_4h_ema_pullback.pine
    cycleOK). This is the LEVEL gate ema_pullback's slope --gate-tf cannot express."""
    d = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h")
    w30 = d["close"].resample("1W").last().rolling(cyclelen).mean().shift(1)
    ceil = (1 + maxext) * w30.reindex(d.index, method="ffill")
    return t[at(d["close"] <= ceil, t.time)]


def get_legs():
    """The validated book legs as (time,R) frames. Reused by allocation/vol-target research."""
    gold = run_bo(resample(load_mt5_csv("data/vantage_xauusd_h1.csv"), "1h"),
                  SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                     "daily_sma": 150, "daily_slope_k": 10}))[["time", "R"]]
    btc = run_bo(resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h"),
                 SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0, "fwd": 300}))[["time", "R"]]
    btc_k = kama_gate_btc(btc)
    dbtc = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h")
    pb = run_pb(dbtc, "long", SimpleNamespace(**{**PB, "csv": "x", "tf": "4h"}), 0.0)[["time", "R"]]
    pb = cycle_gate_pull(pb)            # weekly cycle-phase gate (adopted 2026-06-21) -- the deployed leg
    return {"gold_bo": gold, "btc_bo_kama": btc_k, "btc_pull": pb}


def main():
    legs = get_legs()
    gold, btc_k, pb = legs["gold_bo"], legs["btc_bo_kama"], legs["btc_pull"]
    btc = run_bo(resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h"),
                 SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0, "fwd": 300}))[["time", "R"]]

    print("\n=== book CAGR/DD (1% risk/leg, shared account) -- BTC breakout KAMA-gated ===")
    line("GOLD bo (SMA gate)", gold)
    line("BTC bo (no gate)", btc)
    line("BTC bo + KAMA", btc_k)
    line("BTC pullback", pb)
    print("  --- 2-leg pairs (pick the best) ---")
    line("G.bo + B.bo+KAMA", pd.concat([gold, btc_k]))
    line("G.bo + B.pull", pd.concat([gold, pb]))
    line("B.bo+KAMA + B.pull", pd.concat([btc_k, pb]))
    print("  --- 3-leg ---")
    line("3-leg (all)", pd.concat([gold, btc_k, pb]))
    # annual-R correlation (diversification check)
    gy = gold.assign(y=gold.time.dt.year).groupby("y").R.sum()
    by = btc_k.assign(y=btc_k.time.dt.year).groupby("y").R.sum()
    py = pb.assign(y=pb.time.dt.year).groupby("y").R.sum()
    al = pd.concat([gy, by, py], axis=1).fillna(0); al.columns = ["g", "b", "p"]
    print(f"\n  annual-R corr  G.bo-B.bo+K {al.g.corr(al.b):+.2f}  G.bo-B.pull {al.g.corr(al.p):+.2f}  "
          f"B.bo+K-B.pull {al.b.corr(al.p):+.2f}")
    # ADOPTED: inverse-vol risk weighting on the best 2-leg (gold 0.79% / BTC 1.21%, budget held
    # at 2%). Lift is genuine risk-balancing (helps IS AND OOS; recent era is gold-led not BTC) =
    # 1.65 -> 1.83 via DD cut. See research/portfolio_alloc.py for the full weighting/vol-tgt study.
    print("  --- 2-leg with ADOPTED inv-vol weights (gold 0.79% / BTC 1.21%) ---")
    line("G.bo*.79 + B.K*1.21", pd.concat([gold.assign(R=gold.R * 0.79),
                                           btc_k.assign(R=btc_k.R * 1.21)]))


if __name__ == "__main__":
    main()
