"""gold_bo_retest.py -- the PRESCRIBED test (CLAUDE.md): bolt a pullback-RETEST onto gold_bo
itself and ask if it beats a RANDOM-EQUAL-DROP of gold_bo's own break trades.

The open lead (real_trendline.py Probe2/2b): on the standalone trendline detector, requiring a
retest lifted CAGR/DD far above a random-drop null (99.4%ile) but meanR only borderline (88.9%),
and fakeout% was ~unchanged -> the gain is DD-smoothing via better entry timing, NOT false-break
removal. BUT the standalone trendline is gold_bo-redundant (corr +0.89), so the only test that
moves the BOOK is bolting the retest onto gold_bo. That's this.

  base   = gold_bo (enter on the confirmed break)         [the adopted leg]
  retest = gold_bo but require a pullback-retest+reclaim of the broken high before entry
  null   = randomly keep len(retest) of base's trades, many times -> where does retest land?

PASS (worth adopting) = retest beats the random-drop null on BOTH CAGR/DD and meanR (>=~90%ile),
AND holds across a window plateau (not a lone window). Else it's a luck-sorter / window-spike -> bury.

  .venv/bin/python research/gold_bo_retest.py
"""
import os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run as run_bo, resample
from research.regime_gate_lab import CFG

RNG = np.random.default_rng(7)
GOLD = "data/vantage_xauusd_h1.csv"
BASE_CFG = {**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500, "daily_sma": 150, "daily_slope_k": 10}


def gold_trades(retest=0, tol=0.10):
    d = resample(load_mt5_csv(GOLD), "1h")
    args = SimpleNamespace(**{**BASE_CFG, "retest": retest, "retest_tol": tol})
    return run_bo(d, args)[["time", "R"]].sort_values("time").reset_index(drop=True)


def cdd(t, risk=0.01):
    t = t.sort_values("time")
    eq = (1 + risk * t.R).cumprod()
    dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100
    span = max((t.time.iloc[-1] - t.time.iloc[0]).days / 365.25, 0.5)
    cagr = (eq.iloc[-1] ** (1 / span) - 1) * 100
    return cagr / max(dd, 1e-9)


def isoos(t):
    mid = t.time.min() + (t.time.max() - t.time.min()) / 2
    return t[t.time < mid].R.mean(), t[t.time >= mid].R.mean()


def describe(name, t):
    is_, oos = isoos(t)
    print(f"  {name:<22} n={len(t):>4}  win={(t.R>0).mean()*100:>3.0f}%  meanR={t.R.mean():+.3f}  "
          f"CAGR/DD={cdd(t):5.2f}  IS={is_:+.2f} OOS={oos:+.2f}")


def null_test(base, rt, nrep=3000):
    k = len(rt)
    act_cdd, act_mr = cdd(rt), rt.R.mean()
    ds_cdd = np.empty(nrep); ds_mr = np.empty(nrep)
    for i in range(nrep):
        samp = base.iloc[np.sort(RNG.choice(len(base), k, replace=False))]
        ds_cdd[i] = cdd(samp); ds_mr[i] = samp.R.mean()
    pc = lambda d, a: (d < a).mean() * 100
    print(f"\n  RANDOM-EQUAL-DROP null (keep {k}/{len(base)} base trades, n={nrep})")
    print(f"    CAGR/DD : retest={act_cdd:5.2f}  null mean={ds_cdd.mean():5.2f} "
          f"[{np.percentile(ds_cdd,5):.2f},{np.percentile(ds_cdd,95):.2f}]  -> {pc(ds_cdd,act_cdd):4.1f}%ile")
    print(f"    meanR   : retest={act_mr:+.3f}  null mean={ds_mr.mean():+.3f} "
          f"[{np.percentile(ds_mr,5):+.3f},{np.percentile(ds_mr,95):+.3f}]  -> {pc(ds_mr,act_mr):4.1f}%ile")
    print("    PASS = BOTH >=~90%ile (real filter); mid-pack => luck-sorter (bury).")


def main():
    print("\n=== gold_bo + RETEST vs random-equal-drop (the prescribed book test) ===")
    base = gold_trades(retest=0)
    describe("gold_bo (break)", base)

    print("\n  -- window plateau (retest_tol=0.10 ATR) --")
    rts = {}
    for w in (5, 10, 20, 40):
        rt = gold_trades(retest=w)
        rts[w] = rt
        describe(f"gold_bo+retest w={w}", rt)

    # decisive null on the middle/representative window
    for w in (10, 20):
        print(f"\n--- null test, window={w} ---")
        null_test(base, rts[w])


if __name__ == "__main__":
    main()
