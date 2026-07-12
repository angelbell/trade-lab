"""pwh_plateau.py -- plateau check for the PWH selector on gold_bo (the passing cell).
Dimensions: level lookback {prior 1W high, 2W, 1M} x {HARD keep / SOFT 0.5}, and soft
weight {0.25, 0.5, 0.75} at 1W. Each cell: totR/DD + equal-keep random null %ile.
A real selector = neighbors agree; a spike = curve-fit. Also NEGATIVE control: a FAKE
level = prior week's high shifted by a random offset (median of |PWH move|) -- if fake
levels also pass, the selector is just 'price is high' beta, not structure.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run as run_bo, resample
from research.regime_gate_lab import CFG

rng = np.random.default_rng(7)


def totdd(R):
    eq = np.cumsum(R)
    return R.sum() / max((np.maximum.accumulate(eq) - eq).max(), 1e-9)


def null_pct(base_R, obs, keep_n=None, half_n=None, n=4000):
    out = []
    N = len(base_R)
    for _ in range(n):
        if half_n is None:
            out.append(totdd(base_R[np.sort(rng.choice(N, keep_n, replace=False))]))
        else:
            w = np.ones(N); w[rng.choice(N, half_n, replace=False)] = 0.5
            out.append(totdd(base_R * w))
    return (np.array(out) < obs).mean() * 100


def main():
    g1 = resample(load_mt5_csv("data/vantage_xauusd_h1.csv"), "1h")
    t = run_bo(g1, SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                      "daily_sma": 150, "daily_slope_k": 10}))
    ix = g1.index.get_indexer(pd.DatetimeIndex(t.time))
    R = t.R.values
    base = totdd(R)
    print(f"gold_bo n={len(t)}  base totR/DD={base:.2f}   (基準: null>=90%ile)")
    print(f"{'level':<14} {'ON%':>4} {'HARD t/DD':>10} {'%ile':>5} {'SOFT0.5':>8} {'%ile':>5}")
    for tag, rule, k in [("prior 1W high", "1W", 1), ("prior 2W high", "1W", 2), ("prior 1M high", "ME", 1)]:
        hi = g1["high"].resample(rule).max()
        lv = (hi.rolling(k).max() if k > 1 else hi).shift(1).reindex(g1.index, method="ffill")
        on = t.e_px.values > lv.values[ix]
        hard = totdd(R[on])
        soft = totdd(np.where(on, 1.0, 0.5) * R)
        ph = null_pct(R, hard, keep_n=int(on.sum()))
        ps = null_pct(R, soft, half_n=int((~on).sum()))
        print(f"{tag:<14} {on.mean()*100:>3.0f}% {hard:>10.2f} {ph:>4.0f}% {soft:>8.2f} {ps:>4.0f}%")
    # soft-weight neighbors at 1W
    lv = g1["high"].resample("1W").max().shift(1).reindex(g1.index, method="ffill")
    on = t.e_px.values > lv.values[ix]
    for w in (0.25, 0.5, 0.75):
        soft = totdd(np.where(on, 1.0, w) * R)
        # null: same count of down-weighted trades at weight w
        out = []
        for _ in range(4000):
            ww = np.ones(len(R)); ww[rng.choice(len(R), int((~on).sum()), replace=False)] = w
            out.append(totdd(R * ww))
        print(f"soft w={w:.2f}   totR/DD={soft:.2f}  null%ile={(np.array(out)<soft).mean()*100:.0f}%")
    # negative control: fake level = PWH * random factor per week (same ON-rate target)
    print("\n偽ライン対照 (PWH×週ごと乱数シフト, 10本):")
    hits = 0
    for s in range(10):
        r2 = np.random.default_rng(100 + s)
        wk = g1["high"].resample("1W").max()
        fake = (wk * (1 + r2.normal(0, 0.02, len(wk)))).shift(1).reindex(g1.index, method="ffill")
        onf = t.e_px.values > fake.values[ix]
        if onf.sum() < 20 or onf.sum() > len(R) - 20:
            continue
        hardf = totdd(R[onf])
        pf_ = null_pct(R, hardf, keep_n=int(onf.sum()), n=1500)
        hits += pf_ >= 90
        print(f"  fake#{s}: ON%={onf.mean()*100:.0f}  totR/DD={hardf:.2f}  %ile={pf_:.0f}")
    print(f"  偽ライン合格数: {hits}/10  (本物だけが通るなら選別は構造、偽も通るなら『高い所』ベータ)")


if __name__ == "__main__":
    main()
