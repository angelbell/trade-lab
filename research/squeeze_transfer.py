"""squeeze_transfer.py -- does the EXPANSION (high-ATR) filter TRANSFER to the breakout family?

The expansion filter lifted a generic BTC 4h Donchian breakout (meanR +0.23 -> +0.43). KAMA transferred
across breakout legs; the cycle gate did NOT. Decisive question: does "only break when ATR is in the top
quantile (vol already expanding)" help the REAL book breakouts (gold_bo, btc_bo_kama)?

Falsifier (up front): TRANSFER PASS needs, on the REAL legs, (1) CAGR/DD up AND (2) beating the CAGR/DD
random-equal-DROP null (>=~90%ile) -- not just meanR (meanR-only = n-trimming luck-sorter, the law that
killed stop-width/ER), and (3) a quantile plateau. Else NON-transfer (gold_bo already embodies breakout
structure; expansion may be BTC-vol-specific). In-sample; live-forward arbitrates.
  .venv/bin/python research/squeeze_transfer.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import resample
from research.squeeze_breakout import run, stats, atr
from research.portfolio_kama import get_legs, cagr_dd

SPLIT = 2022


def atr_rank_at(price_df, times, L=120, tf=None):
    """ATR(14) percentile-rank (0..1) on price_df's TF, sampled at each trade entry time (causal: prior bar)."""
    a = atr(price_df, 14)
    rank = a.rolling(L).rank(pct=True).shift(1)         # prior-bar rank = causal
    # map each entry time to the rank of the bar at-or-before it
    r = rank.reindex(price_df.index)
    tt = pd.DatetimeIndex(times)
    if tt.tz is None and r.index.tz is not None:
        tt = tt.tz_localize(r.index.tz)
    elif tt.tz is not None and r.index.tz is None:
        tt = tt.tz_localize(None)
    idx = r.index.searchsorted(tt, side="right") - 1
    idx = np.clip(idx, 0, len(r) - 1)
    return r.values[idx]


def drop_null(t, keep_n, iters=3000, seed=0):
    """CAGR/DD distribution of randomly KEEPING keep_n trades (the n-trimming null)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(iters):
        sub = t.iloc[np.sort(rng.choice(len(t), keep_n, replace=False))]
        out.append(cagr_dd(sub)[2])
    return np.array(out)


def line(tag, t):
    s = stats(t) if "side" in t.columns else None
    if s is None:
        c, dd, cdd, _ = cagr_dd(t)
        print(f"  {tag:<30} n={len(t):>4} meanR={t.R.mean():+5.2f} totR={t.R.sum():>+6.1f} "
              f"IS={t[t.time.dt.year<SPLIT].R.sum():+6.1f} OOS={t[t.time.dt.year>=SPLIT].R.sum():+6.1f} CAGR/DD={cdd:5.2f}")
    else:
        print(f"  {tag:<30} n={s['n']:>4} meanR={s['meanR']:+5.2f} totR={s['totR']:>+6.1f} "
              f"IS={s['IS']:+6.1f} OOS={s['OOS']:+6.1f} PF={s['PF']:4.2f}")


def main():
    gld4 = resample(load_mt5_csv("data/vantage_xauusd_h1.csv"), "4h")
    btc4 = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h")

    print("== TEST A: generic Donchian breakout, expansion gate OFF vs ON (does gold behave like BTC?) ==")
    for name, d in [("GOLD 4h", gld4), ("BTC 4h", btc4)]:
        line(f"{name} gate OFF (plain BO)", run(d, rr=3, sqz=1.0))            # all bars
        line(f"{name} gate ON  (top25% ATR)", run(d, rr=3, hi_atr=True, sqz=0.25))

    print("\n== TEST B: bolt expansion onto the REAL book legs (post-filter by entry ATR-rank) ==")
    legs = get_legs()
    gld1 = load_mt5_csv("data/vantage_xauusd_h1.csv")                          # gold_bo runs on 1h
    leg_src = {"gold_bo": (gld1, 120), "btc_bo_kama": (btc4, 120)}
    for leg in ("gold_bo", "btc_bo_kama"):
        t = legs[leg].copy()
        src, L = leg_src[leg]
        t["arank"] = atr_rank_at(src, t.time.values, L=L)
        c0, dd0, cdd0, _ = cagr_dd(t)
        print(f"\n  --- {leg}: full  n={len(t)} meanR={t.R.mean():+.2f} CAGR/DD={cdd0:.2f} ---")
        print(f"  {'quantile keep':<18}{'n':>5}{'meanR':>8}{'CAGR/DD':>9}{'vs random-drop %ile':>22}")
        for q in (0.50, 0.65, 0.75, 0.85):                                    # keep top (1-q) ATR-rank
            kept = t[t.arank >= q]
            if len(kept) < 12:
                print(f"  top{int((1-q)*100):>2}% (rank>={q})  n={len(kept)} (too few)"); continue
            c, dd, cdd, _ = cagr_dd(kept)
            null = drop_null(t, len(kept))
            pct = (null < cdd).mean() * 100
            flag = "PASS" if pct >= 90 else ("weak" if pct >= 75 else "FAIL(=n-trim)")
            print(f"  top{int((1-q)*100):>2}% (rank>={q:.2f})  {len(kept):>4}{kept.R.mean():>+8.2f}"
                  f"{cdd:>9.2f}{pct:>16.0f}%ile {flag}")
    print("\n  verdict: PASS only if CAGR/DD rises AND beats the random-drop null (>=90%ile) AND plateaus.")
    print("           meanR-up but %ile<90 = the leg just got n-trimmed = NON-transfer (luck-sorter).")


if __name__ == "__main__":
    main()
