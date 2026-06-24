"""bear_short.py -- the bear-cycle-gated SHORT leg, via the user's flow (downside version).

Shorts always DIED on gold/BTC because of the +drift handicap. Untested: shorts with that handicap
REMOVED -- i.e. only in a CONFIRMED weekly downtrend (close < weekly-30SMA AND weekly-30SMA falling).
The prize is NOT standalone PF; it's NEGATIVE correlation to the all-long book (a leg that earns when
gold/BTC sink = the one diversification angle the TSMOM null showed the book is missing).

User's flow, downside version:
  1. pick high-prob breakdowns: confirmed CLOSE below prior-N-low, in a downtrend (close<SMA80).
  2. strength: MFE/MAE on the short (exit-agnostic screen).
  3. how far it falls without retracing: capturable run.
  4. set TP: RR sweep, keep the low-P/high-R tail.
Compare always-on (historical=dead) vs bear-gated. Decisive: meanR>0 with bear/chop years GREEN
(esp 2022) AND negative annual corr to the long book. Else = same dead short side (low-corr noise).

Causal: breakdown=confirmed close; next-bar-open fill; intrabar SL/TP (stop checked first); weekly
gate shifted 1wk + ffill (value known only at week close). In-sample; live-forward arbitrates.

  .venv/bin/python research/bear_short.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import resample
from research.portfolio_kama import get_legs, cagr_dd

SPLIT = 2022
N = 20          # breakdown window (inherited book default)
TREND = 80      # trend SMA (inherited)
ATR_N = 14
MIN_STOP_ATR = 0.5
FWD = 300
COST = 0.001    # round-trip


def atr(d, n=ATR_N):
    h, l, c = d["high"], d["low"], d["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def weekly_bear(d, cyclelen=30):
    """confirmed weekly downtrend: close<weekly-30SMA AND weekly-30SMA falling (causal)."""
    w = d["close"].resample("1W").last().dropna()
    w30 = w.rolling(cyclelen).mean()
    bear = ((w < w30) & (w30 < w30.shift(1))).shift(1)        # known only at week close -> shift
    return bear.reindex(d.index, method="ffill").fillna(False).astype(bool)


def short_trades(d, rr, gated, cyclelen=30):
    d = d.copy()
    d["sma"] = d["close"].rolling(TREND).mean()
    d["lo"] = d["low"].rolling(N).min().shift(1)              # prior-N low (no lookahead)
    d["hi"] = d["high"].rolling(N).max().shift(1)             # prior-N high = structural stop
    d["a"] = atr(d)
    bear = weekly_bear(d, cyclelen) if gated else pd.Series(True, index=d.index)
    sig = (d["close"] < d["lo"]) & (d["close"] < d["sma"]) & bear     # confirmed-close breakdown, downtrend
    H, L, O, C = d["high"].values, d["low"].values, d["open"].values, d["close"].values
    hi, a = d["hi"].values, d["a"].values
    idx = np.where(sig.values)[0]
    rows = []
    last_exit = -1
    for i in idx:
        if i <= last_exit or i + 1 >= len(d):                # no-overlap
            continue
        e = O[i + 1]                                          # next-bar-open fill
        stop = max(hi[i], e + MIN_STOP_ATR * a[i])           # structural stop above, ATR floor
        risk = stop - e
        if risk <= 0:
            continue
        tgt = e - rr * risk
        R = mfe = mae = None
        end = min(i + 1 + FWD, len(d))
        for j in range(i + 1, end):
            mfe = max(mfe or 0, (e - L[j]) / risk)           # favorable = price falls
            mae = max(mae or 0, (H[j] - e) / risk)
            if H[j] >= stop:                                 # stop first (conservative)
                R = -1 - COST; break
            if L[j] <= tgt:
                R = rr - COST; break
        if R is None:
            R = (e - C[end - 1]) / risk - COST               # MTM at window end
        rows.append((d.index[i + 1], R, mfe or 0, mae or 0))
        last_exit = end - 1 if R is None else j
    return pd.DataFrame(rows, columns=["time", "R", "mfe", "mae"])


def summ(tag, t):
    if len(t) < 8:
        print(f"  {tag:<30} n={len(t)} (too few)"); return None
    c, dd, cdd, ret = cagr_dd(t)
    isr = t[t.time.dt.year < SPLIT]; oos = t[t.time.dt.year >= SPLIT]
    mi = isr.R.mean() if len(isr) else np.nan
    mo = oos.R.mean() if len(oos) else np.nan
    mm = t.mfe.mean() / max(t.mae.mean(), 1e-9)
    print(f"  {tag:<30} n={len(t):>4} win%={(t.R>0).mean()*100:>3.0f} meanR={t.R.mean():+5.2f} "
          f"totR={t.R.sum():>+6.0f} | IS={mi:+.2f} OOS={mo:+.2f} | MFE/MAE={mm:4.2f} CAGR/DD={cdd:5.2f}")
    return t


def peryear(t):
    g = t.groupby(t.time.dt.year).R.agg(["count", "sum", "mean"])
    s = "   ".join(f"{y}:{r['sum']:+.0f}({int(r['count'])})" for y, r in g.iterrows())
    print(f"      per-year totR(n): {s}")


def annual_corr(short, leg):
    a = short.groupby(short.time.dt.year).R.sum()
    b = leg.groupby(leg.time.dt.year).R.sum()
    x = pd.concat([a, b], axis=1).dropna()
    return x.iloc[:, 0].corr(x.iloc[:, 1]) if len(x) >= 4 else np.nan


def main():
    legs = get_legs()
    insts = [("BTC 4h", resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h")),
             ("GOLD 4h", resample(load_mt5_csv("data/vantage_xauusd_h1.csv"), "4h"))]

    print("\n== STEP 2-3 (screen): MFE/MAE of the breakdown, always-on vs bear-gated (RR-agnostic) ==")
    for name, d in insts:
        for gated in (False, True):
            t = short_trades(d, 2.0, gated)
            mm = t.mfe.mean() / max(t.mae.mean(), 1e-9) if len(t) else np.nan
            print(f"  {name} {'BEAR-GATED' if gated else 'always-on ':<10} n={len(t):>4} "
                  f"MFE/MAE={mm:4.2f} (>1.2 worth deeper; <1.0 dead)")

    print("\n== STEP 4: realized SHORT leg, RR sweep -- always-on (historical=DEAD) vs BEAR-GATED ==")
    best = {}
    for name, d in insts:
        print(f"\n  --- {name} ---")
        for gated in (False, True):
            tag0 = "BEAR-GATED" if gated else "always-on"
            for rr in (1.5, 2.0, 3.0):
                t = short_trades(d, rr, gated)
                r = summ(f"{tag0} RR{rr}", t)
                if gated and rr == 2.0 and r is not None:
                    peryear(r); best[name] = r

    print("\n== DECISIVE: annual-R corr of BEAR-GATED short (RR2) vs the LONG book legs ==")
    print("   (NEGATIVE corr = real hedge that earns when the long book sinks; ~0/positive = no hedge value)")
    for name, t in best.items():
        cg = annual_corr(t, legs["gold_bo"]); cb = annual_corr(t, legs["btc_bo_kama"]); cp = annual_corr(t, legs["btc_pull"])
        print(f"  {name:<10} corr gold_bo={cg:+.2f}  btc_bo_kama={cb:+.2f}  btc_pull={cp:+.2f}")
    print("\n  verdict: meanR>0 with bear/chop years (esp 2022) GREEN *and* negative corr to book => real hedge;")
    print("           else = same dead short side, low-corr-because-noise (like TSMOM's BTC short side).")


if __name__ == "__main__":
    main()
