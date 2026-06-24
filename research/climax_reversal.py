"""climax_reversal.py -- the SPECIFIC volume-climax + rejection-wick REVERSAL pattern (FX).

Pattern: a bar makes a new L-bar extreme, prints a long REJECTION wick (price pushed to the extreme then
closed back), on HIGH tick-volume (climax) -> fade it (short the up-climax / long the down-climax). Entry
next-bar-open, structural stop beyond the rejected extreme, fixed RR, intrabar SL/TP, no-overlap, cost.

DECISIVE isolation (does VOLUME add, or is the wick the lever?):
  A. wick-only (no volume filter)         B. wick + climax-volume        C. climax-volume only (no wick)
Falsifier (up front): B must BEAT A (volume adds reversal info) AND clear RR breakeven AND meanR>0 AND
IS~OOS. If B~=A -> volume is inert (price-action wick is the lever, user's volume hope dies). If A dead
too -> pattern dead. tick-volume = activity proxy (not order flow). In-sample; live-forward arbitrates.
  .venv/bin/python research/climax_reversal.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from research.volume_reversal_screen import resample, atr

SPLIT = 2018   # USDJPY 2010-2026 -> rough half


def trades(d, mode="B", L=10, wick_k=1.0, vthr=0.8, rr=2.0, buf=0.2, fwd=24, cost=0.0002):
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    a = atr(d).values
    volpr = d["volume"].rolling(100).rank(pct=True).values
    hiL = pd.Series(h).rolling(L).max().values
    loL = pd.Series(l).rolling(L).min().values
    body = np.abs(c - o)
    uwick = h - np.maximum(o, c)        # upper rejection wick
    lwick = np.minimum(o, c) - l        # lower rejection wick
    rows = []; last = -1
    for i in range(L, len(d) - 1):
        if i <= last or not np.isfinite(a[i]) or a[i] <= 0:
            continue
        climax = volpr[i] >= vthr
        # TOP (short): new L-high, long upper wick, closed back below high
        top_wick = (h[i] >= hiL[i]) and (uwick[i] >= wick_k * a[i]) and (c[i] < h[i] - 0.25 * (h[i] - l[i]))
        bot_wick = (l[i] <= loL[i]) and (lwick[i] >= wick_k * a[i]) and (c[i] > l[i] + 0.25 * (h[i] - l[i]))
        if mode == "A":      sig_s, sig_l = top_wick, bot_wick                 # wick only
        elif mode == "B":    sig_s, sig_l = top_wick and climax, bot_wick and climax   # wick + climax
        else:                sig_s = (h[i] >= hiL[i]) and climax; sig_l = (l[i] <= loL[i]) and climax  # vol only
        if not (sig_s or sig_l):
            continue
        isS = sig_s
        e = o[i + 1]
        if isS:
            stop = h[i] + buf * a[i]; risk = stop - e; tgt = e - rr * risk
        else:
            stop = l[i] - buf * a[i]; risk = e - stop; tgt = e + rr * risk
        if risk <= 0:
            continue
        R = None; mfe = mae = 0.0; end = min(i + 1 + fwd, len(d))
        for j in range(i + 1, end):
            mfe = max(mfe, ((e - l[j]) if isS else (h[j] - e)) / risk)
            mae = max(mae, ((h[j] - e) if isS else (e - l[j])) / risk)
            if (h[j] >= stop) if isS else (l[j] <= stop):
                R = -1; break
            if (l[j] <= tgt) if isS else (h[j] >= tgt):
                R = rr; break
        if R is None:
            R = ((e - c[end - 1]) if isS else (c[end - 1] - e)) / risk; j = end - 1
        rows.append((d.index[i], "S" if isS else "L", R - cost * e / risk, mfe, mae)); last = j
    return pd.DataFrame(rows, columns=["time", "side", "R", "mfe", "mae"])


def line(tag, t, rr):
    if len(t) < 10:
        print(f"  {tag:<22} n={len(t)} (too few)"); return
    be = 100 / (1 + rr)
    mm = t.mfe.mean() / max(t.mae.mean(), 1e-9)
    is_ = t[t.time.dt.year < SPLIT].R.mean(); oos = t[t.time.dt.year >= SPLIT].R.mean()
    w, ll = t[t.R > 0].R.sum(), -t[t.R < 0].R.sum()
    print(f"  {tag:<22} n={len(t):>4} win%={(t.R>0).mean()*100:>3.0f}(BE{be:.0f}) meanR={t.R.mean():+5.2f} "
          f"totR={t.R.sum():>+6.1f} PF={w/max(ll,1e-9):4.2f} MFE/MAE={mm:4.2f} | IS={is_:+.2f} OOS={oos:+.2f}")


def main():
    rr = 2.0
    for name, csv, tf in [("USDJPY 1h", "data/vantage_usdjpy_h1.csv", "1h"),
                          ("USDJPY 4h", "data/vantage_usdjpy_h1.csv", "4h"),
                          ("USDX 1h", "data/vantage_usdx.r_h1.csv", "1h")]:
        d = resample(load_mt5_csv(csv), tf)
        print(f"\n== {name}  (fade: new-{10}bar-extreme + rejection wick [+climax vol], RR{rr}) ==")
        line("A wick-only", trades(d, "A", rr=rr), rr)
        line("B wick+climax-vol", trades(d, "B", rr=rr), rr)
        line("C climax-vol-only", trades(d, "C", rr=rr), rr)
    print("\n  verdict: B must beat A (volume adds) AND meanR>0 AND IS~OOS. B~=A => wick is the lever, not vol.")


if __name__ == "__main__":
    main()
