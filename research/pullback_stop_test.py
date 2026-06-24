"""pullback_stop_test.py -- does a TIGHTER (ATR) stop beat the structural dip-low stop?

Current stop = the pullback's dip LOW (structure), with a 0.5*ATR floor. The user asks:
the fake-pullback loser re-breaks the EMA and bleeds to that dip low -- can we narrow
the stop to cut it? In R-terms a stop is always -1R, so narrowing can't make the loss
'smaller'; it changes the WIN RATE (tighter = stopped more often; but target = rr*risk
moves closer too). The danger: 82% of trades re-break the EMA at least once, and many
recover -- a stop near the EMA converts those recoverers into losses.

Modes: dip (current) | atrM (stop = entry - M*ATR, a fixed ATR stop). Same entries,
same RR, only the stop placement changes. Reports IS/VAL win/meanR/totR.

  .venv/bin/python research/pullback_stop_test.py --csv data/vantage_btcusd_h1.csv --tf 4h
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from ema_pullback import resample

CFG = dict(ema_fast=20, ema_slow=80, slope_k=6, thr=0.0, rr=3.0, atr=14, fwd=90, cost=0.001)


def detect_entries(d, p):
    ef = d["close"].ewm(span=p["ema_fast"], adjust=False).mean().values
    es = d["close"].rolling(p["ema_slow"]).mean().values                  # SMA trend (validated)
    a = ta.atr(d["high"], d["low"], d["close"], length=p["atr"]).values
    c, h, l = d["close"].values, d["high"].values, d["low"].values
    K = p["slope_k"]
    slope = np.full(len(c), np.nan)
    slope[K:] = (es[K:] - es[:-K]) / (K * np.where(a[K:] > 0, a[K:], np.nan))
    entries, state, ext = [], 0, None
    for i in range(K + 1, len(c) - 1):
        if np.isnan(slope[i]) or np.isnan(a[i]) or a[i] <= 0:
            continue
        if not (slope[i] >= p["thr"]):
            state, ext = 0, None; continue
        if c[i] < ef[i]:
            state = 1; ext = l[i] if ext is None else min(ext, l[i])
        elif state == 1 and h[i] >= ef[i]:
            entries.append((i, c[i], ext)); state, ext = 0, None
    return entries, ef, a, c, h, l


def evaluate(d, entries, ef, a, c, h, l, mode, p):
    N = p["fwd"]
    rows, busy = [], -1
    for (i, e, dip) in entries:
        if i <= busy:
            continue
        if mode == "dip":
            stop = min(dip, l[i])
            if e - stop < 0.5 * a[i]:
                stop = e - 0.5 * a[i]
        else:                                       # atrM
            m = float(mode[3:])
            stop = e - m * a[i]
        risk = e - stop
        if risk <= 0 or i + 1 >= len(c):
            continue
        tgt, R, exit_j = e + p["rr"] * risk, None, min(i + N, len(c) - 1)
        for j in range(i + 1, min(i + 1 + N, len(c))):
            if l[j] <= stop:
                R, exit_j = -1.0, j; break
            if h[j] >= tgt:
                R, exit_j = float(p["rr"]), j; break
        if R is None:
            R = (c[exit_j] - e) / risk
        R -= p["cost"] / risk * e
        rows.append((d.index[i], R)); busy = exit_j
    return pd.DataFrame(rows, columns=["time", "R"])


def stat(t):
    if len(t) == 0:
        return "n=0"
    return f"n={len(t):>3} win={(t.R>0).mean()*100:>3.0f}% meanR={t.R.mean():+5.2f} totR={t.R.sum():+6.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--split", default="2022-01-01")
    a = ap.parse_args()
    d = resample(load_mt5_csv(a.csv), a.tf)
    entries, ef, atr, c, h, l = detect_entries(d, CFG)
    split_ts = pd.Timestamp(a.split)
    print(f"\n=== EMA pullback STOP test  {os.path.basename(a.csv)} {a.tf}  RR={CFG['rr']}  long ===")
    print(f"  stop placement (entries & RR fixed; only the stop moves)")
    for mode in ("dip", "atr3.0", "atr2.0", "atr1.5", "atr1.0", "atr0.75"):
        t = evaluate(d, entries, ef, atr, c, h, l, mode, CFG)
        if len(t) and t.time.dt.tz is not None and split_ts.tz is None:
            sts = split_ts.tz_localize(t.time.dt.tz)
        else:
            sts = split_ts
        si, sv = t[t.time < sts], t[t.time >= sts]
        lbl = "dip(now)" if mode == "dip" else mode
        print(f"  [{lbl:<8}] ALL {stat(t)}")
        print(f"  {'':<11} IS  {stat(si)}")
        print(f"  {'':<11} VAL {stat(sv)}")
    print("\n  (1R is always -1R by construction; a tighter stop can only change WIN RATE)")


if __name__ == "__main__":
    main()
