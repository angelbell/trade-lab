"""orb_coilbox.py -- the user's path-2b box: a COIL box = N bars that made NO new high AND no new low
(consolidation/contraction), then ENTER WITH the breakout direction (continuation, both sides).

Box = prior N-bar Donchian (high/low over [i-N, i-1], shift1 = strictly past). 'N bars without a new
high/low' is exactly: price stayed inside that box -> the first close beyond it is the first new extreme in
N+1 bars = the breakout. Enter WITH the break (close>box_hi -> long, close<box_lo -> short). Optional
contraction filter (box width <= kw*ATR = a genuine tight coil, not a wide drift). Exit kept = H17
(ride-to-close 20:00, structural stop = opposite box edge, one trade/day) for an apples-to-apples compare.

Falsifier: PASS only if at 15M it beats the session-box SHORT (~1.33) AND nears PF~1.5 AND IS~=VAL AND
plateaus across N (not a lone N). both-sides continuation includes longs (known weak beta) -> also report
short-break-only. Gate on/off + London-window vs 24h tested (the 1H gate was H17's 'body'). In-sample/val.
  .venv/bin/python research/orb_coilbox.py
"""
import os, sys, warnings
from types import SimpleNamespace
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
import research.scalp_lab as sl

CFG = dict(force_exit_h=20, cost=1.4, stop_slip=0.0, htf_tf="1h", htf_ema=80, htf_slope_k=0)


def coil_signals(d, N, bo=(7, 11), side="both", kw=0.0):
    hi, lo, cl = d["high"].values, d["low"].values, d["close"].values
    atr = sl.ta.atr(d["high"], d["low"], d["close"], 14).values
    minute = (d.index.hour * 60 + d.index.minute).values
    day = d.index.normalize().values
    box_hi = pd.Series(hi).rolling(N).max().shift(1).values
    box_lo = pd.Series(lo).rolling(N).min().shift(1).values
    inwin = (minute >= bo[0] * 60) & (minute < bo[1] * 60)
    n = len(cl)
    dir_ = np.zeros(n, np.int8); sl_px = np.full(n, np.nan); tp_px = np.full(n, np.nan)
    cur = None; done = False
    for i in range(n):
        if day[i] != cur:
            cur, done = day[i], False
        if done or not inwin[i] or np.isnan(box_hi[i]) or np.isnan(atr[i]):
            continue
        if kw > 0 and (box_hi[i] - box_lo[i]) > kw * atr[i]:   # coil must be TIGHT (contraction)
            continue
        up = cl[i] > box_hi[i]
        dn = cl[i] < box_lo[i]
        if side in ("both", "long") and up:
            dir_[i] = 1; sl_px[i] = box_lo[i]; done = True
        elif side in ("both", "short") and dn:
            dir_[i] = -1; sl_px[i] = box_hi[i]; done = True
    return dir_, sl_px, tp_px


def run(d, N, bo=(7, 11), side="both", kw=0.0, gate=True):
    p = SimpleNamespace(**CFG)
    if not gate:
        p.htf_tf = ""
    dir_, slx, tpx = coil_signals(d, N, bo, side, kw)
    dir_, slx, tpx = sl.htf_trend_gate(d, dir_, slx, tpx, p)
    return sl.backtest(d, dir_, slx, tpx, p)


def load(split, tf):
    s, e = sl.SPLITS[split]
    d = load_mt5_csv("data/vantage_xauusd_m5.csv").loc[s:e]
    return d.resample(tf, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def pf(t):
    if len(t) < 15:
        return None
    return dict(n=len(t), pf=t[t.pips > 0].pips.sum() / max(abs(t[t.pips < 0].pips.sum()), 1e-9),
                win=(t.pips > 0).mean() * 100, net=t.pips.sum())


def sweep(dis, dval, label, **kw):
    print(f"#### {label} ####")
    for N in (4, 6, 8, 12, 16, 24, 32):
        mi, mv = pf(run(dis, N, **kw)), pf(run(dval, N, **kw))
        if mi is None or mv is None:
            continue
        flag = " <-" if (mi["pf"] >= 1.45 and mv["pf"] >= 1.45) else ""
        print(f"  N={N:>3}  IS PF={mi['pf']:.2f}(n{mi['n']})  VAL PF={mv['pf']:.2f}(n{mv['n']}){flag}")
    print()


def main():
    print("=== COIL box (N-bar no-new-high/low) -> enter WITH the break (continuation). gold M5->TF ===")
    print("    compare target: session-box SHORT 15m~1.33 / 1H~1.46.\n")
    for tf in ("15min", "60min"):
        dis, dval = load("is", tf), load("val", tf)
        print(f"=========== TF={tf} ===========")
        sweep(dis, dval, f"{tf} London7-11 +1Hgate  BOTH sides", bo=(7, 11), side="both", gate=True)
        sweep(dis, dval, f"{tf} London7-11 +1Hgate  SHORT-break only", bo=(7, 11), side="short", gate=True)
        sweep(dis, dval, f"{tf} London7-11 +1Hgate  BOTH + coil kw1.5", bo=(7, 11), side="both", kw=1.5, gate=True)
        sweep(dis, dval, f"{tf} 24h(any time) +1Hgate  BOTH sides", bo=(0, 24), side="both", gate=True)
    print("  read: a real coil-continuation edge = plateau of N with IS~=VAL>~1.45 at 15M beating 1.33.")


if __name__ == "__main__":
    main()
