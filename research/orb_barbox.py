"""orb_barbox.py -- the user's path-2 idea: define the breakout 'box' by BAR-LENGTH (last N bars, a rolling
Donchian) instead of by a CLOCK window (the Asian session 0-7). Everything else kept at the new best config:
SHORT-only, 1H EMA80 trend gate, early-London entry window, ride-to-close. Sweep N (the bar length).

Box = high/low over the last N bars, SHIFTED by 1 (strictly prior bars -> clean, no box-includes-self). Entry
= first bar in the London window whose close breaks the N-bar low (short); SL = N-bar high (+buf) = structural
opposite edge; no TP, force-flat at 20:00; one trade/day. Reuses scalp_lab's validated htf_trend_gate +
backtest (next-bar-open fill, intrabar SL).

Falsifier (up front): a bar-length box PASSES only if at 15M it beats the session-box short (~1.33) AND
approaches the PF~1.5 target AND IS~=VAL AND plateaus across N (not a lone N) AND survives cost. A lone-N
spike or IS>>VAL or sub-session-box => the clock box was already the right frame; bar-length doesn't help.
In-sample/val (sealed TEST spent).
  .venv/bin/python research/orb_barbox.py
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


def barbox_signals(d, N, bo_start=7, bo_end=11, side="short", buf_atr=0.0):
    hi, lo, cl = d["high"].values, d["low"].values, d["close"].values
    atr = sl.ta.atr(d["high"], d["low"], d["close"], 14).values
    minute = (d.index.hour * 60 + d.index.minute).values
    day = d.index.normalize().values
    box_hi = pd.Series(hi).rolling(N).max().shift(1).values     # strictly prior N bars
    box_lo = pd.Series(lo).rolling(N).min().shift(1).values
    bo = (minute >= bo_start * 60) & (minute < bo_end * 60)
    n = len(cl)
    dir_ = np.zeros(n, np.int8); sl_px = np.full(n, np.nan); tp_px = np.full(n, np.nan)
    cur = None; done = False
    for i in range(n):
        if day[i] != cur:
            cur, done = day[i], False
        if done or not bo[i] or np.isnan(box_hi[i]) or np.isnan(atr[i]):
            continue
        buf = buf_atr * atr[i]
        up = cl[i] > box_hi[i] + buf
        dn = cl[i] < box_lo[i] - buf
        if side in ("short", "both") and dn:
            dir_[i] = -1; sl_px[i] = box_hi[i] + buf; done = True
        elif side in ("long", "both") and up:
            dir_[i] = 1; sl_px[i] = box_lo[i] - buf; done = True
    return dir_, sl_px, tp_px


def run(d, N, bo=(7, 11), side="short"):
    p = SimpleNamespace(**CFG)
    dir_, slx, tpx = barbox_signals(d, N, bo[0], bo[1], side)
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
    return dict(n=len(t), pf=t[t.pips > 0].pips.sum() / abs(t[t.pips < 0].pips.sum()),
                win=(t.pips > 0).mean() * 100, net=t.pips.sum())


def line(tag, t):
    m = pf(t)
    if m is None:
        print(f"  {tag:<16} n={len(t)} (too few)"); return
    print(f"  {tag:<16} n={m['n']:>4} win={m['win']:>3.0f}% PF={m['pf']:.2f} net={m['net']:>+7.0f}p")


def main():
    print("=== ORB with a BAR-LENGTH box (rolling N-bar Donchian, shift1) -- SHORT-only + 1H gate + ride ===")
    print("    target: 15M PF~1.5, beat the session-box short (15m~1.33).  bo window 7-11 (early London).\n")
    for tf in ("15min", "30min", "60min"):
        dis, dval = load("is", tf), load("val", tf)
        print(f"#### TF={tf}  (N = box length in bars) ####")
        for N in (4, 6, 8, 12, 16, 24, 32, 48):
            tis, tval = run(dis, N), run(dval, N)
            mi, mv = pf(tis), pf(tval)
            if mi is None or mv is None:
                continue
            flag = " <-" if (mi["pf"] >= 1.45 and mv["pf"] >= 1.45) else ""
            print(f"  N={N:>3}bar  IS PF={mi['pf']:.2f}(n{mi['n']})  VAL PF={mv['pf']:.2f}(n{mv['n']}){flag}")
        print()
    print("  read: a real bar-length edge = a PLATEAU of N where IS~=VAL>~1.45 at 15M. Lone N / IS>>VAL /")
    print("  all<session-box(1.33) => bar-length framing doesn't beat the clock box. Then per-year + cost next.")


if __name__ == "__main__":
    main()
