"""orb_exit_lab.py -- path-2 final shot: hold the ENTRY fixed (15M coil/Donchian-N box, SHORT, London 7-11,
1H gate) and vary the EXIT, to see if a different exit lifts 15M short PF from ~1.4 toward ~1.5. ride-to-close
might be the cap; test fixed-RR TP and ATR-trailing against it (intraday, EOD-flat at 20:00).

Prior (be honest): TP/scale-out HURT the structural-stop session-box short (asym-exit test) and structural
legs generally (scaleout_transfer) -- they cap the deep runners gold shorts depend on. Trailing might do
better than a fixed TP. Falsifier: an exit PASSES only if it beats ride-to-close on PF at 15M, IS AND VAL,
with a plateau across its param. A lone param or IS>>VAL = no exit edge; ride-to-close stays. In-sample/val.
  .venv/bin/python research/orb_exit_lab.py
"""
import os, sys, warnings
from types import SimpleNamespace
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
import research.scalp_lab as sl
from research.orb_coilbox import coil_signals

PIP = sl.PIP


def simulate(d, dir_, sl_px, mode="ride", param=0.0, force_exit_h=20, cost=1.4):
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    atr = sl.ta.atr(d["high"], d["low"], d["close"], 14).values
    minute = (d.index.hour * 60 + d.index.minute).values
    day = d.index.normalize().values
    n = len(c)
    rows = []
    for i in range(n - 1):
        if dir_[i] == 0:
            continue
        e = o[i + 1]; ei = i + 1; stop = sl_px[i]; pos = int(dir_[i])
        if not np.isfinite(stop) or not np.isfinite(atr[i]):
            continue
        risk = abs(e - stop)
        if risk <= 0:
            continue
        tp = e - param * risk if pos > 0 and False else (e + param * risk if pos > 0 else e - param * risk)
        trail = stop
        best = e
        out = None
        for j in range(ei, n):
            if minute[j] >= force_exit_h * 60 and day[j] == day[ei]:
                out = o[j]; break
            if day[j] != day[ei] and minute[j] >= force_exit_h * 60:
                out = o[j]; break
            # intrabar structural stop
            if pos < 0 and h[j] >= stop:
                out = stop; break
            if pos > 0 and l[j] <= stop:
                out = stop; break
            if mode == "rr":
                if pos < 0 and l[j] <= tp:
                    out = tp; break
                if pos > 0 and h[j] >= tp:
                    out = tp; break
            elif mode == "trail":
                if pos < 0:
                    best = min(best, l[j]); newt = best + param * atr[j]
                    trail = min(trail, newt)
                    if h[j] >= trail:
                        out = trail; break
                else:
                    best = max(best, h[j]); newt = best - param * atr[j]
                    trail = max(trail, newt)
                    if l[j] <= trail:
                        out = trail; break
            # end-of-data
            if j == n - 1:
                out = c[j]
        if out is None:
            out = c[n - 1]
        g = (out - e) if pos > 0 else (e - out)
        rows.append((d.index[ei], pos, g / PIP - cost, j - ei))
    return pd.DataFrame(rows, columns=["t_in", "dir", "pips", "bars"])


def load(split, tf):
    s, e = sl.SPLITS[split]
    d = load_mt5_csv("data/vantage_xauusd_m5.csv").loc[s:e]
    return d.resample(tf, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def entries(d, N=24):
    p = SimpleNamespace(force_exit_h=20, cost=1.4, stop_slip=0.0, htf_tf="1h", htf_ema=80, htf_slope_k=0)
    dir_, slx, tpx = coil_signals(d, N, bo=(7, 11), side="short")
    dir_, slx, tpx = sl.htf_trend_gate(d, dir_, slx, tpx, p)
    return dir_, slx


def pf(t):
    if len(t) < 15:
        return None
    return dict(n=len(t), pf=t[t.pips > 0].pips.sum() / max(abs(t[t.pips < 0].pips.sum()), 1e-9),
                win=(t.pips > 0).mean() * 100, net=t.pips.sum())


def line(tag, ti, tv):
    mi, mv = pf(ti), pf(tv)
    if mi is None or mv is None:
        print(f"  {tag:<18} (too few)"); return
    print(f"  {tag:<18} IS PF={mi['pf']:.2f} win{mi['win']:.0f}% net{mi['net']:+.0f}(n{mi['n']})  "
          f"VAL PF={mv['pf']:.2f} win{mv['win']:.0f}% net{mv['net']:+.0f}(n{mv['n']})")


def main():
    tf = "15min"
    dis, dval = load("is", tf), load("val", tf)
    Ed = {sp: entries(d) for sp, d in (("is", dis), ("val", dval))}
    print(f"=== EXIT lab: 15M coil-box N24 SHORT + 1H gate, entry FIXED, vary EXIT (target: beat ride ~1.4) ===\n")
    print(" baseline ride-to-close:")
    line("ride", simulate(dis, *Ed["is"], "ride"), simulate(dval, *Ed["val"], "ride"))
    print("\n fixed-RR TP (caps the move):")
    for rr in (1.0, 1.5, 2.0, 2.5, 3.0):
        line(f"RR{rr}", simulate(dis, *Ed["is"], "rr", rr), simulate(dval, *Ed["val"], "rr", rr))
    print("\n ATR-trailing stop (k*ATR from best):")
    for k in (1.0, 1.5, 2.0, 3.0, 4.0):
        line(f"trail {k}ATR", simulate(dis, *Ed["is"], "trail", k), simulate(dval, *Ed["val"], "trail", k))
    print("\n  read: an exit beats ride only if PF up on IS AND VAL with a param plateau. Else ride-to-close")
    print("  stays (gold shorts run deep -> capping/trailing tends to cut the runners = the asym-exit lesson).")


if __name__ == "__main__":
    main()
