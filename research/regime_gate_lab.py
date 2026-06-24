"""regime_gate_lab.py -- can a LOOKAHEAD-FREE regime gate close the 0.69->1.54 gap?

regime_ceiling.py showed deploy-timing is the biggest lever: mechanical SMA150+slope gate
= CAGR/DD 0.69, perfect-foresight oracle = 1.54. This tests whether smarter PREDICTIVE
(no-lookahead) daily regime gates beat the mechanical baseline toward the ceiling. Same
fixed gold-1H-breakout entries; only the daily deploy-gate changes. Judged on IS/OOS
(robust = IS~=OOS, both up) + per-year (turn off bad years, keep good) -- NOT on IS alone.

  .venv/bin/python research/regime_gate_lab.py
"""
import os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample

CFG = dict(pattern="B", sl_mode="line", sl_buf=0.25, swing="zigzag", zz_k=2.0, pivot_n=5,
           renko_k=2.0, mom_fast=12, mom_slow=26, trend_ema=80, bo_window=20, tp_mode="rr",
           atr=14, cost=0.001, swap_pct=0.0, peryear=False, start=None, end=None,
           daily_sma=0, daily_slope_k=0, wave="all", rr=3.0, fwd=500, risk=0.01, exit_kama=0)
SPLIT = 2022   # IS < 2022, OOS >= 2022


def er(close, n):
    net = close.diff(n).abs()
    vol = close.diff().abs().rolling(n).sum()
    return (net / vol).where(vol > 0)


def metrics(t, risk=0.01):
    t = t.sort_values("time")
    if len(t) < 3:
        return None
    eq = (1 + risk * t.R).cumprod()
    dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100
    span = max((t.time.iloc[-1] - t.time.iloc[0]).days / 365.25, 0.5)
    cagr = (eq.iloc[-1] ** (1 / span) - 1) * 100
    y = t.time.dt.year
    isr = t.R[y < SPLIT].mean(); oos = t.R[y >= SPLIT].mean()
    posyears = (t.groupby(y).R.sum() > 0).sum(); nyears = y.nunique()
    return dict(n=len(t), totR=t.R.sum(), cagr=cagr, dd=dd, cdd=cagr / max(dd, 1e-9),
                isr=isr, oos=oos, py=f"{posyears}/{nyears}")


def show(name, t):
    m = metrics(t)
    if m is None:
        print(f"  {name:<26} (too few)"); return
    print(f"  {name:<26} n={m['n']:>4} totR={m['totR']:+4.0f} CAGR={m['cagr']:+5.1f}% "
          f"DD={m['dd']:4.1f}% CAGR/DD={m['cdd']:5.2f} | IS={m['isr']:+.2f} OOS={m['oos']:+.2f} | yr+ {m['py']}")


def at(daily_bool, times):
    return daily_bool.reindex(times, method="ffill").fillna(False).values


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/vantage_xauusd_h1.csv")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--rr", type=float, default=3.0)
    ap.add_argument("--fwd", type=int, default=500)
    a = ap.parse_args()
    csv = a.csv
    d = resample(load_mt5_csv(csv), a.tf)
    t = run(d, SimpleNamespace(**{**CFG, "csv": csv, "tf": a.tf, "rr": a.rr, "fwd": a.fwd}))
    t = t.sort_values("time").reset_index(drop=True)

    dc = d["close"].resample("1D").last().dropna()
    s50, s150 = dc.rolling(50).mean(), dc.rolling(150).mean()
    a14 = (dc.diff().abs()).rolling(14).mean()                         # daily ATR proxy
    slope = (s150 - s150.shift(10)) / (10 * a14)                       # ATR-norm slope of SMA150
    e20 = er(dc, 20)

    gates = {
        "(base) always-on":      pd.Series(True, index=dc.index),
        "mech SMA150+slope":     (dc > s150) & (s150 > s150.shift(10)),
        "ER20>=0.3":             e20 >= 0.30,
        "ER20>=0.4":             e20 >= 0.40,
        "stack 50>150 & rising": (s50 > s150) & (s50 > s50.shift(10)),
        "slope steep>=0.05":     slope >= 0.05,
        "slope steep>=0.075":    slope >= 0.075,
        "slope steep>=0.10":     slope >= 0.10,
        "slope steep>=0.125":    slope >= 0.125,
        "slope steep>=0.15":     slope >= 0.15,
        "mech & slope>=0.075":   (dc > s150) & (slope >= 0.075),
        "ORACLE fwd20 (cheat)":  (dc.shift(-20) > dc),
    }
    print("\n=== regime-gate lab :: gold 1H breakout (entries FIXED) | IS<2022 OOS>=2022 ===")
    print(f"  ceiling refs: always-on CAGR/DD~0.35, mech~0.69, oracle~1.54\n")
    for name, g in gates.items():
        gs = g if name.startswith("(base)") else g.shift(1) if "ORACLE" not in name else g
        show(name, t[at(gs, t.time)])


if __name__ == "__main__":
    main()
