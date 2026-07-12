"""radar_leg_gate.py -- does the trend-strength RADAR gate add anything to the BTC 15m LEG?

Prior (verified_findings §15): radar MTF gate separates GENERIC barrier-free 15m/5m forward
returns (BTC 2h&4h up&>=5 = +0.537, every year). But it was never tested on the mechanized
leg. Prediction (CLAUDE.md law): a gate helps only when it supplies context the leg LACKS --
the leg already has trend-ema80 + KAMA gate; the radar's core is ER = KAMA's own input, so
expect redundancy. Method: regenerate the audited BTC 15m cell UNGATED (gate_kama=0), then
post-hoc label each entry bar by each causal gate (HTF confirmed bar shift(1) -> ffill):
  kama1d  = daily KAMA(14) rising          (the originally validated gate)
  kama4h  = 4h KAMA(14) rising             (C1 "attack" gate)
  kama4h&1d = hard AND                     (C1 "defense")
  radar4h = 4h stack>0 & strength>=5       (gold default per user)
  radar2h4h = 2h AND 4h up&>=5             (BTC default per user)
  radar & kama4h                           (does radar add on top?)
Strength = the Pine's 5-comp (ER+ADX+align+slope+ATRexp)/5*10. Note: tile arrow uses
stack==3; the TESTED gate is stack>0 (looser) -- we match the tested gate.
Net $15 rt post-hoc via risk column. Regression target (ungated cell has no canon number;
kama1d cell must reproduce N=614 / meanR+0.322 / PF1.37).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample, kama_adaptive

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}
ARGS = dict(pattern="B", sl_mode="line", sl_buf=0.25, swing="zigzag", zz_k=2.0,
            pivot_n=5, renko_k=2.0, mom_fast=12, mom_slow=26, trend_ema=80,
            bo_window=20, tp_mode="rr", rr=4.0, atr=14, cost=0.0, swap_pct=0.0,
            fwd=500, peryear=False, start=None, end=None, daily_sma=0,
            daily_slope_k=0, risk=0.01, gate_kama=0, pullback_frac=0.3, tf="15min", csv="")
RT = 15.0


def radar_up(d15, rule):
    d = d15.resample(rule).agg(AGG).dropna()
    c = d["close"]
    atr = ta.atr(d["high"], d["low"], d["close"], 14)
    er = (c.diff(20).abs() / c.diff().abs().rolling(20).sum()).clip(0, 1)
    adxN = ((ta.adx(d["high"], d["low"], d["close"], 14)["ADX_14"] - 15) / 25).clip(0, 1)
    emaF = c.ewm(span=20, adjust=False).mean()
    emaS = c.ewm(span=50, adjust=False).mean()
    stack = np.sign(c - emaF) + np.sign(emaF - emaS) + np.sign(emaS - emaS.shift(10))
    align = stack.abs() / 3.0
    slopeN = ((emaS - emaS.shift(10)).abs() / (atr * 1.5)).clip(0, 1)
    atrexpN = ((atr / atr.rolling(100).mean() - 0.8) / 0.7).clip(0, 1)
    s = 10.0 * (er.fillna(0) + adxN.fillna(0) + align.fillna(0)
                + slopeN.fillna(0) + atrexpN.fillna(0)) / 5.0
    up = ((stack > 0) & (s >= 5.0)).shift(1)          # confirmed HTF bar only
    return up.reindex(d15.index, method="ffill").fillna(False).values.astype(bool)


def kama_up(d15, rule):
    dc = d15["close"].resample(rule).last().dropna()
    km = kama_adaptive(dc, 14)
    up = (km > km.shift(1)).shift(1)
    return up.reindex(d15.index, method="ffill").fillna(False).values.astype(bool)


def cell(tag, Rn, tm, years_span):
    if len(Rn) == 0:
        print(f"  {tag:>16}: n=0")
        return
    pf = Rn[Rn > 0].sum() / abs(Rn[Rn <= 0].sum()) if (Rn <= 0).any() else float("inf")
    eq = np.cumsum(Rn)
    dd = (np.maximum.accumulate(eq) - eq).max()
    print(f"  {tag:>16}: N={len(Rn):4d} N/yr={len(Rn)/years_span:5.1f}  win={(Rn>0).mean()*100:4.1f}%"
          f"  meanR={Rn.mean():+.3f}  PF={pf:.2f}  totR/yr={Rn.sum()/years_span:+6.1f}  maxDD={dd:5.1f}R")


def peryear(tag, Rn, yr):
    ys = sorted(np.unique(yr))
    print(f"    {tag} per-year totR: " + "  ".join(f"{y}:{Rn[yr==y].sum():+.0f}" for y in ys))


def main():
    d = load_mt5_csv("data/vantage_btcusd_m15.csv")
    d = d[d.index >= "2018-10-01"]
    d15 = resample(d, "15min")
    span = (d15.index[-1] - d15.index[0]).days / 365.25

    gates = {
        "kama1d":  kama_up(d15, "1D"),
        "kama4h":  kama_up(d15, "240min"),
        "radar4h": radar_up(d15, "240min"),
        "radar2h": radar_up(d15, "120min"),
    }
    # gate-series overlap on all bars (context, not trade-weighted)
    g4, r24 = gates["kama4h"], gates["radar2h"] & gates["radar4h"]
    both = (g4 & r24).sum()
    print(f"bar-level overlap: kama4h ON {g4.mean()*100:.0f}%  radar2h&4h ON {r24.mean()*100:.0f}%"
          f"  jaccard={both/ (g4 | r24).sum():.2f}  P(radar|kama4h)={both/g4.sum():.2f}")

    t = run(d15, SimpleNamespace(**ARGS))
    Rn = t["R"].values - RT / t["risk"].values
    yr = t["time"].dt.year.values
    pos = d15.index.get_indexer(t["time"])
    lab = {k: v[pos] for k, v in gates.items()}
    lab["radar2h4h"] = lab["radar2h"] & lab["radar4h"]

    print(f"\nBTC 15m leg (RR4 frac0.3, net $15), {span:.1f}yr:")
    cell("UNGATED base", Rn, t, span)
    cell("kama1d (canon)", Rn[lab["kama1d"]], t, span)
    cell("kama4h (C1)", Rn[lab["kama4h"]], t, span)
    m_and = lab["kama4h"] & lab["kama1d"]
    cell("kama4h&1d (C1)", Rn[m_and], t, span)
    cell("radar4h", Rn[lab["radar4h"]], t, span)
    cell("radar2h&4h", Rn[lab["radar2h4h"]], t, span)
    m_rk = lab["radar2h4h"] & lab["kama4h"]
    cell("radar & kama4h", Rn[m_rk], t, span)
    m_ronly = lab["radar2h4h"] & ~lab["kama4h"]
    cell("radar \\ kama4h", Rn[m_ronly], t, span)
    m_konly = lab["kama4h"] & ~lab["radar2h4h"]
    cell("kama4h \\ radar", Rn[m_konly], t, span)

    print()
    peryear("kama4h    ", Rn[lab["kama4h"]], yr[lab["kama4h"]])
    peryear("radar2h&4h", Rn[lab["radar2h4h"]], yr[lab["radar2h4h"]])
    peryear("radar&kama", Rn[m_rk], yr[m_rk])


if __name__ == "__main__":
    main()
