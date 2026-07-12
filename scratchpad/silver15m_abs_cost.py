"""silver15m_abs_cost.py -- silver 15m/30m Pattern-B breakout, ABSOLUTE-$ round-trip cost.

Fractional --cost misprices silver: the real spread is ~constant in $ (rt ~$0.02-0.05/oz)
while price ran $14->$58, so a constant fraction overtaxes 2018-20 and undertaxes 2024-26.
Reuses breakout_wave.run() verbatim (cost=0) and applies R_net = R - spread/risk post-hoc,
exactly the pullback_cost_abs.py method validated on gold.

Cells: tf {15min,30min} x frac {0=market,0.2,0.25,0.3,0.38} x spread {0.02,0.03,0.05}.
No gate (both daily gates measurably hurt silver base). Reports PF/N/N-yr/win/meanR/
IS-OOS/maxDD(R)/ret-DD(raw R)/green plus cost-risk med/p90 per cell.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample

BASE = dict(pattern="B", sl_mode="line", sl_buf=0.25, swing="zigzag", zz_k=2.0,
            pivot_n=5, renko_k=2.0, mom_fast=12, mom_slow=26, trend_ema=80,
            bo_window=20, tp_mode="rr", rr=4.0, atr=14, cost=0.0, swap_pct=0.0,
            fwd=500, peryear=False, start=None, end=None, daily_sma=0,
            daily_slope_k=0, risk=0.01)

SPREADS = [0.02, 0.03, 0.05]
FRACS = [0.0, 0.2, 0.25, 0.3, 0.38]

def stats(R, ts):
    n = len(R)
    span = (ts.iloc[-1] - ts.iloc[0]).days / 365.25
    pf = R[R > 0].sum() / abs(R[R <= 0].sum()) if (R <= 0).any() else np.inf
    eq = np.cumsum(R); dd = (np.maximum.accumulate(eq) - eq).max()
    half = ts.iloc[n // 2]
    is_, oos = R[ts < half].mean(), R[ts >= half].mean()
    y = ts.dt.year.values
    ys = sorted(set(y)); green = sum(1 for yy in ys if R[y == yy].sum() > 0)
    return dict(n=n, nyr=n / span, win=(R > 0).mean() * 100, pf=pf, mean=R.mean(),
                is_=is_, oos=oos, dd=dd, rdd=R.sum() / dd if dd > 0 else np.inf,
                green=f"{green}/{len(ys)}")

def main():
    raw = load_mt5_csv("data/vantage_xagusd_m15.csv")
    for tf in ["15min", "30min"]:
        d = resample(raw, tf)
        for frac in FRACS:
            args = SimpleNamespace(**BASE, tf=tf, csv="", pullback_frac=frac)
            t = run(d, args)
            if t is None:
                continue
            lbl = "market " if frac == 0 else f"frac{frac:<4}"
            cr = np.array([s / t["risk"].values for s in SPREADS])  # cost/risk per spread
            print(f"\n[{tf} RR4 {lbl}] gross meanR={t['R'].mean():+.3f}  "
                  f"cost/risk@$0.03 med={np.median(cr[1])*100:.1f}% p90={np.percentile(cr[1],90)*100:.1f}%")
            for s in SPREADS:
                Rn = t["R"].values - s / t["risk"].values
                st = stats(pd.Series(Rn), t["time"])
                print(f"  rt=${s:.2f}: n={st['n']:4d} N/yr={st['nyr']:5.1f} win={st['win']:4.1f}% "
                      f"PF={st['pf']:5.2f} meanR={st['mean']:+.3f} IS/OOS={st['is_']:+.2f}/{st['oos']:+.2f} "
                      f"maxDD={st['dd']:6.1f}R ret/DD={st['rdd']:5.2f} green={st['green']}")
            if frac in (0.0, 0.3):
                Rn = t["R"].values - 0.03 / t["risk"].values
                y = t["time"].dt.year.values
                per = "  ".join(f"{yy}:{Rn[y==yy].sum():+.0f}(n{(y==yy).sum()})"
                                for yy in sorted(set(y)))
                print(f"  per-year @$0.03: {per}")

if __name__ == "__main__":
    main()
