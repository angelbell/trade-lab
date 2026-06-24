"""pullback_conviction.py -- does CONVICTION-weighted sizing beat flat sizing?

The exit audit found a real, OOS-consistent QUALITY gradient on the EMA-pullback:
strong-trend + shallow/quick pullbacks continue cleaner (higher meanR). But FILTERING
on it loses money (the weak pullbacks are still net-positive). The legitimate use of a
positive quality gradient is SIZING: keep taking every trade (preserve total participation)
but risk MORE on high-conviction entries and LESS on low. This tests whether that improves
CAGR/DD vs flat sizing -- the only thing that matters for a portfolio leg.

Conviction score = mean of IS-standardised (shallow depth, steep slope, quick reclaim);
the 'strong reclaim' predictor is DROPPED (it flipped sign in VAL). Size multiplier
mult = clip(1 + k*z, lo, hi), then NORMALISED so the IS mean multiplier = 1.0 -> average
risk is held equal to flat (fair comparison: this redistributes risk, it doesn't add leverage).

  .venv/bin/python research/pullback_conviction.py --csv data/vantage_btcusd_h1.csv --tf 4h
"""
import argparse, os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from ema_pullback import resample
from research.pullback_exit_audit import audit

CFG = dict(ema_fast=20, ema_slow=80, slope_k=6, thr=0.0, rr=3.0, trend_ma_type="sma",
           fast_ma_type="ema", min_stop_atr=0.5, atr=14, fwd=90, cost=0.001)


def metrics(t, mult, risk=0.01):
    """equity, CAGR, maxDD, total-R (size-weighted) for a trade set + size multipliers."""
    rmult = risk * mult * t["R"].values
    eq = np.cumprod(1.0 + rmult)
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / peak).max() * 100
    span = max((t["time"].iloc[-1] - t["time"].iloc[0]).days / 365.25, 0.5)
    cagr = (eq[-1] ** (1 / span) - 1) * 100
    totR = (mult * t["R"].values).sum()
    return dict(cagr=cagr, dd=dd, totR=totR, ret=(eq[-1] - 1) * 100,
                cdd=cagr / max(dd, 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--split", default="2022-01-01")
    ap.add_argument("--lo", type=float, default=0.4)
    ap.add_argument("--hi", type=float, default=1.6)
    a = ap.parse_args()
    d = resample(load_mt5_csv(a.csv), a.tf)
    t = audit(d, SimpleNamespace(**CFG)).sort_values("time").reset_index(drop=True)

    split_ts = pd.Timestamp(a.split)
    if t.time.dt.tz is not None:
        split_ts = split_ts.tz_localize(t.time.dt.tz)
    is_m = t.time < split_ts

    # standardise each predictor on IS only (no lookahead), favourable = HIGH z
    z = pd.DataFrame(index=t.index)
    for col, sign in (("depth", -1), ("slopev", +1), ("pbbars", -1)):
        mu, sd = t.loc[is_m, col].mean(), t.loc[is_m, col].std()
        z[col] = sign * (t[col] - mu) / (sd if sd > 0 else 1.0)
    conv = z.mean(axis=1).values                       # combined conviction z-score

    print(f"\n=== conviction-weighted sizing  {os.path.basename(a.csv)} {a.tf}  "
          f"IS<{a.split} (n={is_m.sum()}) | VAL>= (n={(~is_m).sum()}) ===")
    print(f"  metric: CAGR% / maxDD% / CAGR-DD ratio / total-R   (avg risk held = flat via IS-norm)")
    for k in (0.0, 0.3, 0.5, 0.8):
        mult = np.clip(1 + k * conv, a.lo, a.hi)
        mult = mult / mult[is_m].mean()                # normalise: IS mean size = 1.0 (leverage-match)
        tag = "FLAT " if k == 0 else f"k={k:<3}"
        for name, mask in (("IS ", is_m.values), ("VAL", (~is_m).values)):
            sub = t[mask].reset_index(drop=True)
            m = metrics(sub, mult[mask])
            print(f"  [{tag}] {name}: CAGR={m['cagr']:+6.1f}%  maxDD={m['dd']:4.1f}%  "
                  f"CAGR/DD={m['cdd']:5.2f}  totR={m['totR']:+6.1f}")
        print()

    # ---- does the high-conviction TAIL behave differently? (bucket by conviction z) ----
    t = t.assign(conv=conv)
    edges = [-np.inf, -1, -0.5, 0, 0.5, 1, 1.5, 2, np.inf]
    labels = ["<-1", "-1..-.5", "-.5..0", "0..0.5", "0.5..1", "1..1.5", "1.5..2", ">=2"]
    print("  ==== outcome by conviction band  (z-score; >=2 = extreme high conviction) ====")
    print(f"  {'band':<9}| {'IS  n  win  meanR':<22}| VAL  n  win  meanR")
    bi = np.digitize(conv, edges[1:-1])
    for b, lab in enumerate(labels):
        sub = t[bi == b]
        si, sv = sub[sub.time < split_ts], sub[sub.time >= split_ts]
        def cell(x):
            return f"n={len(x):>3} {('win'+str(round((x.R>0).mean()*100)) if len(x) else '   -')[:5]:>5} {x.R.mean():+5.2f}" if len(x) else f"n=  0   -      -  "
        print(f"  {lab:<9}| {cell(si):<22}| {cell(sv)}")


if __name__ == "__main__":
    main()
