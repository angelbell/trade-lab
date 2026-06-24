"""regime_kama_validate.py -- final validation battery for BTC breakout + kama_rising gate.

Pre-registered (docs/scalp_research_log.md, 2026-06-17): BTC 4H breakout + daily KAMA(14)-rising
gate (shift1, no-lookahead). PASS = (1) walk-forward: beats always-on CAGR/DD in the OOS of every
split year 2020/2021/2022/2023; (2) per-year: doesn't wreck most years; (3) cost: survives RT
0.002/0.003 still beating always-on. Fail any -> reject as a 2022-split artifact.

  .venv/bin/python research/regime_kama_validate.py
"""
import argparse, os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from research.regime_gate_lab import CFG, at
from research.regime_adaptive import kama

KN = 14   # KAMA length (pre-registered; overridable via --kama-n)


def cagr_dd(t, risk=0.01):
    t = t.sort_values("time")
    if len(t) < 3:
        return None
    eq = (1 + risk * t.R).cumprod()
    dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100
    span = max((t.time.iloc[-1] - t.time.iloc[0]).days / 365.25, 0.5)
    cagr = (eq.iloc[-1] ** (1 / span) - 1) * 100
    return cagr / max(dd, 1e-9), t.R.mean(), len(t)


def gen(csv, tf, rr, fwd, cost, kn):
    d = resample(load_mt5_csv(csv), tf)
    t = run(d, SimpleNamespace(**{**CFG, "csv": csv, "tf": tf, "rr": rr, "fwd": fwd, "cost": cost}))
    t = t.sort_values("time").reset_index(drop=True)
    dc = d["close"].resample("1D").last().dropna()
    km = kama(dc, kn)
    gate = (km > km.shift(1)).shift(1)                 # KAMA rising, no lookahead
    t["g"] = at(gate, t.time)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/vantage_btcusd_h1.csv")
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--fwd", type=int, default=300)
    ap.add_argument("--kama-n", type=int, default=14)
    a = ap.parse_args()
    csv, tf, rr, fwd, kn = a.csv, a.tf, a.rr, a.fwd, a.kama_n
    print(f"\n=== {os.path.basename(csv)} {tf} breakout + KAMA({kn})-rising gate :: VALIDATION BATTERY ===")
    t = gen(csv, tf, rr, fwd, 0.001, kn)
    base, gated = t, t[t.g]

    print("\n  (1) WALK-FORWARD: OOS (year>=split) always-on vs +kama  [CAGR/DD | meanR | n]")
    for split in (2020, 2021, 2022, 2023):
        b = cagr_dd(base[base.time.dt.year >= split])
        g = cagr_dd(gated[gated.time.dt.year >= split])
        if b and g:
            flag = "OK" if g[0] > b[0] else "**FAIL**"
            print(f"    split {split}: always-on {b[0]:5.2f}/{b[1]:+.2f}/n{b[2]:<3}  "
                  f"+kama {g[0]:5.2f}/{g[1]:+.2f}/n{g[2]:<3}  {flag}")

    print("\n  (2) PER-YEAR totR: always-on vs +kama")
    by = base.groupby(base.time.dt.year).R.sum()
    gy = gated.groupby(gated.time.dt.year).R.sum()
    for y in sorted(by.index):
        print(f"    {y}: always-on {by.get(y,0):+5.1f}   +kama {gy.get(y,0):+5.1f}")

    print("\n  (3) COST STRESS: full-sample CAGR/DD  [always-on -> +kama]")
    for c in (0.001, 0.002, 0.003):
        tc = gen(csv, tf, rr, fwd, c, kn)
        b = cagr_dd(tc); g = cagr_dd(tc[tc.g])
        flag = "OK" if g[0] > b[0] else "**FAIL**"
        print(f"    cost {c}: {b[0]:5.2f} -> {g[0]:5.2f}  (meanR {b[1]:+.2f}->{g[1]:+.2f}, n{b[2]}->{g[2]})  {flag}")


if __name__ == "__main__":
    main()
