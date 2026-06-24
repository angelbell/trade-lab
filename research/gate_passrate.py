"""gate_passrate.py -- year-by-year PASS RATE of candidate trend/regime gates on GOLD.

Goal: the current gate (close>SMA150) is a LEVEL gate -> it let 2021 through (gold
chopped ABOVE a flat SMA = passed yet lost). We want a gate that turns OFF the
chop years (2018/2021/2022/2023, esp 2021 -18R ungated) while KEEPING the trend
years ON (2019/2020/2024/2025/2026). So: compute each candidate as a daily boolean
(longs allowed), report % of days ON per year, and eyeball it against the known
ungated per-year P&L.

No trading here -- pure regime characterization. Daily series from the H1 csv.

  .venv/bin/python research/gate_passrate.py --csv data/vantage_xauusd_h1.csv
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv

# ungated Pattern-B per-year totR (from breakout_wave.py, for reference)
REF = {2018:-3, 2019:+15, 2020:+14, 2021:-18, 2022:-2, 2023:-2, 2024:+20, 2025:+33, 2026:+14}


def choppiness(h, l, c, n=14):
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    hh = h.rolling(n).max(); ll = l.rolling(n).min()
    return 100 * np.log10(tr.rolling(n).sum() / (hh - ll)) / np.log10(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sma", type=int, default=150)
    ap.add_argument("--slope-k", type=int, default=20)
    ap.add_argument("--chop-th", type=float, default=50.0, help="trending if CI < this")
    ap.add_argument("--donch-n", type=int, default=50)
    ap.add_argument("--adx-th", type=float, default=20.0)
    a = ap.parse_args()

    h1 = load_mt5_csv(a.csv)
    d = pd.DataFrame({"open": h1["open"].resample("1D").first(),
                      "high": h1["high"].resample("1D").max(),
                      "low":  h1["low"].resample("1D").min(),
                      "close":h1["close"].resample("1D").last()}).dropna()
    d = d.loc["2018-01-01":]
    H, L, C = d["high"], d["low"], d["close"]

    sma   = C.rolling(a.sma).mean()
    level = C > sma                                   # current gate
    slope = sma > sma.shift(a.slope_k)                # SMA rising
    ci    = choppiness(H, L, C, 14)
    chop  = ci < a.chop_th                            # trending (not choppy)
    dmid  = (H.rolling(a.donch_n).max() + L.rolling(a.donch_n).min()) / 2
    donch = C > dmid                                  # upper half of N-day range
    adx   = ta.adx(H, L, C, length=14)["ADX_14"]
    adxon = adx > a.adx_th

    gates = {
        "level":        level,
        "slope":        slope,
        "level+slope":  level & slope,
        "level+chop":   level & chop,
        "level+donch":  level & donch,
        "level+adx":    level & adx,
    } | {"level+adx": level & adxon}

    years = sorted(set(d.index.year))
    cols = list(gates.keys())
    print(f"\n=== GOLD daily gate PASS RATE (% of days ON) by year  sma{a.sma} slopeK{a.slope_k} "
          f"chop<{a.chop_th} donch{a.donch_n} adx>{a.adx_th} ===")
    print(f"  {'year':<6}{'P&L(R)':>8}   " + "".join(f"{c:>13}" for c in cols))
    for y in years:
        m = d.index.year == y
        ref = REF.get(y, None)
        refs = f"{ref:+d}" if ref is not None else "  ·"
        row = "".join(f"{gates[c][m].mean()*100:>12.0f}%" for c in cols)
        tag = ""
        if ref is not None:
            tag = "  <- KEEP" if ref >= 10 else ("  <- drop" if ref <= -2 else "")
        print(f"  {y:<6}{refs:>8}   {row}{tag}")
    # correlation: does ON% track the good years?
    print("\n  (KEEP = trend year we want ON; drop = chop/loss year we want OFF)")


if __name__ == "__main__":
    main()
