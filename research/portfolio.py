"""portfolio.py -- combine the two validated trend-riders into ONE equity curve.

The point of holding Gold breakout + BTC breakout together is NOT more return --
each is a modest trend-rider -- it is DIVERSIFICATION: their good/bad years are
offset (2021 gold -18R / BTC +3R; 2023 gold ~0 / BTC +7R), so the JOINT equity
curve should have a smaller drawdown than either leg alone. This measures that.

Each leg sized at --risk per trade on a SHARED account; trades from both legs are
merged in chronological order and compounded. (When both legs hold at once the
account briefly risks 2x --risk; that concurrency is real and is left in, not hidden.)

  Gold leg : 1H Pattern-B zz2 EMA80 RR3  + daily-SMA150 + slope10 gate  (gold config)
  BTC leg  : 4H Pattern-B zz2 EMA80 RR2  NO daily gate                  (btc config)

  .venv/bin/python research/portfolio.py
"""
import os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from ema_pullback import run as run_pullback

# full arg set ema_pullback.run() expects (BTC pullback leg = SMA-trend, fast20, RR3, long)
PB = dict(side="long", ema_fast=20, ema_slow=80, slope_k=6, filter="slope", er_period=14,
          swap_pct=0.0, daily_ema=0, exit_sma=0, exit_ma_type="sma", peryear=False,
          no_overlap=True, entry_trigger="close", fill_at_close=True, rr=3.0,
          min_stop_atr=0.5, atr=14, fwd=90, cost=0.001, trend_ma_type="sma", fast_ma_type="ema")

BASE = dict(pattern="B", sl_mode="line", sl_buf=0.25, swing="zigzag", zz_k=2.0,
            pivot_n=5, renko_k=2.0, mom_fast=12, mom_slow=26, trend_ema=80,
            bo_window=20, tp_mode="rr", atr=14, cost=0.001, swap_pct=0.0,
            peryear=True, start=None, end=None, daily_sma=0, daily_slope_k=0)


def get_trades(csv, tf, **over):
    args = SimpleNamespace(**{**BASE, **over, "csv": csv, "tf": tf, "risk": over.get("risk", 0.01)})
    d = resample(load_mt5_csv(csv), tf)
    print(f"\n[{os.path.basename(csv)} {tf}]")
    t = run(d, args)
    return t[["time", "R", "hold"]].copy() if t is not None else None


def get_trades_pullback(csv, tf, **over):
    """3rd leg: BTC EMA pullback (different module, run(d, side, args, thr))."""
    args = SimpleNamespace(**{**PB, **over, "csv": csv, "tf": tf})
    d = resample(load_mt5_csv(csv), tf)
    print(f"\n[{os.path.basename(csv)} {tf} pullback]")
    t = run_pullback(d, args.side, args, 0.0)     # thr=0.0 = no extra filter
    return t[["time", "R", "hold"]].copy() if t is not None else None


def metrics(eq, t):
    dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100
    span = max((t["time"].iloc[-1] - t["time"].iloc[0]).days / 365.25, 0.5)
    cagr = (eq.iloc[-1] ** (1 / span) - 1) * 100
    ret = (eq.iloc[-1] - 1) * 100
    return ret, cagr, dd, ret / max(dd, 1e-9)


def report(name, t, risk):
    t = t.sort_values("time").reset_index(drop=True)
    eq = (1 + risk * t["R"]).cumprod()
    ret, cagr, dd, rdd = metrics(eq, t)
    print(f"  {name:<18} n={len(t):>4}  return={ret:+5.0f}%  CAGR={cagr:+5.1f}%  "
          f"maxDD={dd:4.1f}%  ret/DD={rdd:5.2f}")
    return t


def main():
    risk = 0.01
    gold = get_trades("data/vantage_xauusd_h1.csv", "1h", rr=3.0, fwd=500,
                      daily_sma=150, daily_slope_k=10, risk=risk)
    btc = get_trades("data/vantage_btcusd_h1.csv", "4h", rr=2.0, fwd=300, risk=risk)
    btcpb = get_trades_pullback("data/vantage_btcusd_h1.csv", "4h", risk=risk)

    print(f"\n=== standalone vs COMBINED  (risk {risk*100:.0f}%/trade each, shared account) ===")
    report("GOLD breakout", gold, risk)
    report("BTC breakout", btc, risk)
    report("BTC pullback", btcpb, risk)
    report("2-leg (G.bo+B.bo)", pd.concat([gold, btc], ignore_index=True), risk)
    comb = report("3-leg (+B.pull)", pd.concat([gold, btc, btcpb], ignore_index=True), risk)

    # per-year side by side
    gy = gold.groupby(gold.time.dt.year)["R"].sum()
    by = btc.groupby(btc.time.dt.year)["R"].sum()
    py = btcpb.groupby(btcpb.time.dt.year)["R"].sum()
    cy = comb.groupby(comb.time.dt.year)["R"].sum()
    yrs = sorted(set(gy.index) | set(by.index) | set(py.index))
    print(f"\n  {'year':<6}{'G.bo':>8}{'B.bo':>8}{'B.pull':>8}{'3-leg':>9}")
    for y in yrs:
        print(f"  {y:<6}{gy.get(y,0):>+8.0f}{by.get(y,0):>+8.0f}{py.get(y,0):>+8.0f}{cy.get(y,0):>+9.0f}")
    # pairwise annual-R correlations (diversification check; BTC bo vs BTC pull = same asset)
    al = pd.concat([gy, by, py], axis=1).fillna(0); al.columns = ["g", "b", "p"]
    print(f"\n  annual-R corr  GOLD.bo vs BTC.bo = {al['g'].corr(al['b']):+.2f}  | "
          f"GOLD.bo vs BTC.pull = {al['g'].corr(al['p']):+.2f}  | "
          f"BTC.bo vs BTC.pull = {al['b'].corr(al['p']):+.2f} (same asset -> expect higher)")


if __name__ == "__main__":
    main()
