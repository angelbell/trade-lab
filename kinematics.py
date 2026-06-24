"""kinematics.py -- falsify the "Institutional Kinematic Physics" MQL5 indicator.

Strip the physics branding and it's plain momentum:
  Velocity     = (close[t] - close[t-dt]) / dt           == ROC / Momentum
  Acceleration = (Velocity[t] - Velocity[t-dt]) / dt     == 2nd difference of price
The 'institutional kinematics' label is cosmetic (cf. the 'ML' in ML SuperTrend = k-means,
which added 0 lift). Prior: bare momentum = ~0 lift / trend-beta, and differentiating twice
amplifies noise, so acceleration likely HURTS. Test it.

Indicator defines no entry, so we impose testable rules (always-in, flip on signal change,
next-bar-open fill, R in ATR units, cost charged per round trip):
  vel   : long if velocity>0 else short            (momentum trend-follow)
  accel : long if acceleration>0 else short
  both  : long if vel>0 AND accel>0; short if vel<0 AND accel<0; else flat
The KEY A/B = does 'both' (add acceleration) beat 'vel' alone? And does either survive cost,
the short side (not just long-only beta), and IS/OOS?

  .venv/bin/python kinematics.py --csv data/vantage_btcusd_h1.csv --tf 4h --mode vel --peryear
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

from src.data_loader import load_mt5_csv


def resample(df, rule):
    if rule.lower() in ("1h", "h1", ""):
        return df
    o = {"4h": "4h", "h4": "4h", "1d": "1D", "d1": "1D", "2h": "2h", "12h": "12h"}.get(rule.lower(), rule)
    return pd.DataFrame({"open": df["open"].resample(o).first(), "high": df["high"].resample(o).max(),
                         "low": df["low"].resample(o).min(), "close": df["close"].resample(o).last()}).dropna()


def desired_pos(c, dt, mode):
    n = len(c)
    vel = np.full(n, np.nan); acc = np.full(n, np.nan)
    vel[dt:] = (c[dt:] - c[:-dt]) / dt
    acc[2 * dt:] = (vel[2 * dt:] - vel[dt:-dt]) / dt
    want = np.zeros(n, np.int8)
    if mode == "vel":
        want = np.where(vel > 0, 1, np.where(vel < 0, -1, 0)).astype(np.int8)
    elif mode == "accel":
        want = np.where(acc > 0, 1, np.where(acc < 0, -1, 0)).astype(np.int8)
    else:  # both
        want = np.where((vel > 0) & (acc > 0), 1, np.where((vel < 0) & (acc < 0), -1, 0)).astype(np.int8)
    want[:2 * dt] = 0
    return want


def run(d, args, dt):
    c = d["close"].values; o = d["open"].values
    atr = ta.atr(d["high"], d["low"], d["close"], length=14).values
    want = desired_pos(c, dt, args.mode)
    if args.side == "long":
        want = np.where(want > 0, 1, 0).astype(np.int8)
    elif args.side == "short":
        want = np.where(want < 0, -1, 0).astype(np.int8)
    n = len(c); cost = args.cost
    trades = []
    pos = 0; e_px = e_risk = 0.0; ei = 0
    for i in range(n - 1):
        if want[i] != pos:
            if pos != 0 and e_risk > 0:
                ret = (o[i + 1] - e_px) / e_px if pos > 0 else (e_px - o[i + 1]) / e_px
                trades.append((d.index[ei], pos, (ret - cost) / e_risk))
            if want[i] != 0:
                e_px = o[i + 1]; ei = i + 1; pos = int(want[i])
                e_risk = atr[i] / e_px if atr[i] > 0 else np.nan
            else:
                pos = 0
    t = pd.DataFrame(trades, columns=["t", "dir", "R"]).dropna()
    if len(t) == 0:
        print(f"  dt={dt}: no trades"); return
    wins = t.R[t.R > 0]; loss = t.R[t.R < 0]
    pf = wins.sum() / abs(loss.sum()) if len(loss) and loss.sum() else float("inf")
    t["y"] = t.t.dt.year; yrs = sorted(t.y.unique()); half = yrs[len(yrs) // 2] if len(yrs) > 1 else None
    isr = t.R[t.y < half] if half else t.R; oos = t.R[t.y >= half] if half else t.R
    print(f"  dt={dt:>2} n={len(t):>4} win={(t.R>0).mean()*100:>3.0f}% PF={pf:4.2f} "
          f"meanR={t.R.mean():+.2f} totR={t.R.sum():+5.0f} | IS={isr.mean():+.2f} OOS={oos.mean():+.2f}")
    if args.peryear:
        print("      per-year totR: " + " ".join(f"{y}:{g.R.sum():+.0f}({len(g)})" for y, g in t.groupby("y")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--mode", default="vel", choices=["vel", "accel", "both"])
    ap.add_argument("--side", default="both", choices=["both", "long", "short"])
    ap.add_argument("--dt", type=int, default=3)
    ap.add_argument("--sweep", action="store_true", help="sweep dt {2,3,5,8,13}")
    ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--peryear", action="store_true")
    a = ap.parse_args()
    d = resample(load_mt5_csv(a.csv), a.tf)
    print(f"\n=== {os.path.basename(a.csv)} {a.tf} mode={a.mode} side={a.side} cost{a.cost} "
          f"{d.index[0].date()}->{d.index[-1].date()} ===")
    for dt in ([2, 3, 5, 8, 13] if a.sweep else [a.dt]):
        run(d, a, dt)


if __name__ == "__main__":
    main()
