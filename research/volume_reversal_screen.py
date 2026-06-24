"""volume_reversal_screen.py -- CHEAP screen: does tick-volume carry REVERSAL information on FX?

User's wish: predict reversals from price + volume on USD pairs. HONESTY: FX 'volume' = TICK volume
(price-change count from Vantage), NOT real order flow -- it's an activity/volatility proxy. So this asks
"does an activity SPIKE mark exhaustion -> reversal?" -- a weaker claim than equity volume.

Screen (no strategy yet): bucket bars by (recent move dir over M bars) x (tick-vol percentile over W).
Measure forward K-bar return (ATR-normalized) and reversal rate. Exhaustion hypothesis => after an
UP move with HIGH vol, forward return is NEGATIVE and more so than after UP with LOW vol.

Falsifier (up front): the vol-conditioning must (a) beat the no-vol baseline (a real spread between
high-vol and low-vol forward returns) AND (b) not just proxy volatility (compare to an ATR-spike bucket).
No spread => tick-volume has no reversal edge, stop. In-sample; descriptive screen only.
  .venv/bin/python research/volume_reversal_screen.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv


def resample(df, tf):
    """volume-preserving resample (breakout_wave.resample drops volume)."""
    if tf.lower() in ("1h", "h1", ""):
        return df
    rule = {"4h": "4h", "1d": "1D"}.get(tf.lower(), tf)
    return pd.DataFrame({"open": df["open"].resample(rule).first(),
                         "high": df["high"].resample(rule).max(),
                         "low": df["low"].resample(rule).min(),
                         "close": df["close"].resample(rule).last(),
                         "volume": df["volume"].resample(rule).sum()}).dropna()


def atr(d, n=14):
    h, l, c = d["high"], d["low"], d["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def screen(name, d, M=6, K=6, W=100):
    c = d["close"]
    a = atr(d)
    vol = d["volume"]
    volpr = vol.rolling(W).rank(pct=True)                       # tick-vol percentile (trailing)
    atrpr = a.rolling(W).rank(pct=True)                          # ATR percentile (to separate vol from volatility)
    recent = np.sign((c - c.shift(M)).values)                   # recent move direction (past only)
    fwd = ((c.shift(-K) - c) / a).values                        # forward K-bar return, ATR-normalized (target)
    df = pd.DataFrame({"recent": recent, "fwd": fwd, "volpr": volpr.values, "atrpr": atrpr.values}).dropna()
    df = df[np.isfinite(df.fwd)]

    print(f"\n== {name}  (M={M} recent, K={K} fwd, W={W} pctile) ==")
    print(f"  up-bar base rate = {(np.sign(c.diff().dropna())>0).mean()*100:.1f}%   "
          f"mean fwd (all) = {df.fwd.mean():+.3f} ATR")
    print(f"  {'recent move':>12} {'vol bucket':>11} {'n':>6} {'mean fwd(ATR)':>14} {'reversal%':>10}")
    for rec, rlab in [(1, "UP"), (-1, "DOWN")]:
        sub = df[df.recent == rec]
        for lab, mask in [("vol HIGH(>=.8)", sub.volpr >= 0.8), ("vol LOW(<=.2)", sub.volpr <= 0.2),
                          ("ALL vols", sub.volpr >= 0)]:
            s = sub[mask]
            rev = ((np.sign(s.fwd) != rec)).mean() * 100                # forward opposes recent = reversal
            print(f"  {rlab:>12} {lab:>11} {len(s):>6} {s.fwd.mean():>+14.3f} {rev:>9.1f}%")
    # control: is the HIGH-vol effect just an ATR(volatility)-spike effect?
    for rec, rlab in [(1, "UP"), (-1, "DOWN")]:
        sub = df[df.recent == rec]
        hv = sub[sub.volpr >= 0.8].fwd.mean(); ha = sub[sub.atrpr >= 0.8].fwd.mean()
        print(f"  [ctrl] {rlab}: meanfwd vol-HIGH={hv:+.3f} vs ATR-HIGH={ha:+.3f}  "
              f"(if ~equal, 'volume' is just volatility)")


def main():
    for name, csv, tf in [("USDJPY 1h", "data/vantage_usdjpy_h1.csv", "1h"),
                          ("USDJPY 4h", "data/vantage_usdjpy_h1.csv", "4h"),
                          ("USDX 1h", "data/vantage_usdx.r_h1.csv", "1h")]:
        d = resample(load_mt5_csv(csv), tf)
        screen(name, d)
    print("\n  read: a REAL reversal edge = HIGH-vol forward return clearly MORE negative (after UP) /")
    print("        more positive (after DOWN) than LOW-vol, AND different from the ATR-spike control.")


if __name__ == "__main__":
    main()
