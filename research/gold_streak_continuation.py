"""gold_streak_continuation.py -- the user's screenshot intuition #2: after N consecutive DOWN bars (a
persistent decline), does gold M15 KEEP FALLING (no bounce) or mean-revert? This is the CONTINUATION
question, conditioned on streak LENGTH (count of consecutive same-direction bars) -- distinct from
gold_overextension.py which conditioned on MAGNITUDE (ATR move over a window, mixing streaks & big bars).

Prior: H10 'streak fade' (scalp_lab) died on gold 5m. The continuation side on M15 wasn't measured directly.
Measure it: for a run of L consecutive down bars, the forward move over H bars, signed so + = the decline
CONTINUES (user's 'doesn't come back'), - = it bounces. Report continue% vs 50, mean forward in $ and ATR,
across L buckets, on the full era (2019+) and the current high-vol regime (2024+). Symmetric up-runs too
(beta check: if only down-runs 'continue' it's just the 2025-26 downtrend, not a streak effect).

Honesty: gross, descriptive; cost ~$1.3-2.1 RT noted. A tradeable continuation needs forward move >> cost
AND continue% meaningfully >50 AND to hold both directions / both eras (else it's just trend beta). In-sample.
  .venv/bin/python research/gold_streak_continuation.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from research.volume_reversal_screen import resample


def atr(d, n=14):
    h, l, c = d["high"], d["low"], d["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def run_lengths(c):
    """signed consecutive-bar run length at each bar: +k = k consecutive up-closes, -k = k down."""
    up = (np.diff(c) > 0).astype(int)
    dn = (np.diff(c) < 0).astype(int)
    rl = np.zeros(len(c), dtype=int)
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            rl[i] = rl[i - 1] + 1 if rl[i - 1] > 0 else 1
        elif c[i] < c[i - 1]:
            rl[i] = rl[i - 1] - 1 if rl[i - 1] < 0 else -1
        else:
            rl[i] = 0
    return rl


def screen(d, side, H, px):
    c = d["close"].values
    a = atr(d).values
    rl = run_lengths(c)
    med = np.nanmedian(a)
    n = len(c)
    sgn = -1 if side == "down" else +1                    # down-run: continuation = price falls further
    print(f"\n== {side}-streak -> forward {H} bars ({H*15}min) continuation  (ATR med ${med:.1f}) ==")
    print(f"  {'run len':>8} {'n':>7} {'medRunMove$':>12} {'fwdCont$':>9} {'fwdCont(ATR)':>13} {'continue%':>10}")
    for L in (2, 3, 4, 5, 6, 8):
        hi = L + 1 if L < 8 else 99
        if side == "down":
            m = (rl <= -L) & (rl > -hi)
        else:
            m = (rl >= L) & (rl < hi)
        m[n - H:] = False                                  # need forward window
        m[:14] = False
        idx = np.where(m)[0]
        idx = idx[np.isfinite(a[idx]) & (a[idx] > 0)]
        if len(idx) < 100:
            continue
        fwd = c[idx + H] - c[idx]                           # signed forward move
        cont = sgn * fwd                                   # + = the run direction continues
        run_move = c[idx] - c[idx - L]                      # the L-bar move that built the run
        contp = (cont > 0).mean() * 100
        tag = "CONTINUES" if cont.mean() > 0.2 else ("bounces" if cont.mean() < -0.2 else "flat")
        print(f"  {L:>6}+  {len(idx):>7} {np.median(np.abs(run_move)):>12.1f} {cont.mean():>+9.2f} "
              f"{cont.mean()/med:>+13.2f} {contp:>9.1f}%  {tag}")


def main():
    full = resample(load_mt5_csv("data/vantage_xauusd_m5.csv"), "15min")
    for era, lo in [("2019+", 2019), ("2024+ (current vol)", 2024)]:
        d = full[full.index.year >= lo]
        px = d["close"].iloc[-1]
        print(f"\n############ GOLD M15 {era}  n={len(d)}  last~${px:.0f} ############")
        for side in ("down", "up"):
            for H in (4, 8):
                screen(d, side, H, px)
    print("\n  read: continue% >>50 AND fwdCont$ >> cost (~$1.3-2.1) = the decline really doesn't bounce")
    print("  (user's intuition). If only DOWN-runs continue (not UP) it's 2025-26 downtrend beta, not a")
    print("  streak effect. flat / continue%~50 / fwd<cost = no tradeable continuation edge from the streak.")


if __name__ == "__main__":
    main()
