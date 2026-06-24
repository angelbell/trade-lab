"""gold_overextension.py -- the user's intuition, quantified: on gold M15, after price moves X in ONE
direction (over W bars), does the NEXT move continue or REVERSE? At what magnitude does continuation flip
to reversal -- in $ and in ATR multiples?

This is the raw-material screen behind "if it runs too far one way, it tends to bounce" (the ORB context).
NOT a strategy yet -- a conditional-forward-return measurement. For each bar: extension = signed move over
the last W bars; outcome = signed forward move over the next H bars, ALIGNED to the extension direction
(aligned_fwd = sign(ext) * fwd). aligned_fwd > 0 = continuation, < 0 = reversal/mean-reversion. Bucket by
|extension| in ATR units (and show the $ at gold's current level). Where aligned_fwd turns negative = the
"overextended, likely to bounce" zone -- the magnitude the user is asking about.

Honesty: gross (cost noted separately ~ gold M15 spread+slip ~0.03-0.05% = ~$1.3-2.1 at $4150). Recent dense
M15 (2019+). In-sample, descriptive -- a reversal zone here is raw material, not a validated edge.
  .venv/bin/python research/gold_overextension.py
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


def screen(d, W, H, px):
    c = d["close"].values
    a = atr(d).values
    n = len(c)
    ext, fwd, ea = [], [], []
    for t in range(W, n - H):
        if not np.isfinite(a[t]) or a[t] <= 0:
            continue
        e = c[t] - c[t - W]                       # signed extension over last W bars ($)
        f = c[t + H] - c[t]                        # signed forward move over next H bars ($)
        ext.append(e); fwd.append(f); ea.append(a[t])
    ext, fwd, ea = np.array(ext), np.array(fwd), np.array(ea)
    aligned = np.sign(ext) * fwd                   # >0 continuation, <0 reversal
    extA = np.abs(ext) / ea                        # extension in ATR units
    dollar = np.abs(ext)                           # extension in $
    print(f"\n== extension over {W} bars ({W*15}min) -> forward {H} bars ({H*15}min)  (ATR~${np.median(ea):.1f}) ==")
    print(f"  {'|ext| bucket':>16} {'n':>7} {'med$':>7} {'alignedFwd$':>12} {'cont%':>7}  read")
    edges = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 4), (4, 6), (6, 99)]
    for lo, hi in edges:
        m = (extA >= lo) & (extA < hi)
        if m.sum() < 200:
            continue
        af = aligned[m].mean()
        contp = (aligned[m] > 0).mean() * 100
        tag = "REVERSAL" if af < -0.2 else ("continue" if af > 0.2 else "flat")
        print(f"  {lo:>4.1f}-{hi:<4.1f}ATR {m.sum():>7} {np.median(dollar[m]):>7.1f} {af:>+12.2f} {contp:>6.1f}%  {tag}")
    # $ at current price grid
    print(f"  (at gold ${px:.0f}: 1 ATR~${np.median(ea):.1f}; so e.g. 2 ATR ~ ${2*np.median(ea):.0f} move)")


def main():
    d = resample(load_mt5_csv("data/vantage_xauusd_m5.csv"), "15min")
    d = d[d.index.year >= 2019]                     # dense M15 era
    px = d["close"].iloc[-1]
    print(f"GOLD M15 over-extension -> reversal screen (Vantage, 2019+, n={len(d)}, last~${px:.0f})")
    print("aligned forward move: + = move CONTINUES, - = REVERSES (bounce). cost ~$1.3-2.1 round-trip (note).")
    for W in (4, 8, 12):                            # 1h, 2h, 3h of M15 extension
        for H in (4, 8):                            # forward 1h, 2h
            screen(d, W, H, px)
    print("\n  read: find the |ext| bucket where alignedFwd$ turns clearly NEGATIVE = the 'ran too far, bounces'")
    print("  zone. Compare that $ to the ~$1.3-2.1 cost: a reversal worth trading must be several $ deep, not <cost.")


if __name__ == "__main__":
    main()
