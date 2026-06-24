"""Idea 1b — PRE-REGISTERED session-window test (higher power than the 24-way search).

Instead of searching all 24 hours (which needs the big multiple-comparisons null),
we name a FEW structurally-motivated windows (anchored on the gold volume profile,
server time = UTC+2/+3) and test each ONE directly. Far higher power; only ~5
comparisons to haircut, not 24.

Windows (server hour, [start,end) inclusive-exclusive on the hour):
  asia_range  [1,8)    Asia session, low-vol range building (~23-06 UTC)
  london_am   [9,13)   London open + morning (vol cluster 9-12)
  ny_overlap  [14,18)  London-NY overlap, the volume PEAK (NY open ~15-16 srv)
  ny_pm       [18,22)  NY afternoon
  late_thin   [22,24)+[0,1)  late/rollover, thin

Per window, pre-registered: meanR vs the rest-of-day, a label-permutation p-value
(shuffle in/out membership, keep the count), a plateau check (shrink/expand by 1h),
and per-year sign. PASS = meanR materially > rest AND perm-p < 0.01 (Bonferroni ~5)
AND smooth under +/-1h AND positive in a majority of years.

Run:  .venv/bin/python research/session_windows.py
"""
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/bin/python")
RNG = np.random.default_rng(7)

DENSE = ["--csv", "data/vantage_xauusd_h1.csv", "--tf", "1h", "--pattern", "B",
         "--swing", "zigzag", "--zz-k", "1", "--bo-window", "20",
         "--tp-mode", "rr", "--rr", "3", "--fwd", "500"]
GATED = ["--csv", "data/vantage_xauusd_h1.csv", "--tf", "1h", "--pattern", "B",
         "--swing", "zigzag", "--zz-k", "2", "--trend-ema", "80", "--bo-window", "20",
         "--tp-mode", "rr", "--rr", "3", "--fwd", "500",
         "--daily-sma", "150", "--daily-slope-k", "10"]

WINDOWS = {
    "asia_range": list(range(1, 8)),
    "london_am":  list(range(9, 13)),
    "ny_overlap": list(range(14, 18)),
    "ny_pm":      list(range(18, 22)),
    "late_thin":  [22, 23, 0],
}


def get_trades(extra):
    out = subprocess.run([PY, "breakout_wave.py", *extra, "--dump-trades"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    lines = out.splitlines()
    s = next(i for i, ln in enumerate(lines) if ln.startswith("entry_time"))
    df = pd.read_csv(io.StringIO("\n".join(lines[s:])))
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["hour"] = df["entry_time"].dt.hour
    df["year"] = df["entry_time"].dt.year
    return df


def perm_p(df, hours, n=5000):
    """One-sided p: P(random same-size subset has meanR >= window meanR)."""
    inmask = df["hour"].isin(hours).values
    k = inmask.sum()
    if k == 0:
        return np.nan, np.nan
    R = df["R"].values
    obs = R[inmask].mean()
    idx = np.arange(len(R))
    null = np.empty(n)
    for i in range(n):
        sel = RNG.choice(idx, size=k, replace=False)
        null[i] = R[sel].mean()
    p = (null >= obs).mean()
    return obs, p


def expand(hours, delta):
    """shrink (delta<0) or expand (delta>0) the contiguous window by |delta| on each end."""
    lo, hi = min(hours), max(hours)
    if 0 in hours and 23 in hours:   # wrap-around window: leave as-is
        return hours
    return list(range(lo - delta, hi + 1 + delta)) if delta >= 0 \
        else list(range(lo - delta, hi + 1 + delta))


def analyze(name, df):
    print("=" * 78)
    print(f"{name}   n={len(df)}  base meanR={df['R'].mean():+.3f}  "
          f"({df['year'].min()}-{df['year'].max()})")
    print("=" * 78)
    print(f"  {'window':12s} {'hours':16s} {'n':>4} {'meanR':>7} {'win%':>5} "
          f"{'rest':>7} {'perm-p':>7}  {'-1h':>6} {'+1h':>6}  per-year +")
    rest_all = df["R"]
    for wn, hours in WINDOWS.items():
        sub = df[df["hour"].isin(hours)]
        rest = df[~df["hour"].isin(hours)]["R"]
        if len(sub) == 0:
            print(f"  {wn:12s} {str(hours):16s}    0   (no trades)")
            continue
        obs, p = perm_p(df, hours)
        # plateau: shrink/expand by 1h
        sh = expand(hours, -1); ex = expand(hours, +1)
        m_sh = df[df["hour"].isin(sh)]["R"].mean() if len(sh) else np.nan
        m_ex = df[df["hour"].isin(ex)]["R"].mean()
        by = sub.groupby("year")["R"].sum()
        pos = (by > 0).sum()
        flag = "  <==" if (p < 0.01 and sub["R"].mean() - rest.mean() > 0.15) else ""
        print(f"  {wn:12s} {str(hours):16s} {len(sub):4d} {sub['R'].mean():+.2f} "
              f"{(sub['R']>0).mean()*100:4.0f}% {rest.mean():+.2f} {p:7.3f}  "
              f"{m_sh:+.2f} {m_ex:+.2f}  {pos}/{len(by)}{flag}")
    print("  PASS bar: perm-p<0.01 AND meanR-rest>0.15 AND smooth +/-1h AND majority-yrs+")


if __name__ == "__main__":
    analyze("DENSE all-signals base (zz-k 1, ungated)", get_trades(DENSE))
    print()
    analyze("GATED validated leg (zz-k 2 + daily-SMA)", get_trades(GATED))
