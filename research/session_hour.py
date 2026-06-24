"""Idea 1 — Time-of-day / session-boundary conditioning (KILL screen).

Does the ENTRY HOUR of a gold breakout carry edge? The lab has never used the
calendar clock as a first-class signal. The danger is multiple comparisons:
24 hours x 5 weekdays = one cell WILL look great by luck. So we don't trust a
lone winning hour — we demand a CONTIGUOUS block (>=3h) and we measure it
against a PERMUTATION NULL that shuffles R across trades (breaks the hour->R
link), which prices in exactly that multiple-comparisons luck.

Pre-registered decision rule (set BEFORE looking):
  LEAD if a contiguous >=3h block's meanR beats the permutation null at >=95%ile
       AND that block is positive in a MAJORITY of years (regime-stable).
  else KILL (curve-fit spike / no clock edge).

Data: the DENSE all-signals gold-1h breakout base (zz-k 1, ungated) for sample
size, with a cross-check on the validated gated leg (zz-k 2 + daily-SMA gate).

Run:  .venv/bin/python research/session_hour.py
"""
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/bin/python")
RNG = np.random.default_rng(12345)

DENSE = ["--csv", "data/vantage_xauusd_h1.csv", "--tf", "1h", "--pattern", "B",
         "--swing", "zigzag", "--zz-k", "1", "--bo-window", "20",
         "--tp-mode", "rr", "--rr", "3", "--fwd", "500"]
GATED = ["--csv", "data/vantage_xauusd_h1.csv", "--tf", "1h", "--pattern", "B",
         "--swing", "zigzag", "--zz-k", "2", "--trend-ema", "80", "--bo-window", "20",
         "--tp-mode", "rr", "--rr", "3", "--fwd", "500",
         "--daily-sma", "150", "--daily-slope-k", "10"]


def get_trades(extra) -> pd.DataFrame:
    out = subprocess.run([PY, "breakout_wave.py", *extra, "--dump-trades"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    # skip the config banner; CSV starts at the 'entry_time' header line.
    lines = out.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("entry_time"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["hour"] = df["entry_time"].dt.hour
    df["dow"] = df["entry_time"].dt.dayofweek
    df["year"] = df["entry_time"].dt.year
    return df


def best_contiguous_block(hour_mean: np.ndarray, k: int):
    """Return (best_mean, start_hour) over circular contiguous k-hour windows."""
    best, best_h = -1e9, 0
    for h in range(24):
        idx = [(h + j) % 24 for j in range(k)]
        m = np.nanmean(hour_mean[idx])
        if m > best:
            best, best_h = m, h
    return best, best_h


def block_mean_from_trades(df, hours):
    sub = df[df["hour"].isin(hours)]
    return sub["R"].mean() if len(sub) else np.nan


def perm_pctile(df, k, observed_best, n=3000):
    """Shuffle R across trades; recompute best contiguous k-block meanR each time."""
    R = df["R"].values.copy()
    hours = df["hour"].values
    null = np.empty(n)
    for i in range(n):
        Rs = RNG.permutation(R)
        hm = np.array([Rs[hours == h].mean() if (hours == h).any() else np.nan
                       for h in range(24)])
        null[i], _ = best_contiguous_block(hm, k)
    pct = (null < observed_best).mean() * 100
    return pct, null.mean()


def analyze(name, df):
    print("=" * 74)
    print(f"{name}  n={len(df)}  ({df['year'].min()}-{df['year'].max()})  "
          f"base meanR={df['R'].mean():+.3f}")
    print("=" * 74)

    # Per-hour table.
    hm = df.groupby("hour")["R"].agg(["mean", "count"])
    hm = hm.reindex(range(24))
    print("  hour  meanR    n    win%")
    for h in range(24):
        if h in df["hour"].values:
            sub = df[df["hour"] == h]
            bar = "#" * int(max(sub['R'].mean(), 0) * 10)
            print(f"   {h:2d}   {sub['R'].mean():+.2f}  {len(sub):4d}  "
                  f"{(sub['R']>0).mean()*100:3.0f}%  {bar}")

    hour_mean = hm["mean"].values
    # Test contiguous blocks of several widths; report each vs permutation null.
    print("\n  contiguous-block test (vs permutation null, multiple-comp aware):")
    print("  k   best-block hours        meanR     null-mean   %ile")
    flagged = []
    for k in (3, 4, 6):
        bm, bh = best_contiguous_block(hour_mean, k)
        hours = [(bh + j) % 24 for j in range(k)]
        obs = block_mean_from_trades(df, hours)
        pct, nullm = perm_pctile(df, k, obs)
        star = "  <== >=95%ile" if pct >= 95 else ""
        print(f"  {k}   {str(hours):22s}  {obs:+.3f}    {nullm:+.3f}    {pct:4.1f}%{star}")
        if pct >= 95:
            flagged.append((k, hours, obs))

    # Per-year stability of any flagged block.
    for k, hours, obs in flagged:
        sub = df[df["hour"].isin(hours)]
        by = sub.groupby("year")["R"].sum()
        pos = (by > 0).sum()
        print(f"\n  [flagged k={k} hours={hours}] per-year totR sign: "
              + " ".join(f"{y}:{'+' if v>0 else '-'}" for y, v in by.items())
              + f"   [{pos}/{len(by)} yrs +]")
        verdict = "LEAD (stable)" if pos / len(by) > 0.5 else "KILL (one-era / unstable)"
        print(f"    -> {verdict}")
    if not flagged:
        print("\n  -> no contiguous block beats the null at 95%ile = KILL (no clock edge)")

    # Day-of-week quick look (secondary).
    dm = df.groupby("dow")["R"].agg(["mean", "count"])
    dnames = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print("\n  day-of-week:  " + "  ".join(
        f"{dnames[d]}:{r['mean']:+.2f}(n{int(r['count'])})" for d, r in dm.iterrows()))


if __name__ == "__main__":
    analyze("DENSE all-signals base (zz-k 1, ungated)", get_trades(DENSE))
    print()
    analyze("GATED validated leg (zz-k 2 + daily-SMA)", get_trades(GATED))
    print("\nReminder: KILL screen. A pass = 'worth a real session strategy', NOT an edge yet.")
