"""lowtf_gold_diag.py -- WHERE does the 15M gold breakout (gold_bo @ 15min, RR4) lose?

Pulls the trade stream via `breakout_wave.py --dump-trades`, joins each entry to its
context (session / hour / day-of-week / vol-regime / trend-extension / daily-slope /
KAMA-state), and tabulates n / win% / meanR / totR PER BUCKET, split IS(<2022)/OOS(>=2022).

This is DIAGNOSTIC (hypothesis-generating), NOT a validated filter: slicing a single
stream many ways finds spurious bad buckets. A bucket is only a *candidate* gate if it is
(a) bad in BOTH IS and OOS and (b) big enough n. Validation (random-drop null) is a next step.

  .venv/bin/python research/lowtf_gold_diag.py
"""
import os, sys, subprocess, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from research.regime_adaptive import kama

CSV = "data/vantage_xauusd_m15.csv"
CFG = ["--csv", CSV, "--tf", "15min", "--pattern", "B", "--swing", "zigzag", "--zz-k", "2",
       "--trend-ema", "80", "--bo-window", "20", "--tp-mode", "rr", "--rr", "4", "--fwd", "500",
       "--daily-sma", "150", "--daily-slope-k", "10", "--risk", "0.01", "--cost", "0.0002"]
SPLIT = 2022


def get_trades():
    out = subprocess.run([".venv/bin/python", "breakout_wave.py", *CFG, "--dump-trades"],
                         capture_output=True, text=True).stdout.splitlines()
    i = next(k for k, l in enumerate(out) if l.startswith("entry_time,"))
    t = pd.DataFrame([l.split(",") for l in out[i + 1:] if l],
                     columns=["entry_time", "R", "hold"])
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)
    t["R"] = t["R"].astype(float)
    return t.set_index("entry_time")


def features():
    d = load_mt5_csv(CSV)
    f = pd.DataFrame(index=d.index)
    f["hour"] = d.index.hour
    f["dow"] = d.index.dayofweek
    tr = pd.concat([(d.high - d.low), (d.high - d.close.shift()).abs(),
                    (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    f["atr_pct"] = atr.rolling(480).rank(pct=True)          # vol regime, 5-day window
    # daily features, shifted 1 day (only prior completed day known at entry) -> no lookahead
    dc = d.close.resample("1D").last().dropna()
    sma = dc.rolling(150).mean()
    ext = (dc - sma) / sma * 100                            # % above daily SMA150 (extension)
    dslope = (sma - sma.shift(10)) / sma.shift(10) * 100    # daily SMA slope over 10d, %
    km = kama(dc, 14); krise = (km > km.shift(1)).astype(float)
    for name, s in [("ext", ext), ("dslope", dslope), ("kama_rise", krise)]:
        f[name] = s.shift(1).reindex(d.index, method="ffill")
    return f


def sess(h):
    return "Asia(0-7)" if h < 7 else "London(7-13)" if h < 13 else "NY(13-21)" if h < 21 else "Late(21-24)"


def tbl(t, col, label, order=None, fmt=str):
    rows = []
    g = t.dropna(subset=[col]).copy()
    keys = order if order is not None else sorted(g[col].unique())
    for k in keys:
        s = g[g[col] == k]
        if len(s) == 0:
            continue
        isr = s[s.index.year < SPLIT]["R"]; oosr = s[s.index.year >= SPLIT]["R"]
        rows.append((fmt(k), len(s), (s.R > 0).mean() * 100, s.R.mean(), s.R.sum(),
                     isr.mean() if len(isr) else np.nan, oosr.mean() if len(oosr) else np.nan))
    print(f"\n## {label}")
    print(f"  {'bucket':<14}{'n':>5}{'win%':>6}{'meanR':>7}{'totR':>7}{'IS':>7}{'OOS':>7}")
    for r in rows:
        print(f"  {r[0]:<14}{r[1]:>5}{r[2]:>6.0f}{r[3]:>+7.2f}{r[4]:>+7.0f}"
              f"{r[5]:>+7.2f}{r[6]:>+7.2f}")


def main():
    t = get_trades().join(features())
    print(f"15M gold_bo RR4 diagnostic -- {len(t)} trades, "
          f"overall meanR={t.R.mean():+.2f} win={ (t.R>0).mean()*100:.0f}%  (IS/OOS split @ {SPLIT})")
    t["session"] = [sess(h) for h in t["hour"]]
    tbl(t, "session", "by SESSION (UTC broker)",
        order=["Asia(0-7)", "London(7-13)", "NY(13-21)", "Late(21-24)"])
    tbl(t, "dow", "by DAY-OF-WEEK", order=[0, 1, 2, 3, 4, 6],
        fmt=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(d)])
    for col, lbl, qs in [("atr_pct", "by VOL REGIME (ATR pctile)", [0, .33, .66, 1.0]),
                         ("ext", "by EXTENSION above daily SMA150 (%)", [0, .33, .66, 1.0]),
                         ("dslope", "by DAILY SLOPE (10d %)", [0, .33, .66, 1.0])]:
        q = t[col].quantile(qs).values
        lab = ["low", "mid", "high"]
        t[col + "_b"] = pd.cut(t[col], bins=np.unique(q), labels=lab[:len(np.unique(q)) - 1],
                               include_lowest=True)
        tbl(t, col + "_b", lbl, order=lab)
    tbl(t, "kama_rise", "by daily KAMA rising", order=[0.0, 1.0],
        fmt=lambda v: "falling" if v == 0 else "rising")
    tbl(t, "y" if "y" in t else t.assign(y=t.index.year).columns and "y", "by YEAR") if False else \
        tbl(t.assign(yr=t.index.year), "yr", "by YEAR")


if __name__ == "__main__":
    main()
