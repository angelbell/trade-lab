"""ext_cap_null.py -- does the 15M gold_bo extension-cap BEAT a random-drop of the same N?

The lab law: a within-leg filter must beat the CAGR/DD random-drop null, not just the meanR
null (stop-width/ER scored >90%ile on meanR but ~61%ile on CAGR/DD = pure n-trimming sorters).

PURE-MASK test (isolates the filter's SELECTION from breakout_wave's no-overlap replacement
effect): take the BASELINE 674-trade stream, join each trade's prior-day extension above the
daily SMA, then KEEP only ext<=cap. obs CAGR/DD = on that masked subset (time-ordered, 1% risk).
NULL = random subsets of the SAME size from the baseline stream. Percentile = where obs sits.
>~90%ile = real selection; ~50-60%ile = just n-trimming. Reported at several caps (robust vs spike).

  .venv/bin/python research/ext_cap_null.py
"""
import os, sys, subprocess, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from research.overfit_audit import cdd_R

CSV = "data/vantage_xauusd_m15.csv"
BASE = ["--csv", CSV, "--tf", "15min", "--pattern", "B", "--swing", "zigzag", "--zz-k", "2",
        "--trend-ema", "80", "--bo-window", "20", "--tp-mode", "rr", "--rr", "4", "--fwd", "500",
        "--daily-sma", "150", "--daily-slope-k", "10", "--risk", "0.01", "--cost", "0.0002"]
SPLIT = 2022


def base_trades():
    out = subprocess.run([".venv/bin/python", "breakout_wave.py", *BASE, "--dump-trades"],
                         capture_output=True, text=True).stdout.splitlines()
    i = next(k for k, l in enumerate(out) if l.startswith("entry_time,"))
    t = pd.DataFrame([l.split(",") for l in out[i + 1:] if l], columns=["entry_time", "R", "hold"])
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True); t["R"] = t["R"].astype(float)
    return t.set_index("entry_time").sort_index()


def ext_feature():
    d = load_mt5_csv(CSV)
    dc = d.close.resample("1D").last().dropna()
    sma = dc.rolling(150).mean()
    ext = (dc - sma) / sma * 100.0
    return ext.shift(1).reindex(d.index, method="ffill")     # prior-day, no lookahead


def cdd(r):
    yrs = 1.0  # CAGR/DD ratio is invariant to the year normalization; use 1.0 consistently
    return cdd_R(r, yrs)[2]


def main():
    t = base_trades()
    ext = ext_feature()
    t["ext"] = ext.reindex(t.index, method="ffill").values
    t = t.dropna(subset=["ext"])
    r_all = t.R.values
    obs_all = cdd(r_all)
    print(f"baseline (no cap): n={len(t)}  meanR={t.R.mean():+.2f}  CAGR/DD={obs_all:.2f}  "
          f"IS={t[t.index.year<SPLIT].R.mean():+.2f} OOS={t[t.index.year>=SPLIT].R.mean():+.2f}")
    print(f"\n{'cap%':>5}{'kept':>6}{'meanR':>7}{'IS':>6}{'OOS':>6}{'CAGR/DD':>9}{'null%ile':>10}  verdict")
    rng = np.random.default_rng(0); B = 5000
    for cap in (5, 7, 9, 12):
        m = t[t.ext <= cap]
        k = len(m)
        if k < 30:
            continue
        obs = cdd(m.R.values)
        # null: random subsets of size k from the baseline stream (time order preserved)
        idx = np.arange(len(t))
        nul = np.empty(B)
        for b in range(B):
            sel = np.sort(rng.choice(idx, k, replace=False))
            nul[b] = cdd(r_all[sel])
        pct = (nul < obs).mean() * 100
        verd = "REAL (beats n-trim)" if pct >= 90 else "n-trim/luck" if pct <= 70 else "weak"
        print(f"{cap:>5}{k:>6}{m.R.mean():>+7.2f}{m[m.index.year<SPLIT].R.mean():>+6.2f}"
              f"{m[m.index.year>=SPLIT].R.mean():>+6.2f}{obs:>9.2f}{pct:>9.0f}%  {verd}")
    print("\n(>=90%ile = the extension filter selects better than dropping the same N at random;")
    print(" <=70%ile = it's just trimming trade count, like the stop-width/ER sorters.)")


if __name__ == "__main__":
    main()
