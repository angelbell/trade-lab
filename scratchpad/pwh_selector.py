"""pwh_selector.py -- 案A: prior-WEEK-high structure-position selector for the 1h/4h
breakout legs. Composition of two validated laws:
  (1) structure position (PDH, BTC 15m): breaks above the prior-day high are the real ones
      -- the only setup selector to pass the CAGR/DD random-drop null (100%ile).
  (2) granularity matching (weekly cycle gate): 15m bars pair with daily levels; at 1h/4h
      the PDH label saturates (ON>=73%) -> the right level for 1h/4h should be the PRIOR
      COMPLETED WEEK's high.

Cells: gold_bo (1h, RR3, SMA150 gate) and btc_bo_kama (4h, RR2, KAMA-daily gate), exact
adopted configs. Label each entry: e_px vs prior completed week's high (shift 1wk, ffill;
also PDH shown to confirm the saturation prediction). Report ON%, per-label PF/meanR,
IS/OOS, green years; verdict = equal-keep RANDOM-DROP NULL on the totR/maxDD axis
(4000 draws) for HARD (keep only ON) and SOFT (inside trades at 0.5 size).
Pre-registered: PASS needs >=90%ile on totR/DD, not just meanR (n-trim sorter trap).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run as run_bo, resample
from research.regime_gate_lab import CFG
from research.portfolio_kama import kama_gate_btc

rng = np.random.default_rng(7)


def totdd(R):
    eq = np.cumsum(R)
    dd = (np.maximum.accumulate(eq) - eq).max()
    return R.sum() / max(dd, 1e-9)


def stats(tag, t, span):
    R = t.R.values
    if len(R) < 15:
        print(f"  {tag:<28} n={len(R)} few"); return
    yr = t.time.dt.year.values
    half = np.median(yr)
    pf = R[R > 0].sum() / max(1e-9, abs(R[R <= 0].sum()))
    g = sum(R[yr == y].sum() > 0 for y in np.unique(yr))
    eq = np.cumsum(R)
    dd = (np.maximum.accumulate(eq) - eq).max()
    print(f"  {tag:<28} N/yr={len(R)/span:5.1f} win={(R>0).mean()*100:4.1f}% PF={pf:4.2f} "
          f"meanR={R.mean():+.3f} IS/OOS={R[yr<half].mean():+.2f}/{R[yr>=half].mean():+.2f} "
          f"totR/yr={R.sum()/span:+5.1f} DD={dd:5.1f}R totR/DD={R.sum()/max(dd,1e-9):4.2f} "
          f"grn={g}/{len(np.unique(yr))}")


def null_pct(base_R, stat_obs, keep_n=None, weights=None, n=4000):
    """equal-keep random subset (hard) or random 0.5-weight assignment (soft) null."""
    out = []
    N = len(base_R)
    for _ in range(n):
        if weights is None:
            idx = np.sort(rng.choice(N, keep_n, replace=False))
            out.append(totdd(base_R[idx]))
        else:
            w = np.ones(N)
            w[rng.choice(N, weights, replace=False)] = 0.5
            out.append(totdd(base_R * w))
    out = np.array(out)
    return (out < stat_obs).mean() * 100, np.median(out)


def analyze(name, d, t, span):
    pwh = d["high"].resample("1W").max().shift(1).reindex(d.index, method="ffill")
    pdh = d["high"].resample("1D").max().shift(1).reindex(d.index, method="ffill")
    ix = d.index.get_indexer(pd.DatetimeIndex(t.time))
    t = t.assign(above_w=t.e_px.values > pwh.values[ix],
                 above_d=t.e_px.values > pdh.values[ix]).reset_index(drop=True)
    print(f"\n===== {name} ({span:.1f}yr)  n={len(t)}  "
          f"PWH-ON率={t.above_w.mean()*100:.0f}%  (PDH-ON率={t.above_d.mean()*100:.0f}% ←飽和予測の確認) =====")
    stats("base 全エントリー", t, span)
    stats("PWH上 (前週高値ブレイク圏)", t[t.above_w], span)
    stats("PWH下 (前週レンジ内)", t[~t.above_w], span)
    base_R = t.R.values
    on = t.above_w.values
    # HARD: keep only ON
    hard = totdd(base_R[on])
    p_h, med_h = null_pct(base_R, hard, keep_n=int(on.sum()))
    # SOFT: inside at half size
    soft = totdd(np.where(on, 1.0, 0.5) * base_R)
    p_s, med_s = null_pct(base_R, soft, weights=int((~on).sum()))
    print(f"  [null totR/DD] base={totdd(base_R):.2f}  HARD={hard:.2f} (null中央値{med_h:.2f}, "
          f"{p_h:.0f}%ile)  SOFT0.5={soft:.2f} (null中央値{med_s:.2f}, {p_s:.0f}%ile)   基準>=90%ile")
    # per-year ON% (does the label alternate, or is it one era?)
    ony = t.groupby(t.time.dt.year)["above_w"].mean().round(2)
    print("  年別ON%: " + " ".join(f"{y}:{v*100:.0f}" for y, v in ony.items()))


def main():
    g1 = resample(load_mt5_csv("data/vantage_xauusd_h1.csv"), "1h")
    tg = run_bo(g1, SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                       "daily_sma": 150, "daily_slope_k": 10}))
    span = (g1.index[-1] - g1.index[0]).days / 365.25
    analyze("GOLD_BO 1h (採用形)", g1, tg, span)

    b4 = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h")
    tb = run_bo(b4, SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0, "fwd": 300}))
    tbk = kama_gate_btc(tb)
    span = (b4.index[-1] - b4.index[0]).days / 365.25
    analyze("BTC_BO_KAMA 4h (採用形)", b4, tbk, span)


if __name__ == "__main__":
    main()
