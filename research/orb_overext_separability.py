"""orb_overext_separability.py -- THE MAIN QUESTION, rigorously: does a SEPARABLE over-extension filter
exist for H17 (gold M15 ORB + 1H trend gate)? i.e. can ANY measure of 'price already ran too far at entry'
sort H17's losers from its winners in a way that survives out-of-sample + a random-drop null?

Prior context (already falsified, two proxies): --rsi-max (inert: M15 breakout bars almost never RSI>=80)
and --box-trend-max (IS PF 1.28->1.45 but VAL 1.20->1.07 = curve-fit). max-range cap died the sealed TEST.
This script is the EXHAUSTIVE version the user asked for: a PANEL of extension definitions, the proper
random-drop null on CAGR/DD (a filter must beat n-trimming, not just raise PF), IS AND VAL, threshold
plateau. The deliverable is the negative itself -- confirming non-separability rigorously (prime directive).

Extension features at the SIGNAL bar (signed in the trade direction; higher = entered later into a move that
already ran the trade's way):
  ext4/8/12 = dir*(close - close[-N]) / ATR   (1h/2h/3h aligned run, the gold_overextension.py measure)
  rsi_al    = dir>0 ? RSI : 100-RSI            (oscillator exhaustion, the --rsi-max idea, continuous)
  dist_open = dir*(close - day_open) / ATR     (distance from the day's first bar)

Tests per feature: (A) corr(feature, R) + tertile mean-R (does high extension => worse R at all?).
(B) DECISIVE: exclude the top X% most-extended trades -> CAGR/DD vs a 2000x random-drop null keeping the
same N (>=90%ile = the SELECTION carries info, not just n-trimming), on IS AND VAL, X in {20,33,50}.
A separable filter must beat the null on IS AND repeat on VAL AND plateau across X. In-sample/val only
(sealed TEST is spent); descriptive audit.
  .venv/bin/python research/orb_overext_separability.py
"""
import os, sys, warnings
from types import SimpleNamespace
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
import research.scalp_lab as sl
from research.portfolio_kama import cagr_dd

H17 = dict(asia_start_h=0, asia_end_h=7, bo_start_h=7, bo_end_h=11, force_exit_h=20,
           rr=1.0, buf_atr=0.0, sl_buf_atr=0.0, max_range_atr=0.0, min_range_atr=0.0,
           sl_frac=1.0, rsi_max=100.0, box_trend_max=1.0, no_tp=True, fade=False, dir="both",
           htf_tf="1h", htf_ema=80, htf_slope_k=0, cost=1.4, stop_slip=0.0)
FEATS = ["ext4", "ext8", "ext12", "rsi_al", "dist_open"]


def h17_trades(split):
    s, e = sl.SPLITS[split]
    d = load_mt5_csv("data/vantage_xauusd_m5.csv").loc[s:e]
    d = d.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    p = SimpleNamespace(**H17)
    dir_, slx, tpx = sl.orb_signals(d, p)
    dir_, slx, tpx = sl.htf_trend_gate(d, dir_, slx, tpx, p)
    t = sl.backtest(d, dir_, slx, tpx, p)
    # features at the signal bar (= bar before the fill bar t_in)
    c = d["close"].values
    atr = sl.ta.atr(d["high"], d["low"], d["close"], 14).values
    rsi = sl.ta.rsi(d["close"], 14).values
    day = d.index.normalize()
    dayopen = d["open"].groupby(day).transform("first").values
    pos = d.index.get_indexer(pd.DatetimeIndex(t.t_in))
    sig = pos - 1
    rows = []
    for k in range(len(t)):
        i = sig[k]
        if i < 12 or pos[k] < 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
            rows.append([np.nan] * len(FEATS)); continue
        dr = t.dir.iloc[k]
        rows.append([
            dr * (c[i] - c[i - 4]) / atr[i], dr * (c[i] - c[i - 8]) / atr[i],
            dr * (c[i] - c[i - 12]) / atr[i],
            rsi[i] if dr > 0 else 100 - rsi[i],
            dr * (c[i] - dayopen[i]) / atr[i],
        ])
    f = pd.DataFrame(rows, columns=FEATS, index=t.index)
    t = pd.concat([t, f], axis=1).dropna(subset=FEATS)
    one_r = abs(t.loc[t.pips < 0, "pips"].mean())
    t["R"] = t.pips / one_r
    t["time"] = t.t_in.dt.tz_localize(None)
    return t


def null_pctile(t, keep_n, real_cdd, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.empty(iters)
    for j in range(iters):
        idx = np.sort(rng.choice(len(t), keep_n, replace=False))
        arr[j] = cagr_dd(t.iloc[idx][["time", "R"]])[2]
    return (arr < real_cdd).mean() * 100


def audit(t, label):
    base_cdd = cagr_dd(t[["time", "R"]])[2]
    print(f"\n#### {label}: n={len(t)} base PF={t[t.R>0].R.sum()/abs(t[t.R<0].R.sum()):.2f} "
          f"meanR={t.R.mean():+.3f} CAGR/DD={base_cdd:.2f} ####")
    print("  (A) does extension separate at all?  corr(feat,R) + tertile mean-R (lo/mid/hi extension)")
    for f in FEATS:
        q = t[f].rank(pct=True)
        lo, mid, hi = t.R[q <= .33].mean(), t.R[(q > .33) & (q <= .66)].mean(), t.R[q > .66].mean()
        print(f"    {f:<10} corr={t[f].corr(t.R):+.3f}   meanR lo={lo:+.2f} mid={mid:+.2f} hi={hi:+.2f}"
              f"   {'(hi worse -> separable?)' if hi < lo - 0.05 else '(no separation)'}")
    print("  (B) exclude top-X% most-extended -> CAGR/DD vs random-drop null (>=90%ile = real selection):")
    print(f"    {'feature':<10}{'X%':>5}{'kept':>6}{'PF':>6}{'CAGR/DD':>9}{'vs random-drop':>17}")
    for f in FEATS:
        for X in (0.20, 0.33, 0.50):
            thr = t[f].quantile(1 - X)
            kept = t[t[f] < thr]
            if len(kept) < 25:
                continue
            cdd = cagr_dd(kept[["time", "R"]])[2]
            pf = kept[kept.R > 0].R.sum() / abs(kept[kept.R < 0].R.sum())
            pct = null_pctile(t, len(kept), cdd)
            flag = "PASS" if pct >= 90 else ("weak" if pct >= 75 else "n-trim")
            print(f"    {f:<10}{int(X*100):>4}%{len(kept):>6}{pf:>6.2f}{cdd:>9.2f}{pct:>13.0f}%ile {flag}")


def main():
    print("=== H17 over-extension SEPARABILITY audit (the main question: does a usable filter exist?) ===")
    tis = h17_trades("is")
    tval = h17_trades("val")
    audit(tis, "IS 2018-06..2022")
    audit(tval, "VAL 2023..2024")
    print("\n  verdict: a SEPARABLE filter must (1) show hi-extension worse meanR, (2) beat the random-drop")
    print("  null >=90%ile on CAGR/DD, (3) do BOTH on IS AND VAL, (4) plateau across X. Anything that only")
    print("  shines on IS or only n-trims = NON-separable = the over-extension lever does not exist for H17.")


if __name__ == "__main__":
    main()
