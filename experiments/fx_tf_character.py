"""fx_tf_character.py -- operationalize the user's claim "トレンド銘柄じゃなくても、下足レベルでは
トレンドがある" (even non-trend instruments trend at LOWER timeframes). D2's character screen was
daily-only; this walks the full TF ladder.

GRID: 6 FX pairs (eurusd gbpusd audusd nzdusd usdcad usdjpy) x TF {15min, 1h, 4h, 1d}
      x era {FULL(2000-), 2018-}.
CONTROLS: XAUUSD + BTCUSD at the same TFs (positive controls -- assets where the book's trend
      methods DO work; BTC starts 2017 so its FULL == its own span).

METRICS (reused verbatim -- research.regime_statedet.hurst_rs / variance_ratio,
research.regime_gate_lab.er, d2_fx_reexam's trend% construction):
  Hurst (R/S, log-returns)  VR10 / VR20 (Lo-MacKinlay)  ER(20) mean  trend% (close>SMA200 & SMA200 rising)
  + run-length diagnostic: median / p90 of same-sign close-to-close run lengths (bars).

WEEKEND-GAP GUARD: intraday log-returns whose bar-to-bar time delta exceeds 4x the nominal TF
spacing are DROPPED before Hurst/VR/run-length (a Fri->Mon gap is one giant pseudo-return that
inflates VR at low TFs; ER/trend% use the close series as-is, they are level-based).

Questions answered:
  Q1 does any pair x low-TF approach gold/BTC same-TF persistence?
  Q2 is any pair MORE persistent at low TF than at its own daily (the user's claim)?
  Q3 which cells (if any) are candidate hunting grounds?

Run: .venv/bin/python experiments/fx_tf_character.py           (full grid, tee to out_fx_tf_character.txt)
     .venv/bin/python experiments/fx_tf_character.py --smoke   (eurusd+xauusd, 1h+1d only)
"""
import os, sys, io, contextlib, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data_loader import load_mt5_csv
from breakout_wave import resample
from research.regime_gate_lab import er
from research.regime_statedet import hurst_rs, variance_ratio

PAIRS = ["eurusd", "gbpusd", "audusd", "nzdusd", "usdcad", "usdjpy"]
CONTROLS = ["xauusd", "btcusd"]
TFS = [("15min", "m15", pd.Timedelta("15min")), ("1h", "h1", pd.Timedelta("1h")),
       ("4h", "h4", pd.Timedelta("4h")), ("1d", "d1", pd.Timedelta("1D"))]
ERAS = [("FULL", None), ("2018-", "2018-01-01")]


def load_tf(sym, tf_label, file_suffix):
    """Prefer the native file; fall back to resampling the finest available."""
    path = os.path.join(ROOT, "data", f"vantage_{sym}_{file_suffix}.csv")
    with contextlib.redirect_stderr(io.StringIO()):
        if os.path.exists(path):
            return load_mt5_csv(path), "native"
        # fall back: resample from h1 (controls have no h4/d1 files)
        base = load_mt5_csv(os.path.join(ROOT, "data", f"vantage_{sym}_h1.csv"))
    if tf_label == "1h":
        return base, "h1"
    return resample(base, "1D" if tf_label == "1d" else tf_label), "resampled(h1)"


def clean_logret(close, nominal):
    """log-returns with weekend/holiday-gap returns dropped (delta > 4x nominal spacing)."""
    lr = np.log(close).diff()
    dt = close.index.to_series().diff()
    keep = dt <= 4 * nominal
    return lr[keep].dropna().values


def run_lengths(r):
    """median/p90 of same-sign run lengths of the (gap-cleaned) return series."""
    s = np.sign(r)
    s = s[s != 0]
    if len(s) < 10:
        return np.nan, np.nan
    change = np.where(np.diff(s) != 0)[0]
    lens = np.diff(np.concatenate([[-1], change, [len(s) - 1]]))
    return float(np.median(lens)), float(np.percentile(lens, 90))


def cell(close, nominal):
    close = close.dropna()
    if len(close) < 300:
        return None
    r = clean_logret(close, nominal)
    if len(r) < 300:
        return None
    H = hurst_rs(r)
    VR10 = variance_ratio(r, 10)
    VR20 = variance_ratio(r, 20)
    ER = er(close, 20).mean()
    s200 = close.rolling(200).mean()
    trend = ((close > s200) & (s200 > s200.shift(20))).mean()
    rl_med, rl_p90 = run_lengths(r)
    return dict(n=len(close), H=H, VR10=VR10, VR20=VR20, ER=ER, trend=trend,
                rl_med=rl_med, rl_p90=rl_p90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    syms = (["eurusd"], ["xauusd"]) if a.smoke else (PAIRS, CONTROLS)
    tfs = [t for t in TFS if t[0] in ("1h", "1d")] if a.smoke else TFS

    print("=== TF-LADDER trend-CHARACTER screen (metrics reused from regime_statedet/regime_gate_lab; "
          "weekend-gap returns dropped for Hurst/VR/run-length) ===")
    hdr = (f"  {'sym':<8}{'tf':<7}{'era':<7}{'src':<14}{'n':>8}{'Hurst':>7}{'VR10':>6}{'VR20':>6}"
           f"{'ER':>6}{'trend%':>7}{'runMed':>7}{'runP90':>7}")
    rows = []
    for group_tag, group in (("FX", syms[0]), ("CONTROL", syms[1])):
        print(f"\n--- {group_tag} ---")
        print(hdr)
        for sym in group:
            for tf_label, suffix, nominal in tfs:
                try:
                    d, src = load_tf(sym, tf_label, suffix)
                except FileNotFoundError:
                    print(f"  {sym:<8}{tf_label:<7} no data"); continue
                for era_tag, lo in ERAS:
                    c = d["close"] if lo is None else d["close"].loc[lo:]
                    m = cell(c, nominal)
                    if m is None:
                        continue
                    print(f"  {sym.upper():<8}{tf_label:<7}{era_tag:<7}{src:<14}{m['n']:>8}"
                          f"{m['H']:>7.2f}{m['VR10']:>6.2f}{m['VR20']:>6.2f}{m['ER']:>6.2f}"
                          f"{m['trend']*100:>6.0f}%{m['rl_med']:>7.1f}{m['rl_p90']:>7.1f}")
                    rows.append(dict(group=group_tag, sym=sym, tf=tf_label, era=era_tag, **m))

    if a.smoke:
        print("\n(smoke only)"); return

    df = pd.DataFrame(rows)
    print("\n=== Q1: best FX cell vs control at the SAME tf (Hurst / VR20, era=2018-) ===")
    for tf_label, _, _ in tfs:
        sub = df[(df.tf == tf_label) & (df.era == "2018-")]
        fx = sub[sub.group == "FX"]
        ct = sub[sub.group == "CONTROL"]
        if fx.empty or ct.empty:
            continue
        b = fx.loc[fx.H.idxmax()]
        print(f"  {tf_label:<7} best-FX {b.sym.upper()}: H={b.H:.2f} VR20={b.VR20:.2f}  |  "
              + "  ".join(f"{r.sym.upper()} H={r.H:.2f} VR20={r.VR20:.2f}" for _, r in ct.iterrows()))

    print("\n=== Q2: per pair, low-TF Hurst minus OWN daily Hurst (era=2018-; + = trendier at low TF) ===")
    for sym in PAIRS:
        sub = df[(df.sym == sym) & (df.era == "2018-")].set_index("tf")
        if "1d" not in sub.index:
            continue
        base = sub.loc["1d", "H"]
        deltas = "  ".join(f"{tf}:{sub.loc[tf,'H']-base:+.2f}" for tf, _, _ in tfs[:-1] if tf in sub.index)
        print(f"  {sym.upper():<8} daily H={base:.2f}   {deltas}")


if __name__ == "__main__":
    main()
