"""pwh_adoption.py -- adoption gauntlet for the gold_bo PWH-position feature (the hump).

Stages (all cells reported, plateau default pre-declared = center of each grid):
 B. HARD window grid  : keep dist in [lo, hi] ATR, lo x hi 3x3 -- totR/DD + equal-keep null.
 C. SOFT two-tilt grid : w = (w_in if dist<0 else 1) * (0.5 if dist>X else 1)
    -- composes the two known laws (inside-range=noise down-weight, overextension roll-off)
    with ONE parameter each. w_in x X 3x3 -- totR/DD + same-structure random-weight null.
 D. Formal audit of the PRE-DECLARED default (w_in=0.5, X=10):
    - DSR at trial counts {10,30,60} on the weighted stream (psr/sr0 from overfit_audit)
    - paired block bootstrap: P(soft beats base on CAGR/DD), monthly blocks (the decisive
      test per the audit_kama_stack precedent)
    - per-year table of soft vs base.
 E. BOOK impact: 6-leg new book at total 3% inv-vol, gold_bo -> gold_bo*w. Monthly joint
    bootstrap 1yr multiplier (median/sd/p10/p90) old vs new gold leg.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_mt5_csv
from breakout_wave import run as run_bo, resample
from research.regime_gate_lab import CFG
from research.overfit_audit import psr, sr0
from research.portfolio_kama import get_legs
from radar_gate_race import BASE
from short_mirror_15m import invert

rng = np.random.default_rng(7)


def totdd(R):
    eq = np.cumsum(R)
    return R.sum() / max((np.maximum.accumulate(eq) - eq).max(), 1e-9)


def gold_leg():
    g1 = resample(load_mt5_csv("data/vantage_xauusd_h1.csv"), "1h")
    t = run_bo(g1, SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                      "daily_sma": 150, "daily_slope_k": 10}))
    atr = ta.atr(g1["high"], g1["low"], g1["close"], 14).shift(1)
    pwh = g1["high"].resample("1W").max().shift(1).reindex(g1.index, method="ffill")
    ix = g1.index.get_indexer(pd.DatetimeIndex(t.time))
    dist = (t.e_px.values - pwh.values[ix]) / atr.values[ix]
    return t.reset_index(drop=True), dist


def main():
    t, dist = gold_leg()
    R = t.R.values
    base = totdd(R)
    print(f"gold_bo n={len(t)}  base totR/DD={base:.2f}")

    print("\n-- B. HARD窓 [lo,hi]ATR グリッド (equal-keep null %ile) --")
    print(f"  {'lo\\hi':>6} " + " ".join(f"{hi:>12}" for hi in (8, 10, 12)))
    for lo in (-6, -4, -2):
        cells = []
        for hi in (8, 10, 12):
            on = (dist >= lo) & (dist <= hi)
            obs = totdd(R[on])
            nl = [totdd(R[np.sort(rng.choice(len(R), on.sum(), replace=False))]) for _ in range(2000)]
            cells.append(f"{obs:5.2f}({(np.array(nl)<obs).mean()*100:3.0f}%)")
        print(f"  {lo:>6} " + " ".join(f"{s:>12}" for s in cells))

    print("\n-- C. SOFT二段ティルト w=(w_in if dist<0)*(0.5 if dist>X) --")
    print(f"  {'w_in\\X':>7} " + " ".join(f"{X:>12}" for X in (8, 10, 12)))
    for w_in in (0.25, 0.5, 0.75):
        cells = []
        for X in (8, 10, 12):
            w = np.where(dist < 0, w_in, 1.0) * np.where(dist > X, 0.5, 1.0)
            obs = totdd(R * w)
            nl = []
            for _ in range(2000):
                nl.append(totdd(R * rng.permutation(w)))
            cells.append(f"{obs:5.2f}({(np.array(nl)<obs).mean()*100:3.0f}%)")
        print(f"  {w_in:>7} " + " ".join(f"{s:>12}" for s in cells))

    # ---- D. audit the pre-declared default ----
    w = np.where(dist < 0, 0.5, 1.0) * np.where(dist > 10, 0.5, 1.0)
    Rw = R * w
    print(f"\n-- D. 既定形 (w_in=0.5, X=10) の形式監査 --  totR/DD {base:.2f} -> {totdd(Rw):.2f}")
    for N in (10, 30, 60):
        d_, sr, g1_, g4 = psr(Rw, sr0(N, np.var([Rw.mean() / Rw.std()] ) if False else 0.02))
        print(f"  DSR@{N:<3}= {psr(Rw, sr0(N, 0.02))[0]:.3f}", end="")
    print("   (V=0.02 標準設定)")
    # paired monthly block bootstrap: P(soft > base on CAGR/DD at 1% risk)
    mon_b = pd.Series(R * 0.01, index=pd.DatetimeIndex(t.time)).groupby(
        pd.DatetimeIndex(t.time).to_period("M")).sum()
    mon_s = pd.Series(Rw * 0.01, index=pd.DatetimeIndex(t.time)).groupby(
        pd.DatetimeIndex(t.time).to_period("M")).sum()
    midx = mon_b.index.union(mon_s.index)
    A = mon_b.reindex(midx, fill_value=0).values
    B = mon_s.reindex(midx, fill_value=0).values
    def cdd(x):
        eq = np.cumprod(1 + x)
        dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max()
        yrs = len(x) / 12
        return (eq[-1] ** (1 / yrs) - 1) / max(dd, 1e-9)
    wins = 0
    for _ in range(2000):
        idx = rng.integers(0, len(A), len(A))
        wins += cdd(B[idx]) > cdd(A[idx])
    print(f"  ペア化月次ブートストラップ: P(soft優于base, CAGR/DD) = {wins/2000*100:.0f}%")
    yr = t.time.dt.year.values
    print("  年別 base->soft totR: " + "  ".join(
        f"{y}:{R[yr==y].sum():+.0f}->{Rw[yr==y].sum():+.0f}" for y in np.unique(yr)))

    # ---- E. book impact ----
    print("\n-- E. 新6レッグブック (総リスク3% inv-vol) への影響 --")
    legs = {}
    for k, tt in get_legs().items():
        legs[k] = pd.Series(tt.R.values, index=pd.DatetimeIndex(tt.time))
    legs["gold_bo"] = pd.Series(Rw, index=pd.DatetimeIndex(t.time))  # soft版に置換
    g = resample(load_mt5_csv("data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
    tg = run_bo(g, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                      "ext_cap": 8.0, "pullback_frac": 0.25}))
    legs["gold15m"] = pd.Series(tg["R"].values - 0.3 / tg["risk"].values,
                                index=pd.DatetimeIndex(tg["time"]))
    b = load_mt5_csv("data/vantage_btcusd_m15.csv")
    cnt = b.groupby(b.index.date).size()
    okd = cnt[cnt.rolling(30).median() >= 80]
    d15 = resample(b[b.index.date >= okd.index[0]], "15min")
    tb = run_bo(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3}))
    Rn = tb["R"].values - 15.0 / tb["risk"].values
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    ab = tb["e_px"].values > pdh[d15.index.get_indexer(tb["time"])]
    legs["btc15m_L"] = pd.Series(Rn * np.where(ab, 1.0, 0.5), index=pd.DatetimeIndex(tb["time"]))
    inv = invert(d15)
    ts_ = run_bo(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3}))
    Rs = ts_["R"].values - 15.0 / ts_["risk"].values
    pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
    C = 2 * d15["high"].max()
    mS = (C - ts_["e_px"].values) < pdl[d15.index.get_indexer(ts_["time"])]
    legs["btc15m_S"] = pd.Series(Rs[mS], index=pd.DatetimeIndex(ts_["time"])[mS])

    for tag, gser in [("gold_bo素", pd.Series(R, index=pd.DatetimeIndex(t.time))),
                      ("gold_bo+PWHソフト", pd.Series(Rw, index=pd.DatetimeIndex(t.time)))]:
        L = dict(legs); L["gold_bo"] = gser
        mon = {k: s.groupby(s.index.to_period("M")).sum() for k, s in L.items()}
        start = max(s.index.min() for s in mon.values())
        end = min(s.index.max() for s in mon.values())
        midx2 = pd.period_range(start, end, freq="M")
        M = pd.DataFrame({k: v.reindex(midx2, fill_value=0.0) for k, v in mon.items()})
        wgt = 1.0 / M.std(); wgt = wgt / wgt.sum() * 0.03
        port = (M * wgt).sum(axis=1).values
        mult = np.array([np.prod(1 + port[rng.integers(0, len(port), 12)]) for _ in range(4000)])
        eq = np.cumprod(1 + port)
        ddp = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
        cagr = (eq[-1] ** (12 / len(port)) - 1) * 100
        print(f"  {tag:<18} CAGR={cagr:5.1f}% maxDD={ddp:4.1f}% CAGR/DD={cagr/ddp:5.2f} | "
              f"1yr倍率 med={np.median(mult):.2f} sd={mult.std():.2f} "
              f"p10={np.percentile(mult,10):.2f} p90={np.percentile(mult,90):.2f}")


if __name__ == "__main__":
    main()
