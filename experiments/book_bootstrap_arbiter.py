"""Apply today's own rule to today's own book verdicts.

All of today's book judgements (HH4H sizing REJECTED 12.03 -> 8.6-9.5; the 4-week-extension veto
"improves" 12.03 -> 13.26) were read off a SINGLE PATH of the book's monthly returns. That is
exactly the sin the weekly-ER gate committed at the leg level -- and the rule we then wrote into
CLAUDE.md says: pair it with a circular BLOCK bootstrap, because a single path's maxDD is noisy.

So: rebuild the 6-leg book for each btc15m_L variant, and bootstrap the BOOK's monthly returns
(paired, same resampled months for every arm) over block lengths 1/3/6/12 months.
  base   = the adopted leg (4h gate, RR4.5, PDH soft 0.5)             -> book CAGR/DD 12.03
  hh4h   = the rejected sizing (4H-swing-high + PDH ladder, weak 0.25) -> 9.46 on the single path
  veto50 = down-weight to 0.5 the trades entered after a big 4-week run (IS-75th pct)
  veto0  = skip them entirely                                          -> 13.26 on the single path
Reported: median book CAGR/DD per arm + P(arm beats base).  A verdict that flips sign across block
lengths, or sits near 50%, is noise -- not a finding.
Run: .venv/bin/python experiments/book_bootstrap_arbiter.py
"""
import os, sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from research.portfolio_kama import get_legs
from radar_gate_race import BASE
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample, swings_zigzag
from short_mirror_15m import invert
from trend_leg_aging import atr as atr_fn

ROOT = "/home/angelbell/dev/auto-trade"
OLD = ["gold_bo", "btc_bo_kama", "btc_pull"]
NEW = OLD + ["gold15m", "btc15m_L", "btc15m_S"]
NDRAW = 2000


def book_monthly(legs, leg_series):
    """the book's monthly return series at inv-vol weights, total risk 3%."""
    L = dict(legs); L["btc15m_L"] = leg_series
    mon = {k: s.groupby(s.index.to_period("M")).sum() for k, s in L.items()}
    st = max(s.index.min() for s in mon.values()); en = min(s.index.max() for s in mon.values())
    midx = pd.period_range(st, en, freq="M")
    M = pd.DataFrame({k: v.reindex(midx, fill_value=0.0) for k, v in mon.items()})
    sig = M.std(); w = (1.0 / sig[NEW]); w = w / w.sum() * 0.03
    return (M[NEW] * w).sum(axis=1)


def cdd(port, months):
    eq = np.cumprod(1 + port)
    dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max()
    if dd <= 0: return np.nan
    return ((eq[-1] ** (12 / months) - 1)) / dd


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        legs = {k: pd.Series(t.R.values, index=pd.DatetimeIndex(t.time)) for k, t in get_legs().items()}
        g = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                      "ext_cap": 8.0, "pullback_frac": 0.25}))
        legs["gold15m"] = pd.Series(t["R"].values - 0.3 / t["risk"].values,
                                    index=pd.DatetimeIndex(t["time"]))
        full = load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv")
        d15 = resample(full.loc["2018-10-01":], "15min")
        inv = invert(d15); C = 2 * d15["high"].max()
        ts_ = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3}))
        Rs = ts_["R"].values - 15.0 / ts_["risk"].values
        pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
        mS = (C - ts_["e_px"].values) < pdl[d15.index.get_indexer(ts_["time"])]
        legs["btc15m_S"] = pd.Series(Rs[mS], index=pd.DatetimeIndex(ts_["time"])[mS])

        tL = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                         "pullback_frac": 0.3, "rr": 4.5}))
        Rn = tL["R"].values - 15.0 / tL["risk"].values
        ei = d15.index.get_indexer(tL["time"]); idx = pd.DatetimeIndex(tL["time"])
        pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
        base_w = np.where(tL["e_px"].values > pdh[ei], 1.0, 0.5)
        h4 = d15.resample("4h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        a4 = atr_fn(h4["high"].values, h4["low"].values, h4["close"].values)
        sh = pd.Series(np.nan, index=h4.index)
        for (ci, pi, px, kind) in swings_zigzag(h4["high"].values, h4["low"].values, a4, 2.0):
            if kind == +1:
                sh.iloc[ci] = px
        hh = sh.ffill().shift(1).reindex(d15.index, method="ffill").values
        e = tL["e_px"].values
        a_pdh = e > pdh[ei]; a_hh = np.isfinite(hh[ei]) & (e > hh[ei])
        dcl = full["close"].resample("1D").last()
        ret4w = (dcl / dcl.shift(28) - 1.0).shift(1).reindex(d15.index, method="ffill").values[ei]

    half = idx[len(idx) // 2]
    thr = np.nanquantile(ret4w[idx < half], 0.75)
    hot = np.isfinite(ret4w) & (ret4w >= thr)

    arms = {
        "base (adopted RR4.5)": pd.Series(Rn * base_w, index=idx),
        "hh4h ladder w=0.25": pd.Series(Rn * np.where(a_hh & a_pdh, 1.0,
                                        np.where(a_hh | a_pdh, 0.5, 0.25)), index=idx),
        "ext-veto w=0.5": pd.Series(Rn * base_w * np.where(hot, 0.5, 1.0), index=idx),
        "ext-veto w=0.0": pd.Series(Rn * base_w * np.where(hot, 0.0, 1.0), index=idx),
    }
    P = {k: book_monthly(legs, v) for k, v in arms.items()}
    idxm = None
    for v in P.values():
        idxm = v.index if idxm is None else idxm.union(v.index)
    W = {k: v.reindex(idxm, fill_value=0.0).values for k, v in P.items()}
    m = len(idxm)
    print(f"book months = {m}\n")
    print(f"{'arm':<24}{'single-path book CAGR/DD':>26}")
    for k in arms:
        print(f"{k:<24}{cdd(W[k], m):>26.2f}")

    rng = np.random.default_rng(20260713)
    print(f"\nCIRCULAR BLOCK bootstrap of the BOOK's monthly returns ({NDRAW} draws, paired)")
    print(f"  {'block':<8}" + "".join(f"{k:>22}" for k in arms))
    for blk in (1, 3, 6, 12):
        nb = int(np.ceil(m / blk))
        D = {k: [] for k in arms}
        for _ in range(NDRAW):
            st = rng.integers(0, m, nb)
            k_ = np.concatenate([(np.arange(s, s + blk) % m) for s in st])[:m]
            for k in arms:
                D[k].append(cdd(W[k][k_], m))
        med = {k: np.nanmedian(D[k]) for k in arms}
        base = np.array(D["base (adopted RR4.5)"])
        row = []
        for k in arms:
            a = np.array(D[k])
            win = np.nanmean(a > base) * 100
            row.append(f"{med[k]:.2f} (P {win:.0f}%)")
        print(f"  {f'{blk}mo':<8}" + "".join(f"{r:>22}" for r in row))
    print("\n  P = P(this arm's book CAGR/DD > base's) on the same resampled months.")
    print("  ~50% = indistinguishable from the base. A real change is consistent across block lengths.")


if __name__ == "__main__":
    main()
