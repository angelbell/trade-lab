"""HH4H sizing was rejected at the book (leg CAGR/DD 1.99->3.02 but book 8.55->6.96): down-weighting
the WEAK setups to 0.25 killed exactly the trades that were decorrelated from the other legs.

Question: is there a MILDER weak-weight that keeps some of the leg's gain while leaving the book's
diversification intact?  Sweep it and let the BOOK decide.

  base   = the adopted leg: canonical breakout_wave.run, 4h-KAMA gate, pullback frac 0.30, RR4.5,
           PDH soft 0.5  -> book CAGR/DD 12.03 (this is the number to beat)
  arms   = size ladder (above 4H swing high AND above PDH = 1.0 / one = 0.5 / neither = weak_w)
           for weak_w in {0.25 .. 0.75}, plus an HH4H-only variant (ignore PDH)
Weights are applied as a post-hoc re-scaling of each trade's R -- exactly how book_integration.py
already handles PDH -- so the trade SET is untouched and this is an apples-to-apples book test.
The 4H swing high is shifted one 4H bar before being mapped to 15m (no lookahead).
PASS = book CAGR/DD > 12.03.
Run: .venv/bin/python scratchpad/book_hh4h_weight_sweep.py
"""
import os, sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from research.portfolio_kama import get_legs
from radar_gate_race import BASE
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample, swings_zigzag
from short_mirror_15m import invert
from trend_leg_aging import atr as atr_fn

ROOT = "/home/angelbell/dev/auto-trade"
OLD = ["gold_bo", "btc_bo_kama", "btc_pull"]
NEW = OLD + ["gold15m", "btc15m_L", "btc15m_S"]


def book(legs, leg_series):
    L = dict(legs); L["btc15m_L"] = leg_series
    mon = {k: s.groupby(s.index.to_period("M")).sum() for k, s in L.items()}
    st = max(s.index.min() for s in mon.values()); en = min(s.index.max() for s in mon.values())
    midx = pd.period_range(st, en, freq="M")
    M = pd.DataFrame({k: v.reindex(midx, fill_value=0.0) for k, v in mon.items()})
    sig = M.std(); w = (1.0 / sig[NEW]); w = w / w.sum() * 0.03
    port = (M[NEW] * w).sum(axis=1).values
    eq = np.cumprod(1 + port); dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
    cagr = (eq[-1] ** (12 / len(port)) - 1) * 100
    return cagr, dd, cagr / dd, M["btc15m_L"].corr(M["btc_bo_kama"]), M["btc15m_L"].corr(M["gold_bo"])


def leg_stats(s):
    R = s.values
    eq = np.cumprod(1 + 0.01 * R); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100
    pf = R[R > 0].sum() / abs(R[R <= 0].sum())
    return pf, dd, cagr / dd


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        legs = {k: pd.Series(t.R.values, index=pd.DatetimeIndex(t.time)) for k, t in get_legs().items()}
        g = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                      "ext_cap": 8.0, "pullback_frac": 0.25}))
        legs["gold15m"] = pd.Series(t["R"].values - 0.3 / t["risk"].values,
                                    index=pd.DatetimeIndex(t["time"]))
        b = load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":]
        d15 = resample(b, "15min")
        inv = invert(d15); C = 2 * d15["high"].max()
        ts_ = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3}))
        Rs = ts_["R"].values - 15.0 / ts_["risk"].values
        pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
        mS = (C - ts_["e_px"].values) < pdl[d15.index.get_indexer(ts_["time"])]
        legs["btc15m_S"] = pd.Series(Rs[mS], index=pd.DatetimeIndex(ts_["time"])[mS])

        # the adopted leg: canonical run, 4h gate, RR4.5
        tL = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                         "pullback_frac": 0.3, "rr": 4.5}))
        Rn = tL["R"].values - 15.0 / tL["risk"].values
        ei = d15.index.get_indexer(tL["time"])
        pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
        h4 = d15.resample("4h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        a4 = atr_fn(h4["high"].values, h4["low"].values, h4["close"].values)
        sw = swings_zigzag(h4["high"].values, h4["low"].values, a4, 2.0)
        sh = pd.Series(np.nan, index=h4.index)
        for (ci, pi, px, kind) in sw:
            if kind == +1:
                sh.iloc[ci] = px
        hh = sh.ffill().shift(1).reindex(d15.index, method="ffill").values     # no lookahead
        e = tL["e_px"].values
        above_pdh = e > pdh[ei]
        above_hh = np.isfinite(hh[ei]) & (e > hh[ei])
        idx = pd.DatetimeIndex(tL["time"])

    print(f"leg trades n={len(Rn)}   above PDH {100*above_pdh.mean():.0f}%   "
          f"above 4H swing high {100*above_hh.mean():.0f}%   both {100*(above_pdh&above_hh).mean():.0f}%")
    base = pd.Series(Rn * np.where(above_pdh, 1.0, 0.5), index=idx)
    c, d, cd, k1, k2 = book(legs, base)
    pf, ld, lcd = leg_stats(base)
    print(f"\n{'sizing':<34}{'leg PF':>8}{'leg DD':>8}{'leg C/DD':>10}"
          f"{'BOOK C/DD':>11}{'book DD':>9}{'corr btc_bo':>12}{'corr gold_bo':>13}")
    print(f"{'base: PDH soft 0.5 (adopted)':<34}{pf:>8.2f}{ld:>7.1f}%{lcd:>10.2f}"
          f"{cd:>11.2f}{d:>8.1f}%{k1:>+12.2f}{k2:>+13.2f}")
    for wk in (0.75, 0.6, 0.5, 0.4, 0.25):
        w = np.where(above_hh & above_pdh, 1.0, np.where(above_hh | above_pdh, 0.5, wk))
        s = pd.Series(Rn * w, index=idx)
        c, d, cd, k1, k2 = book(legs, s)
        pf, ld, lcd = leg_stats(s)
        flag = "  <-- PASS" if cd > 12.03 else ""
        print(f"{f'HH4H+PDH ladder, weak={wk}':<34}{pf:>8.2f}{ld:>7.1f}%{lcd:>10.2f}"
              f"{cd:>11.2f}{d:>8.1f}%{k1:>+12.2f}{k2:>+13.2f}{flag}")
    for wk in (0.75, 0.5):
        w = np.where(above_hh, 1.0, wk)                       # HH4H only, PDH ignored
        s = pd.Series(Rn * w, index=idx)
        c, d, cd, k1, k2 = book(legs, s)
        pf, ld, lcd = leg_stats(s)
        flag = "  <-- PASS" if cd > 12.03 else ""
        print(f"{f'HH4H only, weak={wk}':<34}{pf:>8.2f}{ld:>7.1f}%{lcd:>10.2f}"
              f"{cd:>11.2f}{d:>8.1f}%{k1:>+12.2f}{k2:>+13.2f}{flag}")


if __name__ == "__main__":
    main()
