"""The cycle screen's by-product: trades taken while BTC is SNAPPING BACK toward its all-time high
are dead weight (top quartile of 4-week drawdown-healing: meanR +0.011, PF 1.02, 25% of trades).
Two questions, in order:

  Q1  IS IT EVEN A CYCLE VARIABLE?  "the drawdown-from-ATH improved by X% in 4 weeks" is, whenever
      no new ATH is printed, almost exactly "the 4-week return". If corr(heal, ret_4w) ~ 1.0 then
      this is not a cycle finding at all -- it is a re-discovery of OVER-EXTENSION (gold 15m's
      ext-cap, btc_pull's 30wMA cycle gate). Report both and use the simpler one if they tie.
  Q2  DOES DOWN-WEIGHTING IT SURVIVE THE BOOK?  Same discipline that just killed HH4H sizing: the
      leg is not the arbiter. Weight the top-quartile trades by w in {0, 0.25, 0.5, 0.75} and read
      the 6-leg CAGR/DD.  PASS = book CAGR/DD > 12.03 (the adopted RR4.5 leg's book).

Threshold is fixed on the IS half and applied throughout (no in-sample threshold shopping).
All context is computed on the PRIOR COMPLETED daily bar.
Run: .venv/bin/python scratchpad/book_healing_veto.py
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
from breakout_wave import run, resample
from short_mirror_15m import invert
from book_hh4h_weight_sweep import book, leg_stats, ROOT, NEW


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        legs = {k: pd.Series(t.R.values, index=pd.DatetimeIndex(t.time)) for k, t in get_legs().items()}
        g = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                      "ext_cap": 8.0, "pullback_frac": 0.25}))
        legs["gold15m"] = pd.Series(t["R"].values - 0.3 / t["risk"].values,
                                    index=pd.DatetimeIndex(t["time"]))
        full = load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv")
        b = full.loc["2018-10-01":]
        d15 = resample(b, "15min")
        inv = invert(d15); C = 2 * d15["high"].max()
        ts_ = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3}))
        Rs = ts_["R"].values - 15.0 / ts_["risk"].values
        pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
        mS = (C - ts_["e_px"].values) < pdl[d15.index.get_indexer(ts_["time"])]
        legs["btc15m_S"] = pd.Series(Rs[mS], index=pd.DatetimeIndex(ts_["time"])[mS])

        tL = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                         "pullback_frac": 0.3, "rr": 4.5}))
        Rn = tL["R"].values - 15.0 / tL["risk"].values
        ei = d15.index.get_indexer(tL["time"])
        idx = pd.DatetimeIndex(tL["time"])
        pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
        base_w = np.where(tL["e_px"].values > pdh[ei], 1.0, 0.5)

        # --- the two candidate context variables (prior completed daily bar) ---
        dhi = full["high"].resample("1D").max().cummax()
        dcl = full["close"].resample("1D").last()
        dd_ath = (dcl / dhi - 1.0).shift(1)
        heal = (dd_ath - dd_ath.shift(28)).reindex(d15.index, method="ffill").values
        ret4w = (dcl / dcl.shift(28) - 1.0).shift(1).reindex(d15.index, method="ffill").values

    H, R4 = heal[ei], ret4w[ei]
    ok = np.isfinite(H) & np.isfinite(R4)
    print(f"leg n={len(Rn)}")
    print(f"Q1  corr(4-week healing, 4-week return) = {np.corrcoef(H[ok], R4[ok])[0,1]:+.3f}"
          f"   (~1.0 => this is NOT a cycle variable, it is over-extension)")

    half = idx[len(idx) // 2]
    for nm, V in (("healing (DD-from-ATH, 4w)", H), ("plain 4-week return", R4)):
        thr = np.nanquantile(V[idx < half], 0.75)          # IS top quartile, then applied throughout
        hot = np.isfinite(V) & (V >= thr)
        sub = Rn[hot] * base_w[hot]
        pf = sub[sub > 0].sum() / abs(sub[sub <= 0].sum()) if (sub <= 0).any() else np.nan
        print(f"\n--- {nm} ---   IS-75th threshold = {thr:+.3f}   flagged {100*hot.mean():.0f}% of trades")
        print(f"    flagged trades: n={hot.sum()}  meanR {sub.mean():+.3f}  PF {pf:.2f}  "
              f"totR {sub.sum():+.1f}   |   rest: meanR "
              f"{(Rn[~hot]*base_w[~hot]).mean():+.3f}")
        print(f"    {'weight on the flagged trades':<30}{'leg PF':>8}{'leg DD':>8}{'leg C/DD':>10}"
              f"{'BOOK C/DD':>11}{'book DD':>9}")
        for w in (1.0, 0.75, 0.5, 0.25, 0.0):
            wv = base_w * np.where(hot, w, 1.0)
            s = pd.Series(Rn * wv, index=idx)
            s = s[s != 0] if w == 0.0 else s
            c, d, cd, k1, k2 = book(legs, s)
            lpf, ld, lcd = leg_stats(s)
            flag = "  <-- PASS" if cd > 12.03 else ""
            tag = "1.0 (= the adopted leg)" if w == 1.0 else f"{w}"
            print(f"    {tag:<30}{lpf:>8.2f}{ld:>7.1f}%{lcd:>10.2f}{cd:>11.2f}{d:>8.1f}%{flag}")


if __name__ == "__main__":
    main()
