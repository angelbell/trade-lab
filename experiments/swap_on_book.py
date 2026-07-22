"""The terminal spec sheet says BTCUSD swap_long = -30 with swap mode "percent of current price"
= a 30%/yr financing charge on the NOTIONAL of every long BTC position. Shorts pay nothing.

Nothing in this repo has ever charged it. The book has FOUR BTC legs, three of them long, and one of
them (btc_pull) is a 4H pullback with an RR3 target -- it holds for days. The cost scales as

    swap_R_per_day = (annual% / 365) x (price / stop_distance)

and price/stop is the position's effective leverage, which for a structural stop is large (107x on
btc15m_A). So a leg that holds for a week can pay half an R in financing.

Charge it and re-price every BTC leg and the 6-leg book. Gold's swap is a separate question -- the
spec sheet the user sent is BTCUSD only, so gold is left at zero and FLAGGED, not guessed.
Run: .venv/bin/python experiments/swap_on_book.py
"""
import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from src.data_loader import load_mt5_csv, GOLD_H1_START
from breakout_wave import run, resample
from ema_pullback import run as run_pb
from research.regime_gate_lab import CFG
from research.portfolio_kama import kama_gate_btc, cycle_gate_pull, PB
from radar_gate_race import BASE
from short_mirror_15m import invert
from book_spec_fix import book, w_trade

ROOT = "/home/angelbell/dev/auto-trade"
SIX = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]
SWAP = 30.0 / 365.0          # BTC ロング: 年率30% -> %/日。ショートは 0（仕様書どおり）


def build(swap):
    """swap=0 で現行、swap>0 で BTC ロングにだけスワップを課す。gold は不明なので 0 のまま（要確認）。"""
    out = {}
    with contextlib.redirect_stderr(io.StringIO()):
        g1 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv").loc[GOLD_H1_START:], "1h")
        t = run(g1, SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                       "daily_sma": 150, "daily_slope_k": 10}))
        out["gold_bo"] = pd.Series(t["R"].values, index=pd.DatetimeIndex(t["time"]))

        b4 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h")
        t = kama_gate_btc(run(b4, SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0,
                                                     "fwd": 300, "swap_pct": swap})))
        out["btc_bo_kama"] = pd.Series(t["R"].values, index=pd.DatetimeIndex(t["time"]))

        t = cycle_gate_pull(run_pb(b4, "long", SimpleNamespace(**{**PB, "csv": "x", "tf": "4h",
                                                                  "swap_pct": swap}), 0.0))
        out["btc_pull"] = pd.Series(t["R"].values, index=pd.DatetimeIndex(t["time"]))

        g15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g15, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10, "ext_cap": 8.0,
                                        "pullback_frac": 0.25, "fill_win": 200}))
        out["gold15m"] = pd.Series(t["R"].values - 0.30 / t["risk"].values,
                                   index=pd.DatetimeIndex(t["time"]))

        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3, "rr": 4.5, "fill_win": 200,
                                        "swap_pct": swap}))
        pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
        w = np.where(t["e_px"].values > pdh[d15.index.get_indexer(t["time"])], 1.0, 0.5)
        out["btc15m_L"] = pd.Series(t["R"].values * w - 15.0 / (t["risk"].values / w),
                                    index=pd.DatetimeIndex(t["time"]))

        inv = invert(d15); C = 2 * d15["high"].max()
        t = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3, "rr": 4.5,
                                        "fill_win": 200}))            # ショートはスワップ 0
        pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
        m = (C - t["e_px"].values) < pdl[d15.index.get_indexer(t["time"])]
        out["btc15m_S"] = pd.Series((t["R"].values - 15.0 / t["risk"].values)[m],
                                    index=pd.DatetimeIndex(t["time"])[m])
    return out


def main():
    pf = lambda x: x[x > 0].sum() / abs(x[x <= 0].sum())
    a, b = build(0.0), build(SWAP)
    print("BTC ロングに年率30%のスワップを課す（ショート=0、gold=不明のため 0 のまま・要確認）\n")
    print(f"  {'leg':<14}{'保有(日)中央値':>14}{'meanR(現行)':>13}{'meanR(swap)':>13}"
          f"{'差':>9}{'PF(現行)':>10}{'PF(swap)':>10}")
    for k in SIX:
        if k in ("gold_bo", "gold15m", "btc15m_S"):
            note = "（BTC ロングでない）"
            print(f"  {k:<14}{'—':>14}{a[k].mean():>+13.3f}{b[k].mean():>+13.3f}"
                  f"{0.0:>+9.3f}{pf(a[k].values):>10.2f}{pf(b[k].values):>10.2f}   {note}")
            continue
        d = b[k].mean() - a[k].mean()
        print(f"  {k:<14}{'':>14}{a[k].mean():>+13.3f}{b[k].mean():>+13.3f}"
              f"{d:>+9.3f}{pf(a[k].values):>10.2f}{pf(b[k].values):>10.2f}"
              + ("   🚨" if abs(d) > 0.15 else ""))
    print()
    for tag, L in (("現行（スワップ未計上）", a), ("**スワップ込み（正しい）**", b)):
        c, dd, r, n = book(L, SIX)
        print(f"  {tag:<28} ブック: CAGR {c:+.1f}%  maxDD {dd:.2f}%  **CAGR/DD {r:.2f}**")


if __name__ == "__main__":
    main()
