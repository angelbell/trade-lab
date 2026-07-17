"""The two spec sheets, implemented exactly. Nothing in this repo has ever charged financing.

BTCUSD   swap mode = PERCENT of current price.  long -30  short 0.
         -> cost per day, in R = (30/365/100) x (entry_price / stop_distance)
            i.e. it scales with the position's leverage (price/stop), which for a structural stop
            on a 15m chart is about 107x. Shorts pay nothing.

XAUUSD+  swap mode = POINTS.  long -78.29  short +29.49.  point = 0.01, contract = 100 oz.
         -> charge per lot per day = 78.29 x 0.01 x 100 = $78.29  ->  $0.7829 per OUNCE per day.
            This is a FIXED DOLLAR amount, so as a fraction of price it FALLS as gold rises
            (22%/yr at $1,300, 11%/yr at $2,600). Shorts RECEIVE $0.2949/oz/day.
         -> cost per day, in R = 0.7829 / stop_distance($/oz)   (no price term)

Both charge 7 units per week (BTC: triple Friday; gold: triple Wednesday), so a flat per-calendar-day
rate is the right average. Applied per trade with its own hold time and its own stop distance.
Gold's spread+commission is ALREADY over-charged in the backtest (canon $0.30-2.60/oz vs a real
$0.15-0.35), which offsets part of this -- reported separately so the two are not confused.
Run: .venv/bin/python scratchpad/swap_canon.py
"""
import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from src.data_loader import load_mt5_csv, GOLD_H1_START
from breakout_wave import run, resample
from ema_pullback import run as run_pb
from research.regime_gate_lab import CFG
from research.portfolio_kama import kama_gate_btc, cycle_gate_pull, PB
from radar_gate_race import BASE
from short_mirror_15m import invert
from book_spec_fix import book

ROOT = "/home/angelbell/dev/auto-trade"
SIX = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]
BTC_LONG_PCT_YR = 30.0                  # 現在価格に対する年率%
GOLD_LONG_USD_OZ_DAY = 0.7829           # 78.29 points x 0.01 x 100oz / 100oz
GOLD_SHORT_USD_OZ_DAY = -0.2949         # ショートは受け取る（符号: コストとして負）


def legs():
    """(R_現行, 保有日数, 損切り幅, 建値, スワップの種類) を全レッグぶん。"""
    out = {}
    with contextlib.redirect_stderr(io.StringIO()):
        g1 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv").loc[GOLD_H1_START:], "1h")
        t = run(g1, SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                       "daily_sma": 150, "daily_slope_k": 10}))
        out["gold_bo"] = (t["R"].values, t["hold"].values, t["risk"].values, t["e_px"].values,
                          pd.DatetimeIndex(t["time"]), "gold_long")

        b4 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h")
        t = kama_gate_btc(run(b4, SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0,
                                                     "fwd": 300})))
        out["btc_bo_kama"] = (t["R"].values, t["hold"].values, t["risk"].values, t["e_px"].values,
                              pd.DatetimeIndex(t["time"]), "btc_long")

        t = cycle_gate_pull(run_pb(b4, "long", SimpleNamespace(**{**PB, "csv": "x", "tf": "4h"}), 0.0))
        out["btc_pull"] = (t["R"].values, t["hold"].values, t["risk"].values, t["e_px"].values,
                           pd.DatetimeIndex(t["time"]), "btc_long")

        g15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g15, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10, "ext_cap": 8.0,
                                        "pullback_frac": 0.25, "fill_win": 200}))
        out["gold15m"] = (t["R"].values - 0.30 / t["risk"].values, t["hold"].values,
                          t["risk"].values, t["e_px"].values, pd.DatetimeIndex(t["time"]), "gold_long")

        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3, "rr": 4.5, "fill_win": 200}))
        pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
        w = np.where(t["e_px"].values > pdh[d15.index.get_indexer(t["time"])], 1.0, 0.5)
        rk = t["risk"].values / w
        out["btc15m_L"] = (t["R"].values * w - 15.0 / rk, t["hold"].values, rk, t["e_px"].values,
                           pd.DatetimeIndex(t["time"]), "btc_long")

        inv = invert(d15); C = 2 * d15["high"].max()
        t = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3, "rr": 4.5,
                                        "fill_win": 200}))
        pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
        m = (C - t["e_px"].values) < pdl[d15.index.get_indexer(t["time"])]
        out["btc15m_S"] = ((t["R"].values - 15.0 / t["risk"].values)[m], t["hold"].values[m],
                           t["risk"].values[m], (C - t["e_px"].values)[m],
                           pd.DatetimeIndex(t["time"])[m], "btc_short")
    return out


def swap_R(hold, risk, px, kind):
    if kind == "btc_long":
        return (BTC_LONG_PCT_YR / 365.0 / 100.0) * (px / risk) * hold
    if kind == "btc_short":
        return np.zeros_like(hold)                      # BTC ショートはスワップ 0
    if kind == "gold_long":
        return GOLD_LONG_USD_OZ_DAY * hold / risk
    return GOLD_SHORT_USD_OZ_DAY * hold / risk


def main():
    L = legs()
    pf = lambda x: x[x > 0].sum() / abs(x[x <= 0].sum())
    print("BTC: 年率30%×(価格/損切り幅)   gold: $0.7829/oz/日 ÷ 損切り幅   ショート: BTC=0 / gold=受取\n")
    print(f"  {'leg':<14}{'向き':<10}{'保有(日)中央値':>13}{'保有(日)平均':>12}{'損切り中央':>11}"
          f"{'スワップ/本(平均)':>16}{'meanR(現行)':>12}{'meanR(込み)':>12}{'PF(現行)':>9}{'PF(込み)':>9}")
    A, B = {}, {}
    for k in SIX:
        R, hold, risk, px, ti, kind = L[k]
        s = swap_R(hold, risk, px, kind)
        Rn = R - s
        A[k] = pd.Series(R, index=ti); B[k] = pd.Series(Rn, index=ti)
        d = "ロング" if "long" in kind else "ショート"
        unit = "$/oz" if "gold" in kind else "$"
        print(f"  {k:<14}{d:<10}{np.median(hold):>13.2f}{hold.mean():>12.2f}"
              f"{np.median(risk):>9,.0f}{unit:<2}{s.mean():>+16.3f}{R.mean():>+12.3f}{Rn.mean():>+12.3f}"
              f"{pf(R):>9.2f}{pf(Rn):>9.2f}" + ("  🚨" if s.mean() > 0.12 else ""))
    print()
    for tag, X in (("現行（スワップ未計上）", A), ("**スワップ込み（正典）**", B)):
        c, dd, r, n = book(X, SIX)
        print(f"  {tag:<28} ブック: CAGR {c:+.1f}%  maxDD {dd:.2f}%  **CAGR/DD {r:.2f}**")

    print("\n\n  参考: gold のスプレッド+手数料は、バックテストで既に過大計上されている")
    print("    実勢 = スプレッド($0.10〜0.30) + 手数料($0.058/oz RT) = **$0.15〜0.35/oz**")
    print("    gold_bo のバックテスト = 価格の0.1% = **$2.6/oz**（金$2,600時）＝ **7〜17倍の過大計上**")
    print("    gold15m のバックテスト = **$0.30/oz** ＝ 実勢の上限どおり（妥当）")
    R, hold, risk, px, ti, _ = L["gold_bo"]
    over = (0.001 * px - 0.25) / risk                   # 過大計上ぶん（実勢を$0.25と置く）
    print(f"    → gold_bo は 1トレードあたり **{over.mean():+.3f}R** を余計に払わされている")
    print(f"       （スワップ {swap_R(hold,risk,px,'gold_long').mean():+.3f}R と相殺すると "
          f"実質 {swap_R(hold,risk,px,'gold_long').mean()-over.mean():+.3f}R）")


if __name__ == "__main__":
    main()
