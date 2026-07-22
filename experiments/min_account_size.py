"""Can a JPY 100,000 account actually TRADE this book?

The book's weights (0.32%-0.72% of the account per trade) assume you can size a position freely.
You cannot. Vantage's minimum is 0.01 lot, and a leg's stop distance is set by market structure,
not by your account. So the smallest position you are allowed to open already risks a fixed number
of dollars -- and if that exceeds w_leg * account, the account is too small to trade the leg at its
intended weight. You would be forced to over-risk (or skip the trade).

Contract sizes (Vantage RAW/ECN):
  XAUUSD+ : 1 lot = 100 oz   -> 0.01 lot = 1 oz.    $1/oz move = $1.00 per 0.01 lot
  BTCUSD  : 1 lot = 1 BTC    -> 0.01 lot = 0.01 BTC. $1 move  = $0.01 per 0.01 lot
  (gold is CONFIRMED by the measured commission: $3/lot/side / 100 oz = $0.03/oz/side = $0.06 RT,
   which matches CLAUDE.md. BTC's contract size is the standard assumption and is FLAGGED below --
   it must be checked in the terminal before acting on any of this.)

For each leg: min-lot dollar risk = stop_distance x $-per-point-per-0.01-lot.
Required account  = min-lot dollar risk / w_leg.
Reported at the median and the 90th percentile of the stop distance, and separately for the RECENT
era (2025+), because BTC's stop distances scale with its price -- what mattered in 2019 is not what
will matter next year.
Run: .venv/bin/python experiments/min_account_size.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
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
from book_spec_fix import build, w_trade

ROOT = "/home/angelbell/dev/auto-trade"
SIX = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]
USDJPY = 155.0        # 概算。円口座なので $ 建てのリスクを円に直すのに使う

# $ P&L per 1.0 price-unit move, at the MINIMUM lot (0.01)
PER_POINT_MIN_LOT = {"gold_bo": 1.00, "gold15m": 1.00,          # 0.01 lot = 1 oz
                     "btc_bo_kama": 0.01, "btc_pull": 0.01,     # 0.01 lot = 0.01 BTC
                     "btc15m_L": 0.01, "btc15m_S": 0.01}


def stop_distances():
    """price-unit stop distance (= the leg's 'risk' column) for every trade, per leg."""
    out = {}
    with contextlib.redirect_stderr(io.StringIO()):
        g1 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv").loc[GOLD_H1_START:], "1h")
        t = run(g1, SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                       "daily_sma": 150, "daily_slope_k": 10}))
        out["gold_bo"] = pd.Series(t["risk"].values, index=pd.DatetimeIndex(t["time"]))

        b4 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h")
        t = run(b4, SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0, "fwd": 300}))
        t = kama_gate_btc(t)
        out["btc_bo_kama"] = pd.Series(t["risk"].values, index=pd.DatetimeIndex(t["time"]))

        t = run_pb(b4, "long", SimpleNamespace(**{**PB, "csv": "x", "tf": "4h"}), 0.0)
        t = cycle_gate_pull(t)
        out["btc_pull"] = pd.Series(t["risk"].values, index=pd.DatetimeIndex(t["time"]))

        g15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g15, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                        "ext_cap": 8.0, "pullback_frac": 0.25}))
        out["gold15m"] = pd.Series(t["risk"].values, index=pd.DatetimeIndex(t["time"]))

        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3, "rr": 4.5}))
        out["btc15m_L"] = pd.Series(t["risk"].values, index=pd.DatetimeIndex(t["time"]))

        inv = invert(d15); C = 2 * d15["high"].max()
        t = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3, "rr": 4.5}))
        pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
        m = (C - t["e_px"].values) < pdl[d15.index.get_indexer(t["time"])]
        out["btc15m_S"] = pd.Series(t["risk"].values[m], index=pd.DatetimeIndex(t["time"])[m])
    return out


def main():
    L = build("2018-01-01", False)
    w = w_trade(L, SIX)
    D = stop_distances()

    for era, lo in (("全期間", "2019"), ("直近（2025年以降）", "2025")):
        print(f"\n=== {era} ===")
        print(f"{'leg':<13}{'重み%':>7}{'損切り幅(中央値)':>18}{'0.01ロットの$リスク':>20}"
              f"{'必要な口座(中央値)':>20}{'必要な口座(90%点)':>20}")
        worst = 0.0
        for k in SIX:
            s = D[k][D[k].index >= lo]
            if len(s) == 0:
                continue
            med, p90 = np.median(s.values), np.percentile(s.values, 90)
            usd_med = med * PER_POINT_MIN_LOT[k]
            usd_p90 = p90 * PER_POINT_MIN_LOT[k]
            acc_med = usd_med / w[k] * USDJPY
            acc_p90 = usd_p90 / w[k] * USDJPY
            worst = max(worst, acc_med)
            unit = "$/oz" if k.startswith("gold") else "$/BTC"
            print(f"{k:<13}{w[k]*100:>7.3f}{med:>13.0f} {unit:<4}{usd_med:>17.2f}"
                  f"{acc_med:>17,.0f}円{acc_p90:>17,.0f}円")
        print(f"\n  → この6レッグ全部を、狙った重みどおりに建てるのに必要な口座 ≈ **{worst:,.0f}円**"
              f"（中央値ベース。半分のトレードはこれより広い損切りになる）")

    print("\n" + "=" * 96)
    print("100,000円 の口座では、各レッグの最小ロットが口座の何%のリスクになるか（直近の中央値）")
    print(f"  {'leg':<13}{'狙った重み':>10}{'実際に強いられるリスク':>22}{'超過':>10}")
    for k in SIX:
        s = D[k][D[k].index >= "2025"]
        if len(s) == 0:
            continue
        usd = np.median(s.values) * PER_POINT_MIN_LOT[k]
        forced = usd * USDJPY / 100_000 * 100
        print(f"  {k:<13}{w[k]*100:>9.2f}%{forced:>21.2f}%{forced/(w[k]*100):>9.1f}倍")


if __name__ == "__main__":
    main()
