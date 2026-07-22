"""Structural law 9 -- trend legs don't age, so a far fixed target always wins -- was derived with
ZERO financing cost. The terminal says otherwise: a long BTC position pays 30%/yr on its notional
and a long gold position pays $0.78 per ounce per day. A far target means a LONGER HOLD, and a
longer hold now costs money. The law's premise is broken, so the RR optimum may have moved.

Two levers, one at a time, with the swap charged exactly (BTC = %-of-price, gold = $/oz/day):
  L1  RR       -- does the optimum come DOWN now that time is expensive?
  L2  max hold -- a time stop was mechanically invalid before (surviving trades get BETTER), but
                  that too was measured with no carry. Re-ask it.
Judged on the LEG and on the 6-leg BOOK, whose swap-included baseline is CAGR/DD 4.84 (was 7.88).
The exposed legs are the slow ones: btc_bo_kama holds 12.8 days on average, gold_bo 4.6.
Run: .venv/bin/python experiments/rr_with_swap.py
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
from book_spec_fix import book

ROOT = "/home/angelbell/dev/auto-trade"
SIX = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]
BTC_PCT_YR, GOLD_USD_OZ_DAY = 30.0, 0.7829


def sw(hold, risk, px, kind):
    if kind == "btc_long":
        return (BTC_PCT_YR / 365.0 / 100.0) * (px / risk) * hold
    if kind == "gold_long":
        return GOLD_USD_OZ_DAY * hold / risk
    return np.zeros_like(hold)


_D = {}


def data():
    if not _D:
        with contextlib.redirect_stderr(io.StringIO()):
            _D["g1"] = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv").loc[GOLD_H1_START:], "1h")
            _D["b4"] = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h")
            _D["g15"] = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
            _D["d15"] = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    return _D


def leg(name, rr=None, fwd=None):
    d = data()
    with contextlib.redirect_stderr(io.StringIO()):
        if name == "gold_bo":
            t = run(d["g1"], SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": rr or 3.0,
                                                "fwd": fwd or 500, "daily_sma": 150,
                                                "daily_slope_k": 10}))
            R, k, risk = t["R"].values, "gold_long", t["risk"].values
        elif name == "btc_bo_kama":
            t = kama_gate_btc(run(d["b4"], SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h",
                                                              "rr": rr or 2.0, "fwd": fwd or 300})))
            R, k, risk = t["R"].values, "btc_long", t["risk"].values
        elif name == "btc_pull":
            cfg = {**PB, "csv": "x", "tf": "4h"}
            if rr:
                cfg["rr"] = rr
            if fwd:
                cfg["fwd"] = fwd
            t = cycle_gate_pull(run_pb(d["b4"], "long", SimpleNamespace(**cfg), 0.0))
            R, k, risk = t["R"].values, "btc_long", t["risk"].values
        elif name == "gold15m":
            t = run(d["g15"], SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                                 "ext_cap": 8.0, "pullback_frac": 0.25,
                                                 "fill_win": 200, "rr": rr or 4.0, "fwd": fwd or 500}))
            R = t["R"].values - 0.30 / t["risk"].values
            k, risk = "gold_long", t["risk"].values
        elif name == "btc15m_L":
            t = run(d["d15"], SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                                 "pullback_frac": 0.3, "rr": rr or 4.5,
                                                 "fill_win": 200, "fwd": fwd or 500}))
            pdh = d["d15"]["high"].resample("1D").max().dropna().shift(1).reindex(
                d["d15"].index, method="ffill").values
            w = np.where(t["e_px"].values > pdh[d["d15"].index.get_indexer(t["time"])], 1.0, 0.5)
            risk = t["risk"].values / w
            R, k = t["R"].values * w - 15.0 / risk, "btc_long"
        else:  # btc15m_S -- short: BTC swap is ZERO
            inv = invert(d["d15"]); C = 2 * d["d15"]["high"].max()
            t = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3,
                                            "rr": rr or 4.5, "fill_win": 200, "fwd": fwd or 500}))
            pdl = d["d15"]["low"].resample("1D").min().dropna().shift(1).reindex(
                d["d15"].index, method="ffill").values
            m = (C - t["e_px"].values) < pdl[d["d15"].index.get_indexer(t["time"])]
            R = (t["R"].values - 15.0 / t["risk"].values)[m]
            return (pd.Series(R, index=pd.DatetimeIndex(t["time"])[m]), t["hold"].values[m],
                    np.zeros(int(m.sum())))
    px = t["e_px"].values
    s = sw(t["hold"].values, risk, px, k)
    return pd.Series(R - s, index=pd.DatetimeIndex(t["time"])), t["hold"].values, s


def main():
    B0 = {k: leg(k)[0] for k in SIX}
    c0, d0, r0, _ = book(B0, SIX)
    pf = lambda x: x[x > 0].sum() / abs(x[x <= 0].sum())
    print(f"スワップ込みの正典 = ブック CAGR {c0:+.1f}% / maxDD {d0:.2f}% / **CAGR/DD {r0:.2f}**")
    print("（スワップ未計上の旧数字は 7.88。以下すべてスワップ込みで比較）\n")

    print("L1  RR を振り直す（保有が長いほどスワップを払う。最適点は下がるか？）\n")
    grids = {"gold_bo": [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
             "btc_bo_kama": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
             "btc_pull": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
             "gold15m": [2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
             "btc15m_L": [3.0, 3.5, 4.0, 4.5, 5.0, 6.0]}
    cur = {"gold_bo": 3.0, "btc_bo_kama": 2.0, "btc_pull": 3.0, "gold15m": 4.0, "btc15m_L": 4.5}
    for nm, g in grids.items():
        print(f"  {nm}  （現行 RR={cur[nm]}）")
        print(f"    {'RR':>5}{'n':>6}{'保有(日)平均':>13}{'スワップ/本':>12}{'PF':>7}{'meanR':>9}"
              f"{'ブック':>9}{'差':>8}")
        for rr in g:
            s, hold, sp = leg(nm, rr=rr)
            L = dict(B0); L[nm] = s
            rb = book(L, SIX)[2]
            mk = "  ← 現行" if rr == cur[nm] else ("  ★" if rb > r0 + 0.05 else "")
            print(f"    {rr:>5.1f}{len(s):>6}{hold.mean():>13.2f}{sp.mean():>+12.3f}"
                  f"{pf(s.values):>7.2f}{s.mean():>+9.3f}{rb:>9.2f}{rb-r0:>+8.2f}{mk}")
        print()

    print("\nL2  最大保有（時間ストップ）— スワップがある今なら成立するか\n")
    fw = {"gold_bo": [24, 48, 96, 200, 500], "btc_bo_kama": [12, 24, 48, 100, 300],
          "btc_pull": [12, 24, 48, 100, 300], "btc15m_L": [96, 192, 300, 500]}
    hrs = {"gold_bo": 1, "btc_bo_kama": 4, "btc_pull": 4, "btc15m_L": 0.25}
    for nm, g in fw.items():
        print(f"  {nm}")
        print(f"    {'最大保有':>9}{'≒日数':>8}{'n':>6}{'保有平均':>10}{'スワップ':>10}{'PF':>7}"
              f"{'meanR':>9}{'ブック':>9}{'差':>8}")
        for f in g:
            s, hold, sp = leg(nm, fwd=f)
            L = dict(B0); L[nm] = s
            rb = book(L, SIX)[2]
            print(f"    {f:>6}本{f*hrs[nm]/24:>8.1f}{len(s):>6}{hold.mean():>10.2f}{sp.mean():>+10.3f}"
                  f"{pf(s.values):>7.2f}{s.mean():>+9.3f}{rb:>9.2f}{rb-r0:>+8.2f}"
                  + ("  ★" if rb > r0 + 0.05 else ""))
        print()


if __name__ == "__main__":
    main()
