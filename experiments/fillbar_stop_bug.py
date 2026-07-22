"""BUG: breakout_wave's pullback-limit path never checks the stop on the FILL BAR.

    e_px, e_bar = lim, fj            # filled on bar fj
    for j in range(e_bar + 1, ...):  # <-- walk starts AFTER the fill bar
        if l[j] <= stop: R = -1.0

If the bar that reaches down to the limit ALSO reaches down through the stop, the model fills you
and then grants a free pass on that bar. In reality you are filled and stopped inside the same bar.

The contamination scales with the pullback depth, because the limit and the stop converge:
    gap between them = (1 - frac) x risk
    frac 0.25 -> the bar must span 75% of the risk to cheat.  frac 0.70 -> only 30%.
That is exactly why the book rose monotonically to frac 0.70 (meanR +1.74, PF 2.83) -- the deeper
the limit, the more free rides the model hands out.

Measure it at the ADOPTED settings (gold15m frac 0.25, btc15m_L/S frac 0.30), then re-price the
book with the stop checked ON the fill bar (conservative: stop wins a same-bar tie, as everywhere else).
Run: .venv/bin/python experiments/fillbar_stop_bug.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE

ROOT = "/home/angelbell/dev/auto-trade"




def main():
    print("採用中の設定で、『指値に刺さった足が、その足で損切りも突き抜けていた』割合を数える。\n")
    for tag, csv, start, cfg, cost in (
        ("gold15m (押し目0.25)", "vantage_xauusd_m5.csv", "2018-09-14",
         dict(daily_sma=150, daily_slope_k=10, ext_cap=8.0, pullback_frac=0.25, rr=4.0), 0.30),
        ("btc15m_L (押し目0.30)", "vantage_btcusd_m15.csv", "2018-10-01",
         dict(gate_kama=14, gate_kama_tf="240min", pullback_frac=0.30, rr=4.5), 15.0),
    ):
        with contextlib.redirect_stderr(io.StringIO()):
            d = resample(load_mt5_csv(f"{ROOT}/data/{csv}").loc[start:], "15min")
            t = run(d, SimpleNamespace(**{**BASE, **cfg, "fill_win": 200}))
        # t gives the FILL bar (time), fill price (e_px) and the realised risk (= e_px - stop).
        idx = d.index.get_indexer(pd.DatetimeIndex(t["time"]))
        stop = t["e_px"].values - t["risk"].values          # the structural stop
        lo_fill = d["low"].values[idx]                      # the low of the bar we were filled on
        cheated = lo_fill <= stop                           # ...which also went through the stop
        print(f"  {tag}:  n={len(t)}   **約定した足がそのまま損切りも突き抜けていた = "
              f"{cheated.sum()}本 ({100*cheated.mean():.1f}%)**")
        R = t["R"].values
        Rfix = R.copy()
        Rfix[cheated] = -1.0 - cost / t["risk"].values[cheated]   # 現実: その足で狩られる
        pf0 = R[R > 0].sum() / abs(R[R <= 0].sum())
        pf1 = Rfix[Rfix > 0].sum() / abs(Rfix[Rfix <= 0].sum())
        won = cheated & (R > 0)
        print(f"      うち、モデル上は**勝ちになっていた**もの: {won.sum()}本 "
              f"(合計 {R[won].sum():+.1f}R を計上していた)")
        print(f"      PF {pf0:.2f} → **{pf1:.2f}**    meanR {R.mean():+.3f} → **{Rfix.mean():+.3f}**")
        print()

    print("押し目の深さ別に、汚染率がどう増えるか（＝単調な右肩上がりの正体）")
    print(f"  {'押し目':>7}{'n':>6}{'指値と損切りの間隔':>18}{'汚染された足':>14}{'うち勝ち計上':>14}"
          f"{'PF(現行)':>10}{'PF(修正)':>10}")
    with contextlib.redirect_stderr(io.StringIO()):
        d = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
    for fr in (0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70):
        with contextlib.redirect_stderr(io.StringIO()):
            t = run(d, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                          "ext_cap": 8.0, "pullback_frac": fr, "rr": 4.0,
                                          "fill_win": 200}))
        idx = d.index.get_indexer(pd.DatetimeIndex(t["time"]))
        stop = t["e_px"].values - t["risk"].values
        cheated = d["low"].values[idx] <= stop
        R = t["R"].values - 0.30 / t["risk"].values
        Rf = R.copy(); Rf[cheated] = -1.0 - 0.30 / t["risk"].values[cheated]
        pf0 = R[R > 0].sum() / abs(R[R <= 0].sum())
        pf1 = Rf[Rf > 0].sum() / abs(Rf[Rf <= 0].sum())
        print(f"  {fr:>7.2f}{len(t):>6}{100*(1-fr):>16.0f}%{cheated.sum():>10} "
              f"({100*cheated.mean():>3.0f}%){(cheated & (R>0)).sum():>13}{pf0:>10.2f}{pf1:>10.2f}")


if __name__ == "__main__":
    main()
