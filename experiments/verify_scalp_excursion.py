"""Excursion first, RR second -- the step that was skipped for the FOMC 1-minute scalp.

The frozen spec is: market at the confirm close, stop at the impulse origin P0, time exit at
5 minutes. Because the stop is the leg itself, R is defined = |P_entry - P0|, so the favourable
excursion measured in R IS the reachable reward-to-risk. That was never reported; the time-exit
results and the stop sweep came first, which is backwards.

Reports, for several windows: MFE and MAE as a distribution (median, std, quartiles) in percent,
in dollars at today's price, and in R -- plus the reach rate P(MFE >= x) at a ladder of targets,
which is what decides whether a fixed target beats the 5-minute clock.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from experiments.event_scalp import build_scalp_table, COST_ROUNDTRIP
from experiments.event_scalp_cond import threshold_subset
from experiments.fomc_event_study import price_before

COST = COST_ROUNDTRIP["GOLD"]["base"]
df = load_mt5_csv("data/vantage_xauusd_m1.csv")
T_END, PX = df.index.max(), float(df["close"].iloc[-1])
ev = pd.read_csv("experiments/fomc_stmt_2019.csv", parse_dates=["dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())
sub = threshold_subset(build_scalp_table(df, events, 1, [5], "ex"),
                       "confirm_move_atr", 0.50)[0].sort_values("t0").reset_index(drop=True)


def exc(t0, d, W):
    P0 = price_before(df, t0)
    t_in = t0 + pd.Timedelta(minutes=1)
    t_out = t_in + pd.Timedelta(minutes=W)
    if t_out > T_END:
        return None
    pe = price_before(df, t_in)
    if P0 is None or pe is None or pe <= 0:
        return None
    b = df.loc[t_in:t_out]
    if len(b) < 2:
        return None
    R = abs(pe - P0)                      # the stop distance = 1R
    if R <= 0:
        return None
    mfe = (d * (b["high"].max() - pe)) if d > 0 else (d * (b["low"].min() - pe))
    mae = (d * (b["low"].min() - pe)) if d > 0 else (d * (b["high"].max() - pe))
    return mfe / pe, mae / pe, mfe / R, mae / R, R / pe


for W in [5, 10, 15]:
    rows = [x for x in (exc(r["t0"], r["d"], W) for _, r in sub.iterrows()) if x is not None]
    mfe, mae, mfeR, maeR, Rpct = (np.array([r[i] for r in rows]) for i in range(5))
    print(f"\n########## 保有 {W}分  n={len(rows)} ##########")
    print(f"  1R(=脚の幅) 中央値 {np.median(Rpct)*100:.3f}% = {np.median(Rpct)*PX:.2f}$/oz  "
          f"標準偏差 {Rpct.std(ddof=1)*100:.3f}%")
    for lab, v, vR in [("巡行幅 MFE", mfe, mfeR), ("逆行幅 MAE", mae, maeR)]:
        print(f"  {lab}: 中央値 {np.median(v)*100:+.3f}% ({np.median(v)*PX:+.2f}$)  "
              f"標準偏差 {v.std(ddof=1)*100:.3f}%  "
              f"25%点 {np.percentile(v, 25)*100:+.3f}%  75%点 {np.percentile(v, 75)*100:+.3f}%")
        print(f"      R単位: 中央値 {np.median(vR):+.2f}R  標準偏差 {vR.std(ddof=1):.2f}R  "
              f"25%点 {np.percentile(vR, 25):+.2f}R  75%点 {np.percentile(vR, 75):+.2f}R")
    print(f"  到達率 P(巡行幅 ≥ x)  ※コスト {COST}$ は含めず素の値幅")
    print("      " + "  ".join(f"{x:.1f}R:{np.mean(mfeR >= x)*100:4.0f}%" for x in
                               [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]))
    print("      " + "  ".join(f"{x*100:.2f}%:{np.mean(mfe >= x)*100:4.0f}%" for x in
                               [0.0005, 0.001, 0.002, 0.003, 0.005]))
    print(f"  MFE/|MAE| の比（中央値） {np.median(mfeR) / abs(np.median(maeR)):.2f}   "
          f"損切り(1R)到達 {np.mean(maeR <= -1.0)*100:.0f}%")
