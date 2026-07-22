"""Local verification of the concentration flag raised by the measurement pass.

Claim to check: one event (2026-06-17) carries most of the gold FOMC 1-min scalp, and the
documented "IS/OOS both positive" (IS +0.94 / OOS +2.92) survives only because that event
sits in the OOS half. Recomputed on the DOCUMENTED setting (F0 fill, w_c=1, H=5, frac=0.50,
cost $0.30/oz) using the frozen machinery, not the new exec script.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from experiments.event_scalp import build_scalp_table, COST_ROUNDTRIP
from experiments.event_scalp_cond import threshold_subset

COST = COST_ROUNDTRIP["GOLD"]["base"]

ev = pd.read_csv("experiments/fomc_stmt_2019.csv", parse_dates=["dt_utc", "dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())
df = load_mt5_csv("data/vantage_xauusd_m1.csv")

real = build_scalp_table(df, events, 1, [5, 10, 15], "verify")
sub, thr = threshold_subset(real, "confirm_move_atr", 0.50)
sub = sub.sort_values("t0").reset_index(drop=True)
sub["net"] = sub["g_5"] - COST

print(f"n={len(sub)}  thr(confirm_move_atr)={thr:.3f}  net_mean={sub['net'].mean():+.4f}  "
      f"net_sum={sub['net'].sum():+.2f}  win%={(sub['g_5'] > COST).mean()*100:.1f}")

print("\n-- 寄与の大きい順 トップ6 --")
top = sub.reindex(sub["net"].abs().sort_values(ascending=False).index).head(6)
for _, r in top.iterrows():
    print(f"  {r['t0'].date()}  net={r['net']:+8.2f}  "
          f"confirm_move_atr={r['confirm_move_atr']:6.2f}  シェア={r['net']/sub['net'].sum()*100:5.1f}%")

print("\n-- leave-one-out（1件抜いたときの net_mean）--")
for i in sub["net"].abs().sort_values(ascending=False).head(3).index:
    d = sub.drop(i)
    print(f"  {sub.loc[i, 't0'].date()} を除く: net_mean={d['net'].mean():+.4f} (n={len(d)}) "
          f"win%={(d['g_5'] > COST).mean()*100:.1f}")

half = len(sub) // 2
IS, OOS = sub.iloc[:half], sub.iloc[half:]
print(f"\n-- IS/OOS（台帳の +0.94 / +2.92 の再現）--")
print(f"  IS : net_mean={IS['net'].mean():+.4f} (n={len(IS)})  {IS['t0'].min().date()}..{IS['t0'].max().date()}")
print(f"  OOS: net_mean={OOS['net'].mean():+.4f} (n={len(OOS)})  {OOS['t0'].min().date()}..{OOS['t0'].max().date()}")
imax = sub["net"].abs().idxmax()
if imax in OOS.index:
    o = OOS.drop(imax)
    print(f"  OOS から {sub.loc[imax,'t0'].date()} を除く: net_mean={o['net'].mean():+.4f} (n={len(o)}) "
          f"win%={(o['g_5'] > COST).mean()*100:.1f}")

print("\n-- 中央値でも見る（平均は1件で動くので）--")
print(f"  net 中央値={sub['net'].median():+.4f}  標準偏差={sub['net'].std():.2f}  "
      f"最大={sub['net'].max():+.2f}  最小={sub['net'].min():+.2f}")
