"""Scalp vs 4-hour hold, judged the way an account actually experiences them.

Everything in one unit: percent of the entry price (so the 1282 -> 5292 run in gold does not
distort the comparison), then translated to dollars at today's price for a fixed 0.01 lot
(= 1 oz), which is the user's actual bet size.

The account accumulates the SUM, so the annual figure comes from the MEAN, not the median --
and it is the worst loss plus the drawdown, not the stop, that sets the balance required.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from scratchpad.event_scalp import build_scalp_table, COST_ROUNDTRIP
from scratchpad.event_scalp_cond import threshold_subset
from scratchpad.fomc_event_study import price_before

COST = COST_ROUNDTRIP["GOLD"]["base"]
ev = pd.read_csv("scratchpad/fomc_stmt_2019.csv", parse_dates=["dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())
df = load_mt5_csv("data/vantage_xauusd_m1.csv")
T_END, PX = df.index.max(), float(df["close"].iloc[-1])
span = (events[-1] - events[0]).days / 365.25


def stats(pct, label, hold):
    v = pct * PX
    eq = np.cumsum(v)
    peak = np.maximum.accumulate(np.concatenate([[0], eq]))
    dd = (peak[1:] - eq).max()
    run = mx = 0
    for p in pct:
        run = run + 1 if p < 0 else 0
        mx = max(mx, run)
    per_yr = len(pct) / span
    print(f"\n--- {label}（保有 {hold}）---")
    print(f"  n={len(pct)}  年{per_yr:.1f}本  勝率 {np.mean(pct > 0)*100:.1f}%")
    print(f"  中央値 {np.median(pct)*100:+.3f}% = {np.median(v):+6.2f}$   "
          f"平均 {pct.mean()*100:+.3f}% = {v.mean():+6.2f}$")
    print(f"  **年間の期待額 {v.mean()*per_yr:+6.1f}$**（平均×本数）")
    print(f"  最悪トレード {v.min():+6.2f}$   最大連敗 {mx}回   最大DD {dd:.0f}$")
    for risk in [0.02, 0.03]:
        need = abs(v.min()) / risk
        print(f"  最悪損を口座の{risk*100:.0f}%に抑える → 口座 ${need:,.0f} → 年率 "
              f"{v.mean()*per_yr/need*100:+.1f}%")
    return v.mean() * per_yr


# ---- A: the 1-minute scalp (threshold top 50%, exit at 5 min, no workable stop) ----
real = build_scalp_table(df, events, 1, [5], "hh")
sub, _ = threshold_subset(real, "confirm_move_atr", 0.50)
epx = np.array([price_before(df, t + pd.Timedelta(minutes=1)) for t in sub["t0"]])
scalp = ((sub["g_5"] - COST).values) / epx
a = stats(scalp, "1分スキャルプ（上位50%・初動と同方向）", "5分")

# ---- B: the 4-hour hold, always long, 1% stop --------------------------------------
out = []
for e in events:
    t_in = e + pd.Timedelta(minutes=1)
    t_out = t_in + pd.Timedelta(minutes=240)
    if t_out > T_END:
        continue
    pe = price_before(df, t_in)
    if pe is None or not np.isfinite(pe) or pe <= 0:
        continue
    seg = df.loc[t_in:t_out]
    if len(seg) < 2:
        continue
    lvl = pe * 0.99
    if (seg["low"] <= lvl).any():
        out.append((lvl - COST - pe) / pe)
        continue
    px = price_before(df, t_out)
    if px is not None and np.isfinite(px):
        out.append((px - COST - pe) / pe)
b = stats(np.array(out), "4時間・常時ロング（損切り1.0%）", "4時間")

print(f"\n=== 年間の期待額 ===  1分 {a:+.1f}$  vs  4時間 {b:+.1f}$   → {b/a:.1f} 倍")
print("※ 1分側はしきい値がin-sample選択で、逐次化すると効果が約半減する（さらに下がる）")
