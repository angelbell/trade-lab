"""Put the 1-minute scalp and the 4-hour hold in the SAME unit before comparing them.

The scalp was reported in absolute $/oz measured at the price of the day; the 4-hour hold was
reported in percent. Gold went 1282 -> 4018 over the sample, so a given percentage move is
worth ~3x more dollars at the end than at the start -- comparing the two directly understated
the scalp and inflated the ratio between them. Redo both as log returns on the same events,
then translate to $/oz at today's price.
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
PX = float(df["close"].iloc[-1])
T_END = df.index.max()

# --- the scalp, re-expressed as a percentage of the price on the day ---------------
real = build_scalp_table(df, events, 1, [5, 10, 15], "u")
sub, _ = threshold_subset(real, "confirm_move_atr", 0.50)
sub = sub.sort_values("t0").reset_index(drop=True)
entry_px = np.array([price_before(df, t + pd.Timedelta(minutes=1)) for t in sub["t0"]])
net_usd = (sub["g_5"] - COST).values
pct = net_usd / entry_px
drop = np.argmax(np.abs(net_usd))
print("=== 1分スキャルプ (w_c=1, frac=0.50, H=5) ===")
print(f"  建値の中央値 {np.median(entry_px):.0f} $/oz  (最初 {entry_px[0]:.0f} → 最後 {entry_px[-1]:.0f})")
for lab, m in [("全29件", np.ones(len(pct), bool)),
               ("外れ値除去28件", np.arange(len(pct)) != drop)]:
    print(f"  {lab}: 中央値 {np.median(net_usd[m]):+.2f} $/oz = {np.median(pct[m])*100:+.3f}%"
          f"   → 現在価格({PX:.0f})換算 {np.median(pct[m])*PX:+.2f} $/oz   勝率 {np.mean(net_usd[m] > 0)*100:.1f}%")

# --- the 4-hour hold on the same events -------------------------------------------
def seg(ts, h):
    t_in = ts + pd.Timedelta(minutes=1)
    t_out = t_in + pd.Timedelta(minutes=h)
    if t_out > T_END:
        return None
    pe, px = price_before(df, t_in), price_before(df, t_out)
    if pe is None or px is None or not np.isfinite(pe) or not np.isfinite(px) or pe <= 0:
        return None
    return (px - COST - pe) / pe

r4 = np.array([x for x in (seg(t, 240) for t in events) if x is not None])
print("\n=== 4時間・常時ロング（全会合）===")
print(f"  n={len(r4)}  中央値 {np.median(r4)*100:+.3f}%  → 現在価格換算 {np.median(r4)*PX:+.2f} $/oz"
      f"   勝率 {np.mean(r4 > 0)*100:.1f}%")

s = np.median(pct[np.arange(len(pct)) != drop])
print(f"\n=== 同じ単位での比較 ===")
print(f"  1回あたり: {np.median(r4)/s:.1f} 倍   （スキャルプ {s*100:+.3f}% vs 4時間 {np.median(r4)*100:+.3f}%）")
print(f"  年間: スキャルプ 年{len(pct)/7.5:.1f}本 × {s*100:+.3f}% = {s*len(pct)/7.5*100:+.2f}%/年")
print(f"        4時間     年{len(r4)/7.5:.1f}本 × {np.median(r4)*100:+.3f}% = {np.median(r4)*len(r4)/7.5*100:+.2f}%/年"
      f"   → 年間 {np.median(r4)*len(r4)/(s*len(pct)):.1f} 倍")
