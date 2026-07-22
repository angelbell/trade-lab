"""What the 4-hour FOMC trade actually costs to run: stop size, drawdown, account needed.

Fixed 0.01 lot gold (= 1 oz), entry at statement +1 min, 1.0% stop, time exit at +4h.
Reports in dollars at TODAY's price (forward-looking, since gold ran 1282 -> 4349 over the
sample and a fixed lot therefore earned/risked ~3x more at the end than at the start), plus
the run of consecutive losses and the equity drawdown that set the balance requirement.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from experiments.fomc_event_study import price_before

COST = 0.30
STOP = 0.010
H = 240
ev = pd.read_csv("experiments/fomc_stmt_2019.csv", parse_dates=["dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())
df = load_mt5_csv("data/vantage_xauusd_m1.csv")
T_END = df.index.max()
PX = float(df["close"].iloc[-1])


def trade(ts):
    t_in = ts + pd.Timedelta(minutes=1)
    t_out = t_in + pd.Timedelta(minutes=H)
    if t_out > T_END:
        return None
    pe = price_before(df, t_in)
    if pe is None or not np.isfinite(pe) or pe <= 0:
        return None
    seg = df.loc[t_in:t_out]
    if len(seg) < 2:
        return None
    lvl = pe * (1 - STOP)
    if (seg["low"] <= lvl).any():
        return pe, (lvl - COST - pe) / pe, True
    px = price_before(df, t_out)
    if px is None or not np.isfinite(px):
        return None
    return pe, (px - COST - pe) / pe, False


rows = [t for t in (trade(e) for e in events) if t is not None]
entry = np.array([r[0] for r in rows])
pct = np.array([r[1] for r in rows])
hit = np.array([r[2] for r in rows])
n = len(rows)
span = (events[-1] - events[0]).days / 365.25

usd_hist = pct * entry          # what the account actually made per trade back then
usd_now = pct * PX              # what the same % is worth at today's price

print(f"n={n}  年{n/span:.1f}本   建値 {entry.min():.0f}-{entry.max():.0f}$/oz  現在 {PX:.0f}$/oz")
print(f"\n=== 1トレードの損益（0.01ロット=1oz、損切り{STOP*100:.1f}%）===")
for lab, v in [("当時の実額", usd_hist), ("現在価格換算", usd_now)]:
    print(f"  {lab}: 中央値 {np.median(v):+7.2f}$  平均 {v.mean():+7.2f}$  "
          f"最大益 {v.max():+7.2f}$  **最大損 {v.min():+7.2f}$**")
print(f"  損切り発動 {hit.mean()*100:.1f}%  勝率 {np.mean(pct > 0)*100:.1f}%")
print(f"  損切り幅 = 建値の{STOP*100:.1f}% → 現在価格で **${PX*STOP:.0f}/oz**"
      f"（1oz なので1トレードの最大損失も約${PX*STOP:.0f}）")

print("\n=== 連敗と資産曲線（現在価格換算・固定0.01ロット）===")
eq = np.cumsum(usd_now)
peak = np.maximum.accumulate(np.concatenate([[0], eq]))
dd = peak[1:] - eq
run, mx = 0, 0
for p in pct:
    run = run + 1 if p < 0 else 0
    mx = max(mx, run)
print(f"  最大連敗 {mx} 回 → 連続で {mx*PX*STOP:.0f}$ まで（損切り連発の最悪ケース）")
print(f"  最大ドローダウン **{dd.max():.0f}$**（累積の山からの落ち込み）")
print(f"  累計 {eq[-1]:+.0f}$ / {span:.1f}年 = 年 {eq[-1]/span:+.0f}$")

print("\n=== 必要な口座残高 ===")
print(f"  必要証拠金（gold 1oz = ${PX*1:.0f} の建玉）:")
for lev in [100, 200, 500]:
    print(f"    レバレッジ 1:{lev} → ${PX/lev:.0f}")
print(f"  ドローダウン耐性（実測DD ${dd.max():.0f} × 1.5〜2 ＝ CLAUDE.mdの規約）: "
      f"${dd.max()*1.5:.0f}〜${dd.max()*2:.0f}")
print(f"\n  1トレードの損失を口座の何%に抑えたいかで決まる（損失は約${PX*STOP:.0f}）:")
for risk in [0.01, 0.02, 0.03, 0.05]:
    print(f"    {risk*100:.0f}% に抑える → 口座 ${PX*STOP/risk:,.0f}")
