"""Is the weakness of 2024-2026 decay, or just the noise of an 8-trade-per-year sample?

Per-year medians (H=240): 2019 +0.22, 2020 +0.36, 2021 +0.50, 2022 +0.58, 2023 +0.60,
2024 +0.15, 2025 -0.53, 2026 -0.71. The last stretch is the worst, and it is also the stretch
the live-forward would be sampling -- so the decision to start on 2026-07-29 rests on whether
that run means anything.

Test it the only honest way at n=59: ask how often a random reordering of the SAME trades
produces a final block this bad. If a large fraction do, the run is noise; if almost none do,
the effect is fading. Reported for several block lengths so the answer does not hinge on where
the line is drawn.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from scratchpad.fomc_event_study import price_before

COST = 0.30
B = 20000
rng = np.random.default_rng(42)
df = load_mt5_csv("data/vantage_xauusd_m1.csv")
T_END = df.index.max()
ev = pd.read_csv("scratchpad/fomc_stmt_2019.csv", parse_dates=["dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())


def hold(ts, h=240):
    t_in = ts + pd.Timedelta(minutes=1)
    t_out = t_in + pd.Timedelta(minutes=h)
    if t_out > T_END:
        return None
    pe, px = price_before(df, t_in), price_before(df, t_out)
    if pe is None or px is None or not np.isfinite(pe) or not np.isfinite(px) or pe <= 0:
        return None
    return (px - COST - pe) / pe


rows = [(t, hold(t)) for t in events]
rows = [(t, v) for t, v in rows if v is not None]
ts = pd.Series([r[1] for r in rows], index=[r[0] for r in rows]).sort_index()
r = ts.values
print(f"n={len(r)}  {ts.index[0].date()} .. {ts.index[-1].date()}")

print("\n=== 前半 vs 後半（時系列で分割）===")
for cut in ["2023-01-01", "2024-01-01", "2025-01-01"]:
    a = ts[ts.index < cut].values
    b = ts[ts.index >= cut].values
    if len(b) < 5:
        continue
    print(f"  {cut}で分割: 前 n={len(a)} 中央値{np.median(a)*100:+.3f}% 平均{a.mean()*100:+.3f}% | "
          f"後 n={len(b)} 中央値{np.median(b)*100:+.3f}% 平均{b.mean()*100:+.3f}%")

print("\n=== 末尾ブロックの悪さは、並べ替えでどれだけ起きるか ===")
print(f"{'末尾':>6} {'実測 中央値':>11} {'実測 平均':>10} | {'P(並べ替えでこれ以下)':>22}")
for k in [8, 12, 16, 20, 24]:
    if k >= len(r):
        continue
    real_med, real_mean = np.median(r[-k:]), r[-k:].mean()
    idx = np.argsort(rng.random((B, len(r))), axis=1)[:, :k]
    perm = r[idx]
    p_med = np.mean(np.median(perm, axis=1) <= real_med) * 100
    p_mean = np.mean(perm.mean(axis=1) <= real_mean) * 100
    print(f"{k:>6} {real_med*100:+11.3f}% {real_mean*100:+10.3f}% | "
          f"中央値 {p_med:5.1f}%   平均 {p_mean:5.1f}%")

print("\n=== 参考: 直近12本・20本の内訳 ===")
for k in [12, 20]:
    tail = ts.iloc[-k:]
    print(f"  直近{k}本 ({tail.index[0].date()}〜): 勝率 {np.mean(tail.values > 0)*100:.1f}%  "
          f"中央値 {np.median(tail.values)*100:+.3f}%  合計 {tail.values.sum()*100:+.2f}%")
