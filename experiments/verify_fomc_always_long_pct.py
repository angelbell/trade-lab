"""Corrected version of verify_fomc_always_long.py.

Two defects in the first pass, found before reporting:
  1. the event list per year was never checked (2026 showed n=8, which the calendar cannot
     support by mid-July) -- print the dates and let the count speak;
  2. the "share of gold's rise" was computed in DOLLARS while gold went 1282 -> 4018, so a
     given % move counts ~3x more in recent years. Redo everything in log returns.

Question unchanged: are the FOMC windows special, or a proportional slice of the uptrend?
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from experiments.fomc_event_study import price_before, candidate_dates

COST_FRAC = None   # cost applied as $/oz below, converted per-trade
COST = 0.30
HS = [60, 120, 240, 480]
B = 10000
rng = np.random.default_rng(42)

ev = pd.read_csv("experiments/fomc_stmt_2019.csv", parse_dates=["dt_utc", "dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())
df = load_mt5_csv("data/vantage_xauusd_m1.csv")

print("=== 1. 事象リストの年別本数（暦との整合）===")
yrs = pd.Series([e.year for e in events]).value_counts().sort_index()
print(yrs.to_string())
print("2026年の日付:", [str(e.date()) for e in events if e.year == 2026])
print(f"データ終端: {df.index.max()}")


def long_logret(ts, h):
    pe = price_before(df, ts + pd.Timedelta(minutes=1))
    px = price_before(df, ts + pd.Timedelta(minutes=1 + h))
    if pe is None or px is None or not np.isfinite(pe) or not np.isfinite(px) or pe <= 0:
        return None
    return np.log((px - COST) / pe)   # cost charged inside the return


cand = candidate_dates(df, events)
clock = events[0].time()
cand_ts = [pd.Timestamp.combine(d.date(), clock).tz_localize("UTC") for d in cand]

print(f"\n=== 2. 対数リターンで比較（常時ロング・コスト後）===")
print(f"{'H(分)':>6} {'n':>3} | {'FOMC 中央値%':>12} {'平均%':>7} {'勝率%':>6} | "
      f"{'対照 中央値%':>12} {'平均%':>7} {'勝率%':>6} | {'中央値%ile':>9}")
res = {}
for h in HS:
    r = np.array([x for x in (long_logret(t, h) for t in events) if x is not None])
    c = np.array([x for x in (long_logret(t, h) for t in cand_ts) if x is not None])
    d = rng.integers(0, len(c), size=(B, len(r)))
    pct = np.mean(np.median(c[d], axis=1) < np.median(r)) * 100
    res[h] = r
    print(f"{h:>6} {len(r):>3} | {np.median(r)*100:+12.3f} {r.mean()*100:+7.3f} {np.mean(r > 0)*100:6.1f} | "
          f"{np.median(c)*100:+12.3f} {c.mean()*100:+7.3f} {np.mean(c > 0)*100:6.1f} | {pct:9.1f}")

print("\n=== 3. 窓が特別か（対数リターンで、時間シェアと比較）===")
span_h = (df.index.max() - df.index.min()).total_seconds() / 3600
total_log = np.log(df["close"].iloc[-1] / df["close"].iloc[0])
print(f"全期間の対数リターン合計 {total_log*100:+.1f}%  ({span_h/24/365.25:.1f}年)")
for h in HS:
    win_h = len(res[h]) * h / 60
    share_move = res[h].sum() / total_log
    share_time = win_h / span_h
    print(f"  H={h:>3}分: 窓の合計 {res[h].sum()*100:+7.1f}% = 全上昇の {share_move*100:5.1f}%  | "
          f"時間シェア {share_time*100:4.2f}%  → 倍率 {share_move/share_time:5.1f}x")

print("\n=== 4. 年別（H=480, 対数リターン%, コスト後）===")
rows = [(e, long_logret(e, 480)) for e in events]
t = pd.DataFrame([(e, v) for e, v in rows if v is not None], columns=["t0", "lr"])
t["yr"] = t["t0"].dt.year
g = t.groupby("yr")["lr"].agg(n="count", 中央値=lambda s: s.median() * 100,
                              平均=lambda s: s.mean() * 100,
                              勝率=lambda s: (s > 0).mean() * 100)
print(g.round(3).to_string())
print(f"\n中央値が正の年: {(g['中央値'] > 0).sum()} / {len(g)}")
