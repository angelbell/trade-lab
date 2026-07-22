"""Is the FOMC window SPECIAL, or just a slice of gold's uptrend?

The user's objection is fair: "beta" is not by itself a reason to reject something. What
actually matters is whether waiting for FOMC buys you anything you could not get by simply
holding gold. So test the rule I discarded without testing -- IGNORE the statement direction
and just BUY after the statement -- and ask one decisive question:

    those 63 windows are ~64 hours per year. Did they deliver MORE than 64 hours' worth of
    gold's move, or exactly their share of the clock?

Share-of-move > share-of-time  => the window is special (timing edge, worth riding).
Share-of-move ~ share-of-time  => it is gold's drift sampled, obtainable by just holding.

Also compares against non-FOMC weekdays at the SAME clock time (the honest control) and
reports per-year, because a drift effect that only exists in a bull market is a regime bet.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from scratchpad.fomc_event_study import price_before, candidate_dates

COST = 0.30
HS = [60, 120, 240, 480]
JPY = 150.0
B = 10000
rng = np.random.default_rng(42)

ev = pd.read_csv("scratchpad/fomc_stmt_2019.csv", parse_dates=["dt_utc", "dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())
df = load_mt5_csv("data/vantage_xauusd_m1.csv")


def long_ret(ts, h):
    """Always-long: enter at the same bar the scalp enters (t0+1min), exit h minutes later."""
    pe = price_before(df, ts + pd.Timedelta(minutes=1))
    px = price_before(df, ts + pd.Timedelta(minutes=1 + h))
    if pe is None or px is None or not np.isfinite(pe) or not np.isfinite(px):
        return None
    return px - pe


cand = candidate_dates(df, events)
clock = events[0].time()
cand_ts = [pd.Timestamp.combine(d.date(), clock).tz_localize("UTC") for d in cand]

print(f"FOMC声明 {len(events)}本 / 対照(非FOMC平日・同時刻) {len(cand_ts)}本\n")
print(f"{'H(分)':>6} {'n':>3} | {'FOMC 中央値':>11} {'平均':>7} {'勝率%':>6} | "
      f"{'対照 中央値':>11} {'平均':>7} {'勝率%':>6} | {'中央値%ile':>9} {'円/回(中央値)':>13}")

res = {}
for h in HS:
    r = np.array([x for x in (long_ret(t, h) for t in events) if x is not None]) - COST
    c = np.array([x for x in (long_ret(t, h) for t in cand_ts) if x is not None]) - COST
    d = rng.integers(0, len(c), size=(B, len(r)))
    pct = np.mean(np.median(c[d], axis=1) < np.median(r)) * 100
    res[h] = r
    print(f"{h:>6} {len(r):>3} | {np.median(r):+11.2f} {r.mean():+7.2f} {np.mean(r > 0)*100:6.1f} | "
          f"{np.median(c):+11.2f} {c.mean():+7.2f} {np.mean(c > 0)*100:6.1f} | "
          f"{pct:9.1f} {np.median(r)*JPY:13.0f}")

# --- the decisive one: share of gold's total move vs share of the clock ------------
print("\n=== 窓が特別か、時計を等分しただけか ===")
span_h = (df.index.max() - df.index.min()).total_seconds() / 3600
total_move = df["close"].iloc[-1] - df["close"].iloc[0]
print(f"gold の全期間: {df.index.min().date()} .. {df.index.max().date()}  "
      f"({span_h/24/365.25:.1f}年)  終値 {df['close'].iloc[0]:.1f} → {df['close'].iloc[-1]:.1f} "
      f"= {total_move:+.1f} $/oz")
for h in HS:
    r_gross = res[h] + COST
    win_h = len(r_gross) * h / 60
    print(f"  H={h:>3}分: 窓の合計 {r_gross.sum():+8.1f} $/oz = 全期間上昇の {r_gross.sum()/total_move*100:5.1f}%  "
          f"| 占有時間 {win_h:6.0f}h = 全時間の {win_h/span_h*100:4.2f}%  "
          f"→ 倍率 {(r_gross.sum()/total_move)/(win_h/span_h):5.1f}x")

print("\n=== 年別（H=480, 常時ロング, コスト後 $/oz）===")
t = pd.DataFrame({"t0": [e for e in events if long_ret(e, 480) is not None]})
t["net"] = [long_ret(e, 480) - COST for e in t["t0"]]
t["yr"] = t["t0"].dt.year
g = t.groupby("yr")["net"].agg(["count", "median", "mean", lambda s: (s > 0).mean() * 100])
g.columns = ["n", "中央値", "平均", "勝率%"]
print(g.round(2).to_string())
