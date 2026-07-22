"""Does the chart's first-minute direction add anything over just taking the era's direction?

The user's question: you cannot read an FOMC statement in one minute, so shouldn't you bet the
direction the chart is already moving? Answer it directly by putting four rules side by side on
gold, 2019-2026 (an easing era, so the era direction is LONG), horizon 4 hours:

  A 常時ロング          -- ignore the chart entirely (era direction only)
  B 一致時のみロング     -- long only when the first minute also went up
  C チャート追随        -- long if the first minute went up, short if it went down
  D 不一致時のみロング   -- long only when the first minute went DOWN (fade the chart)

If the chart carries information at this horizon, B and C beat A. If the era carries it, A and D
hold up and C bleeds on its short leg. Judged on the median (n is small and one 2026 event owns
the mean), with the same-clock non-FOMC control as the null.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from experiments.fomc_event_study import price_before, candidate_dates

COST = 0.30
H = 240
B = 10000
JPY = 150.0
rng = np.random.default_rng(42)

ev = pd.read_csv("experiments/fomc_stmt_2019.csv", parse_dates=["dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())
df = load_mt5_csv("data/vantage_xauusd_m1.csv")
T_END = df.index.max()
PX = float(df["close"].iloc[-1])


def leg(ts, h):
    """Returns (first-minute direction, long log-return over h after that minute)."""
    p0 = price_before(df, ts)
    t_in = ts + pd.Timedelta(minutes=1)
    t_out = t_in + pd.Timedelta(minutes=h)
    if t_out > T_END:
        return None
    pe, px = price_before(df, t_in), price_before(df, t_out)
    if None in (p0, pe, px) or not all(np.isfinite([p0, pe, px])) or pe <= 0:
        return None
    d = np.sign(pe - p0)
    if d == 0:
        return None
    return d, np.log((px - COST) / pe), np.log(pe / (px + COST))   # long, short


cand = candidate_dates(df, events)
clock = events[0].time()
ctl_ts = [pd.Timestamp.combine(d.date(), clock).tz_localize("UTC") for d in cand]

real = [x for x in (leg(t, H) for t in events) if x is not None]
ctl = [x for x in (leg(t, H) for t in ctl_ts) if x is not None]
print(f"FOMC n={len(real)}  対照 n={len(ctl)}   gold {PX:.0f}$/oz  0.01ロット=1oz  H={H}分")

RULES = {
    "A 常時ロング":        lambda r: [x[1] for x in r],
    "B 一致時のみロング":   lambda r: [x[1] for x in r if x[0] > 0],
    "C チャート追随":      lambda r: [x[1] if x[0] > 0 else x[2] for x in r],
    "D 不一致時のみロング": lambda r: [x[1] for x in r if x[0] < 0],
}

print(f"\n{'rule':>18} {'n':>3} {'本/年':>6} | {'中央値%':>8} {'勝率%':>6} {'25%点':>7} {'75%点':>7} "
      f"| {'対照中央値%':>11} {'%ile':>6} | {'円/回':>7} {'年間円':>8}")
span = (events[-1] - events[0]).days / 365.25
for name, f in RULES.items():
    r = np.array(f(real))
    c = np.array(f(ctl))
    if len(r) < 5:
        continue
    d = rng.integers(0, len(c), size=(B, len(r)))
    pct = np.mean(np.median(c[d], axis=1) < np.median(r)) * 100
    yen = np.median(r) * PX * JPY
    print(f"{name:>18} {len(r):>3} {len(r)/span:>6.1f} | {np.median(r)*100:+8.3f} "
          f"{np.mean(r > 0)*100:6.1f} {np.percentile(r, 25)*100:+7.3f} {np.percentile(r, 75)*100:+7.3f} "
          f"| {np.median(c)*100:+11.3f} {pct:6.1f} | {yen:7.0f} {yen*len(r)/span:8.0f}")

print("\n=== 短い保有(H=5分)でも同じ比較 — チャート追随が効くのはこちらのはず ===")
real5 = [x for x in (leg(t, 5) for t in events) if x is not None]
ctl5 = [x for x in (leg(t, 5) for t in ctl_ts) if x is not None]
for name, f in RULES.items():
    r = np.array(f(real5)); c = np.array(f(ctl5))
    if len(r) < 5:
        continue
    d = rng.integers(0, len(c), size=(B, len(r)))
    pct = np.mean(np.median(c[d], axis=1) < np.median(r)) * 100
    print(f"{name:>18} {len(r):>3} | 中央値 {np.median(r)*100:+7.3f}%  勝率 {np.mean(r > 0)*100:5.1f}%  "
          f"%ile {pct:5.1f}  円/回 {np.median(r)*PX*JPY:6.0f}")
