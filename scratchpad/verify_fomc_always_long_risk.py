"""Always-long the FOMC statement window: the tradeable version, with the risk side.

Fixes the defect found in the previous pass: the event file contains FUTURE meeting dates
(2026-07-29 onward) past the end of the data, and calling price_before directly turned them
into phantom trades that entered and exited on the same last bar (= exactly minus cost). The
frozen machinery guards against this; my script did not. Guard added here.

Reports what the decision actually needs, not just the upside:
  - distribution of the outcome (median, std, quantiles) -- not the mean, n is small
  - MAE: how far underwater the trade goes before it works
  - per-year medians, and the control (non-FOMC weekdays, same clock)
  - everything converted to yen at 0.01 lot (= 1 oz) at the CURRENT gold price, since that
    is what the account actually earns and risks
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
T_END = df.index.max()
PX_NOW = float(df["close"].iloc[-1])


def trade(ts, h):
    """Always-long. Returns (log return after cost, MAE in %) or None if not fully covered."""
    t_in = ts + pd.Timedelta(minutes=1)
    t_out = t_in + pd.Timedelta(minutes=h)
    if t_out > T_END:
        return None                      # the guard the previous script was missing
    pe = price_before(df, t_in)
    px = price_before(df, t_out)
    if pe is None or px is None or not np.isfinite(pe) or not np.isfinite(px) or pe <= 0:
        return None
    seg = df.loc[t_in:t_out]
    if len(seg) < 2:
        return None
    mae = (seg["low"].min() - pe) / pe    # most negative excursion while long
    return np.log((px - COST) / pe), mae


cand = candidate_dates(df, events)
clock = events[0].time()
cand_ts = [pd.Timestamp.combine(d.date(), clock).tz_localize("UTC") for d in cand]

print(f"データ終端 {T_END}  / 現在価格 {PX_NOW:.1f} $/oz  (0.01ロット=1oz)")
print(f"{'H(分)':>5} {'n':>3} | {'中央値%':>8} {'平均%':>7} {'標準偏差%':>9} {'勝率%':>6} "
      f"{'25%点':>7} {'75%点':>7} | {'対照中央値%':>11} {'%ile':>5} | {'円/回':>7} {'年間円':>8}")

store = {}
for h in HS:
    rr = [trade(t, h) for t in events]
    r = np.array([x[0] for x in rr if x is not None])
    mae = np.array([x[1] for x in rr if x is not None])
    cc = [trade(t, h) for t in cand_ts]
    c = np.array([x[0] for x in cc if x is not None])
    d = rng.integers(0, len(c), size=(B, len(r)))
    pct = np.mean(np.median(c[d], axis=1) < np.median(r)) * 100
    store[h] = (r, mae)
    yen = np.median(r) * PX_NOW * JPY
    print(f"{h:>5} {len(r):>3} | {np.median(r)*100:+8.3f} {r.mean()*100:+7.3f} {r.std(ddof=1)*100:9.3f} "
          f"{np.mean(r > 0)*100:6.1f} {np.percentile(r, 25)*100:+7.3f} {np.percentile(r, 75)*100:+7.3f} | "
          f"{np.median(c)*100:+11.3f} {pct:5.1f} | {yen:7.0f} {yen*len(r)/7.5:8.0f}")

print("\n=== リスク側: 保有中の最大逆行 MAE（%と、0.01ロットの円）===")
print(f"{'H(分)':>5} | {'中央値':>8} {'標準偏差':>8} {'95%点':>8} {'最悪':>8} | {'中央値 円':>9} {'95%点 円':>9} {'最悪 円':>9}")
for h in HS:
    m = store[h][1]
    q = lambda v: v * PX_NOW * JPY
    print(f"{h:>5} | {np.median(m)*100:+8.3f} {m.std(ddof=1)*100:8.3f} {np.percentile(m, 5)*100:+8.3f} "
          f"{m.min()*100:+8.3f} | {q(np.median(m)):9.0f} {q(np.percentile(m, 5)):9.0f} {q(m.min()):9.0f}")

print("\n=== 年別（H=480 と H=60, 対数リターン中央値 %）===")
rows = []
for h in [60, 480]:
    for e in events:
        t = trade(e, h)
        if t is not None:
            rows.append({"yr": e.year, "H": h, "lr": t[0]})
t = pd.DataFrame(rows)
print(t.pivot_table(index="yr", columns="H", values="lr",
                    aggfunc=[lambda s: s.median() * 100, "count"]).round(3).to_string())
