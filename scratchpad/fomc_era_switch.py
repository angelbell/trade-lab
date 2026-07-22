"""Is there an ON/OFF switch? Measure the post-FOMC dollar drift across four policy eras.

fomc_hawkish_era_v2.py showed the drift is absent in 2004-2006 (hiking) and present in
2019-2026 (easing). That is only half the claim: "absent when hawkish" does not establish
"present when dovish" -- 2019-2026 could be the one special stretch. So add the other maximally
dovish era, 2008-2012 (QE1/QE2/QE3), and put all four side by side on the identical machine.

If the drift is ON in both dovish eras and OFF in both hawkish eras, the user gets a switch
they can watch (the Fed's stance) instead of a black box that might stop working silently.

Release located per meeting from the data (largest |5-min move| in the afternoon window), so
nothing depends on the broker's historical timezone convention -- which is NOT constant.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv

COST = 0.009
B = 10000
rng = np.random.default_rng(42)
WIN_LO, WIN_HI = "18:00", "23:00"

SCHED = {
    2004: ["01-28", "03-16", "05-04", "06-30", "08-10", "09-21", "11-10", "12-14"],
    2005: ["02-02", "03-22", "05-03", "06-30", "08-09", "09-20", "11-01", "12-13"],
    2006: ["01-31", "03-28", "05-10", "06-29", "08-08", "09-20", "10-25", "12-12"],
    2008: ["01-30", "03-18", "04-30", "06-25", "08-05", "09-16", "10-29", "12-16"],
    2009: ["01-28", "03-18", "04-29", "06-24", "08-12", "09-23", "11-04", "12-16"],
    2010: ["01-27", "03-16", "04-28", "06-23", "08-10", "09-21", "11-03", "12-14"],
    2011: ["01-26", "03-15", "04-27", "06-22", "08-09", "09-21", "11-02", "12-13"],
    2012: ["01-25", "03-13", "04-25", "06-20", "08-01", "09-13", "10-24", "12-12"],
    2017: ["02-01", "03-15", "05-03", "06-14", "07-26", "09-20", "11-01", "12-13"],
}
days_of = lambda ys: [pd.Timestamp(f"{y}-{d}").date() for y in ys for d in SCHED[y]]

loc = pd.read_csv("data/ext_fomc_dates.csv", parse_dates=["dt_broker"])
loc_days = sorted(set(loc["dt_broker"].dt.date))

df = load_mt5_csv("data/vantage_usdjpy_m5.csv").loc["2003-01-01":]
lr = np.log(df["close"]).diff()
by_day = {d: g for d, g in df.groupby(df.index.date)}


def trade(day, h):
    g = by_day.get(day)
    if g is None or len(g) < 50:
        return None
    w = g.between_time(WIN_LO, WIN_HI)
    if len(w) < 20:
        return None
    r = lr.reindex(w.index).abs()
    if not np.isfinite(r).any():
        return None
    i = df.index.get_loc(r.idxmax())
    t_out = df.index[i] + pd.Timedelta(minutes=h)
    if t_out > df.index.max():
        return None
    j = df.index.searchsorted(t_out, side="left")
    if j <= i or j >= len(df):
        return None
    pe, px = df["close"].iloc[i], df["close"].iloc[j - 1]
    if not np.isfinite(pe) or not np.isfinite(px) or pe <= 0:
        return None
    return np.log((px - COST) / pe)


ERAS = [
    ("2004-2006 利上げ", days_of([2004, 2005, 2006]), "タカ"),
    ("2008-2012 QE",     days_of([2008, 2009, 2010, 2011, 2012]), "ハト"),
    ("2017-2018 利上げ", days_of([2017]) + [d for d in loc_days if d.year == 2018], "タカ"),
    ("2019-2026 緩和",   [d for d in loc_days if d.year >= 2019], "ハト"),
]

print(f"USDJPY m5 {df.index.min().date()}..{df.index.max().date()}   マイナス=ドル安")
print(f"{'era':>18} {'政策':>5} {'H':>4} {'n':>3} | {'中央値%':>8} {'平均%':>7} {'ドル安率%':>9} "
      f"| {'対照中央値%':>11} {'対照n':>6} {'下側%ile':>8}")
for label, days, stance in ERAS:
    days = sorted(set(days))
    ex = set(days)
    ctl = [d for d in by_day
           if min(days) <= d <= max(days) and d.weekday() < 5
           and not any(abs((d - x).days) <= 2 for x in ex)]
    for h in [60, 240]:
        r = np.array([x for x in (trade(d, h) for d in days) if x is not None])
        c = np.array([x for x in (trade(d, h) for d in ctl) if x is not None])
        if len(r) < 5 or len(c) < 30:
            print(f"{label:>18} {stance:>5} {h:>4} {len(r):>3} | n不足")
            continue
        d_ = rng.integers(0, len(c), size=(B, len(r)))
        lower = np.mean(np.median(c[d_], axis=1) > np.median(r)) * 100
        print(f"{label:>18} {stance:>5} {h:>4} {len(r):>3} | {np.median(r)*100:+8.3f} {r.mean()*100:+7.3f} "
              f"{np.mean(r < 0)*100:9.1f} | {np.median(c)*100:+11.3f} {len(c):>6} {lower:8.1f}")

print("\n=== 年別（H=60, 中央値% と ドル安率%）— スイッチが切り替わる様子 ===")
rows = []
for label, days, stance in ERAS:
    for d in sorted(set(days)):
        v = trade(d, 60)
        if v is not None:
            rows.append({"yr": d.year, "stance": stance, "lr": v})
t = pd.DataFrame(rows)
g = t.groupby(["stance", "yr"])["lr"].agg(n="count", 中央値=lambda s: s.median() * 100,
                                          ドル安率=lambda s: (s < 0).mean() * 100)
print(g.round(3).to_string())
print(f"\nハト期 中央値がマイナスの年: "
      f"{(g.loc['ハト']['中央値'] < 0).sum()} / {len(g.loc['ハト'])}   "
      f"タカ期: {(g.loc['タカ']['中央値'] < 0).sum()} / {len(g.loc['タカ'])}")
