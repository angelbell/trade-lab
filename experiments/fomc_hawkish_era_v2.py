"""Hawkish vs dovish era, with the release located per-meeting instead of assumed.

v1 exposed the problem: the volatility profile puts the 2017 release at broker 21:00 (= 14:00
ET, exactly right, so the method works) but the 2004-2006 release at broker 20:15, an hour off
any wall-clock story -- the broker's historical timezone convention is not constant. Assuming
a clock would silently smear the window and could manufacture or destroy the result.

So: for every meeting, find the 5-min bar with the largest |return| inside a wide afternoon
window and treat THAT as the release. Entry = that bar's close, exit H minutes later. Controls
get the identical treatment (their own largest afternoon bar), so both sides are conditioned on
"a big move happened", and nothing depends on timezone bookkeeping.

2019-2026 is re-measured the same way so the era comparison is method-consistent.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv

COST = 0.009
B = 10000
rng = np.random.default_rng(42)
WIN_LO, WIN_HI = "18:00", "23:00"     # broker clock, wide enough to contain every era's release

HAWK = {
    2004: ["01-28", "03-16", "05-04", "06-30", "08-10", "09-21", "11-10", "12-14"],
    2005: ["02-02", "03-22", "05-03", "06-30", "08-09", "09-20", "11-01", "12-13"],
    2006: ["01-31", "03-28", "05-10", "06-29", "08-08", "09-20", "10-25", "12-12"],
    2017: ["02-01", "03-15", "05-03", "06-14", "07-26", "09-20", "11-01", "12-13"],
}
hawk_days = [pd.Timestamp(f"{y}-{d}").date() for y, ds in HAWK.items() for d in ds]

loc = pd.read_csv("data/ext_fomc_dates.csv", parse_dates=["dt_broker"])
loc_days = sorted(set(loc["dt_broker"].dt.date))
days_2018 = [d for d in loc_days if d.year == 2018]
days_dovish = [d for d in loc_days if d.year >= 2019]

df = load_mt5_csv("data/vantage_usdjpy_m5.csv").loc["2003-01-01":]
lr = np.log(df["close"]).diff()
by_day = {}
for d, g in df.groupby(df.index.date):
    by_day[d] = g

print(f"USDJPY m5 {df.index.min().date()}..{df.index.max().date()}  "
      f"利上げ期 {len(hawk_days)}会合 + 2018 {len(days_2018)} / 緩和期 {len(days_dovish)}")


def release_trade(day, h):
    """Largest |5-min move| in the afternoon window = the release. Enter at its close."""
    g = by_day.get(day)
    if g is None or len(g) < 50:
        return None
    w = g.between_time(WIN_LO, WIN_HI)
    if len(w) < 20:
        return None
    r = lr.reindex(w.index).abs()
    if not np.isfinite(r).any():
        return None
    t_rel = r.idxmax()
    i = df.index.get_loc(t_rel)
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


def controls(lo, hi, exclude):
    ex = set(exclude)
    out = []
    for d in by_day:
        if lo <= d <= hi and d.weekday() < 5 and not any(abs((d - x).days) <= 2 for x in ex):
            out.append(d)
    return out


ERAS = [
    ("2004-2006 利上げ", [d for d in hawk_days if d.year <= 2006]),
    ("2017-2018 利上げ", [d for d in hawk_days if d.year == 2017] + days_2018),
    ("利上げ期 合計", [d for d in hawk_days] + days_2018),
    ("2019-2026 緩和", days_dovish),
]

print(f"\n{'era':>18} {'H':>4} {'n':>3} | {'中央値%':>8} {'平均%':>7} {'ドル安率%':>9} | "
      f"{'対照中央値%':>11} {'対照n':>6} {'下側%ile':>8}")
for label, days in ERAS:
    days = sorted(set(days))
    ctl = controls(min(days), max(days), days)
    for h in [60, 240]:
        r = np.array([x for x in (release_trade(d, h) for d in days) if x is not None])
        c = np.array([x for x in (release_trade(d, h) for d in ctl) if x is not None])
        if len(r) < 5 or len(c) < 30:
            print(f"{label:>18} {h:>4} {len(r):>3} | n不足")
            continue
        d_ = rng.integers(0, len(c), size=(B, len(r)))
        lower = np.mean(np.median(c[d_], axis=1) > np.median(r)) * 100
        print(f"{label:>18} {h:>4} {len(r):>3} | {np.median(r)*100:+8.3f} {r.mean()*100:+7.3f} "
              f"{np.mean(r < 0)*100:9.1f} | {np.median(c)*100:+11.3f} {len(c):>6} {lower:8.1f}")

print("\n=== 検出された発表足の時刻分布（ブローカー時計・手法の健全性確認）===")
for label, days in ERAS:
    ts = []
    for d in sorted(set(days)):
        g = by_day.get(d)
        if g is None:
            continue
        w = g.between_time(WIN_LO, WIN_HI)
        if len(w) < 20:
            continue
        r = lr.reindex(w.index).abs()
        if np.isfinite(r).any():
            ts.append(r.idxmax().time())
    if ts:
        s = pd.Series(ts).value_counts().head(3)
        print(f"  {label:>18}: " + " / ".join(f"{t} x{n}" for t, n in s.items()) + f"   (n={len(ts)})")
