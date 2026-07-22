"""Era comparison with the press conference excluded from the anchor.

fomc_era_switch.py found the split is NOT hawkish/dovish (QE 2008-2012 shows nothing, hiking
2017-2018 shows the effect) but old/recent. Before believing that, one confound has to go:
the anchor is "largest |5-min move| in the afternoon", and press conferences only exist in the
recent era (from 2011 for 4 meetings/yr, from 2019 for all). Measured: in 2019-2026 the anchor
lands on the presser bar (21:35/21:40) 18 times out of 58. Old eras have no presser to land on,
so the eras may be anchored on different events entirely.

Fix: two stages. First learn each era's modal release time from the data, separately for EU
summer and EU winter (the broker's offset shifts). Then re-detect inside a +/-25 min band
around that modal time -- wide enough to absorb any residual clock drift, narrow enough to
exclude a presser that starts 30-40 min after the statement.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv

COST = 0.009
B = 10000
BAND = 25
rng = np.random.default_rng(42)

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
summer = lambda d: 4 <= d.month <= 10


def peak(day, lo="18:00", hi="23:00"):
    g = by_day.get(day)
    if g is None or len(g) < 50:
        return None
    w = g.between_time(lo, hi)
    if len(w) < 10:
        return None
    r = lr.reindex(w.index).abs()
    return None if not np.isfinite(r).any() else r.idxmax()


def modal(days, season):
    ts = [peak(d) for d in days if season(d)]
    ts = [t.time() for t in ts if t is not None]
    return pd.Series(ts).value_counts().index[0] if ts else None


def trade(day, h, anchor):
    """Anchor = era modal release time; re-detect the peak inside +/-BAND minutes of it."""
    if anchor is None:
        return None
    base = pd.Timestamp.combine(day, anchor)
    lo = (base - pd.Timedelta(minutes=BAND)).time()
    hi = (base + pd.Timedelta(minutes=BAND)).time()
    t_rel = peak(day, lo.strftime("%H:%M"), hi.strftime("%H:%M"))
    if t_rel is None:
        return None
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


ERAS = [
    ("2004-2006 利上げ", days_of([2004, 2005, 2006]), "タカ"),
    ("2008-2012 QE", days_of([2008, 2009, 2010, 2011, 2012]), "ハト"),
    ("2017-2018 利上げ", days_of([2017]) + [d for d in loc_days if d.year == 2018], "タカ"),
    ("2019-2026 緩和", [d for d in loc_days if d.year >= 2019], "ハト"),
]

print("=== 各時代の発表時刻（ブローカー時計・夏/冬別に学習）===")
anch = {}
for label, days, _ in ERAS:
    days = sorted(set(days))
    a_s, a_w = modal(days, summer), modal(days, lambda d: not summer(d))
    anch[label] = (a_s, a_w)
    print(f"  {label:>18}: 夏 {a_s}  冬 {a_w}")

print(f"\n=== 会見を除外（発表時刻±{BAND}分に限定）した時代比較  マイナス=ドル安 ===")
print(f"{'era':>18} {'政策':>5} {'H':>4} {'n':>3} | {'中央値%':>8} {'平均%':>7} {'ドル安率%':>9} "
      f"| {'対照中央値%':>11} {'下側%ile':>8}")
for label, days, stance in ERAS:
    days = sorted(set(days))
    a_s, a_w = anch[label]
    ex = set(days)
    ctl = [d for d in by_day if min(days) <= d <= max(days) and d.weekday() < 5
           and not any(abs((d - x).days) <= 2 for x in ex)]
    for h in [60, 240]:
        f = lambda d: trade(d, h, a_s if summer(d) else a_w)
        r = np.array([x for x in map(f, days) if x is not None])
        c = np.array([x for x in map(f, ctl) if x is not None])
        if len(r) < 5 or len(c) < 30:
            print(f"{label:>18} {stance:>5} {h:>4} {len(r):>3} | n不足")
            continue
        d_ = rng.integers(0, len(c), size=(B, len(r)))
        lower = np.mean(np.median(c[d_], axis=1) > np.median(r)) * 100
        print(f"{label:>18} {stance:>5} {h:>4} {len(r):>3} | {np.median(r)*100:+8.3f} {r.mean()*100:+7.3f} "
              f"{np.mean(r < 0)*100:9.1f} | {np.median(c)*100:+11.3f} {lower:8.1f}")
