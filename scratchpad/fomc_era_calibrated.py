"""Era comparison with the broker clock calibrated per year, anchors derived, not guessed.

broker_clock_calibration.py established that USDJPY m5 before ~2013 sits ONE HOUR behind the
modern convention (cross-correlation of the intraday volatility profile against 2019-2026:
+1h for 2004-2006 and 2008-2012, 0h for 2017-2026, in both EU summer and winter).

That also showed the earlier "2017-2018 winter anchor = 20:00" was NOT a clock problem (that
era needs no shift) but a detection failure on ~8 meetings -- and the "2017-2018 shows nothing"
conclusion was resting on it. So stop learning anchors from noisy modes. Compute the anchor
from the published release time in ET, convert it, apply the calibrated per-year offset, then
allow a narrow +/-20 min re-detection (tolerates residual drift, excludes a presser at +30).

Release time in ET: 14:15 before 2013, 14:00 from 2013 (the 2011-2012 press-conference meetings
released at 12:30 and will show up as anchor misses -- counted and reported, not hidden).
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv

COST = 0.009
B = 10000
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
loc = pd.read_csv("data/ext_fomc_dates.csv", parse_dates=["dt_broker"])
loc_days = sorted(set(loc["dt_broker"].dt.date))
ALL_DAYS = sorted({pd.Timestamp(f"{y}-{d}").date() for y, ds in SCHED.items() for d in ds}
                  | set(loc_days))

df = load_mt5_csv("data/vantage_usdjpy_m5.csv").loc["2003-01-01":]
lr = np.log(df["close"]).diff()
absr = lr.abs() * 1e4
by_day = {d: g for d, g in df.groupby(df.index.date)}

# ---- per-year clock offset vs the 2019-2026 reference profile ---------------------
s = absr.loc["2019-01-01":].dropna()
s = s[s.index.dayofweek < 5]
ref = s.groupby(s.index.hour).mean()
ref_c = (ref - ref.mean()).values
OFF = {}
for y in range(2003, 2027):
    v = absr.loc[f"{y}-01-01":f"{y}-12-31"].dropna()
    v = v[v.index.dayofweek < 5]
    if len(v) < 5000:
        continue
    p = v.groupby(v.index.hour).mean().reindex(range(24))
    b = (p - p.mean()).values
    OFF[y] = max(range(-3, 4), key=lambda k: np.nansum(ref_c * np.roll(b, k)))
print("=== 年ごとのブローカー時計のずれ（+1 = その年のデータは1時間遅れている）===")
print("  " + "  ".join(f"{y}:{o:+d}" for y, o in sorted(OFF.items())))


def anchor(day):
    """Published ET release time -> broker wall clock, corrected by the measured offset."""
    et = "14:15" if day.year < 2013 else "14:00"
    t = pd.Timestamp(f"{day} {et}").tz_localize("America/New_York")
    b = t.tz_convert("Europe/Riga").tz_localize(None)
    # OFF=+1 means that era's timestamps sit an hour EARLIER than the modern
    # convention for the same real event, so convert modern -> that era by subtracting.
    return (b - pd.Timedelta(hours=OFF.get(day.year, 0))).tz_localize("UTC")


def trade(day, h, report_miss=None):
    g = by_day.get(day)
    if g is None or len(g) < 50:
        return None
    a = anchor(day)
    w = g.between_time((a - pd.Timedelta(minutes=20)).time(), (a + pd.Timedelta(minutes=20)).time())
    if len(w) < 4:
        return None
    r = absr.reindex(w.index)
    if not np.isfinite(r).any():
        return None
    t_rel = r.idxmax()
    if report_miss is not None:
        report_miss.append((t_rel - a).total_seconds() / 60)
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


# ---- sanity: does the anchor actually sit on a volatility spike? -------------------
print("\n=== アンカーの健全性: FOMC日のアンカー±20分の平均|5分リターン| vs 対照日 (bp) ===")
ERAS = [("2004-2006", 2004, 2006), ("2008-2012", 2008, 2012),
        ("2017-2018", 2017, 2018), ("2019-2026", 2019, 2026)]
for label, y0, y1 in ERAS:
    days = [d for d in ALL_DAYS if y0 <= d.year <= y1]
    ctl = [d for d in by_day if y0 <= d.year <= y1 and d.weekday() < 5
           and not any(abs((d - x).days) <= 2 for x in days)]
    def band(dd):
        out = []
        for d in dd:
            g = by_day.get(d)
            if g is None:
                continue
            a = anchor(d)
            w = g.between_time((a - pd.Timedelta(minutes=20)).time(),
                               (a + pd.Timedelta(minutes=20)).time())
            v = absr.reindex(w.index).dropna()
            if len(v):
                out.append(v.mean())
        return np.mean(out) if out else np.nan
    print(f"  {label}: FOMC {band(days):5.2f} bp  vs 対照 {band(ctl):5.2f} bp  "
          f"→ 差 {band(days)-band(ctl):+5.2f}")

# ---- the era test -----------------------------------------------------------------
print("\n=== 較正済みアンカーでの時代比較  マイナス=ドル安 ===")
print(f"{'era':>12} {'H':>4} {'n':>3} | {'中央値%':>8} {'平均%':>7} {'ドル安率%':>9} "
      f"| {'対照中央値%':>11} {'下側%ile':>8} | {'アンカー誤差(分)中央値':>10}")
for label, y0, y1 in ERAS:
    days = [d for d in ALL_DAYS if y0 <= d.year <= y1]
    ctl = [d for d in by_day if y0 <= d.year <= y1 and d.weekday() < 5
           and not any(abs((d - x).days) <= 2 for x in days)]
    miss = []
    for h in [60, 240]:
        m = [] if h == 60 else None
        r = np.array([x for x in (trade(d, h, m) for d in days) if x is not None])
        c = np.array([x for x in (trade(d, h) for d in ctl) if x is not None])
        if m:
            miss = m
        if len(r) < 5 or len(c) < 30:
            continue
        d_ = rng.integers(0, len(c), size=(B, len(r)))
        lower = np.mean(np.median(c[d_], axis=1) > np.median(r)) * 100
        mm = f"{np.median(miss):+.0f}" if miss else "-"
        print(f"{label:>12} {h:>4} {len(r):>3} | {np.median(r)*100:+8.3f} {r.mean()*100:+7.3f} "
              f"{np.mean(r < 0)*100:9.1f} | {np.median(c)*100:+11.3f} {lower:8.1f} | {mm:>10}")
