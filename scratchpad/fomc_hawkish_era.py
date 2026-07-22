"""Does the post-FOMC dollar-down drift survive a HAWKISH era?

The 2019-2026 result (gold up / dollar down for 1-8h after the statement, 100th percentile vs
same-clock control, present in both gold trend regimes) has one unresolved risk: 2019-2026 was
a systematically dovish stretch (COVID QE -> "transitory" -> the 2024-26 cutting cycle). A
dovish-surprise bias produces exactly that signature.

So test it where the Fed was HIKING: 2004-2006 (17 consecutive hikes) and 2017-2018.
Mechanism is a dollar factor (gold x USDJPY rank correlation -0.857 over the same meetings),
so USDJPY alone settles it -- and USDJPY m5 goes back to 2000 while gold m1 starts 2019.

STEP 1 does NOT trust my memory of the release time (2:15pm ET pre-2013, 2:00pm after). It
locates the release empirically from the volatility profile: on statement days the market jumps
at the release minute and control days do not. The anchor used in STEP 2 is the measured one.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv

COST = 0.009            # USDJPY round trip ~0.9 pip
HS = [60, 240, 480]
B = 10000
rng = np.random.default_rng(42)

HAWKISH = {
    2004: ["2004-01-28", "2004-03-16", "2004-05-04", "2004-06-30",
           "2004-08-10", "2004-09-21", "2004-11-10", "2004-12-14"],
    2005: ["2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30",
           "2005-08-09", "2005-09-20", "2005-11-01", "2005-12-13"],
    2006: ["2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29",
           "2006-08-08", "2006-09-20", "2006-10-25", "2006-12-12"],
    2017: ["2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
           "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13"],
}

df = load_mt5_csv("data/vantage_usdjpy_m5.csv")
df = df.loc["2000-01-01":]
print(f"USDJPY m5: {len(df)}本  {df.index.min()} .. {df.index.max()}")


def to_broker(day, et_hour, et_min):
    """FOMC wall-clock in ET -> broker wall clock (Europe/Riga), labelled UTC like the CSV index."""
    t = pd.Timestamp(f"{day} {et_hour:02d}:{et_min:02d}").tz_localize("America/New_York")
    return pd.Timestamp(t.tz_convert("Europe/Riga").tz_localize(None)).tz_localize("UTC")


# ---------------------------------------------------------------- STEP 1: find the release
print("\n=== STEP 1: 発表時刻をデータから特定（FOMC日 − 対照日 の平均|5分リターン|、単位 bp）===")
for era, yrs in [("2004-2006", [2004, 2005, 2006]), ("2017", [2017])]:
    days = [d for y in yrs for d in HAWKISH[y]]
    sl = df.loc[f"{yrs[0]}-01-01":f"{yrs[-1]}-12-31"]
    r = (np.log(sl["close"]).diff().abs() * 1e4).dropna()
    dd = pd.Series(r.index.normalize().date, index=r.index)
    is_ev = dd.isin([pd.Timestamp(d).date() for d in days])
    prof_ev = r[is_ev].groupby(r[is_ev].index.time).mean()
    prof_ct = r[~is_ev].groupby(r[~is_ev].index.time).mean()
    diff = (prof_ev - prof_ct).dropna().sort_values(ascending=False)
    print(f"  [{era}] 差が大きい時刻トップ5（ブローカー時計）:")
    for t, v in diff.head(5).items():
        et = pd.Timestamp(f"2000-01-01 {t}").tz_localize("Europe/Riga").tz_convert("America/New_York").time()
        print(f"      broker {t}  (= ET {et})   +{v:.1f} bp")

# ---------------------------------------------------------------- STEP 2: the window test
def window(ts, h):
    t_in = ts + pd.Timedelta(minutes=5)
    t_out = t_in + pd.Timedelta(minutes=h)
    if t_out > df.index.max():
        return None
    i = df.index.searchsorted(t_in, side="left")
    j = df.index.searchsorted(t_out, side="left")
    if i == 0 or j >= len(df) or i >= j:
        return None
    pe, px = df["close"].iloc[i - 1], df["close"].iloc[j - 1]
    if not np.isfinite(pe) or not np.isfinite(px) or pe <= 0:
        return None
    return np.log(px / pe), np.log((px - COST) / pe)


print("\n=== STEP 2: 利上げ時代のドル方向（プラス=ドル高。2019-26では中央値 −0.108%）===")
for et_h, et_m, label in [(14, 15, "14:15 ET"), (14, 0, "14:00 ET")]:
    print(f"\n-- アンカー {label} --")
    print(f"{'era':>12} {'n':>3} | {'中央値%':>8} {'平均%':>7} {'ドル安率%':>9} | "
          f"{'対照中央値%':>11} {'対照n':>6} {'下側%ile':>8}")
    for era, yrs in [("2004-2006", [2004, 2005, 2006]), ("2017-2018", [2017]),
                     ("全利上げ期", [2004, 2005, 2006, 2017])]:
        days = [d for y in yrs for d in HAWKISH[y]]
        evs = [to_broker(d, et_h, et_m) for d in days]
        dayset = {pd.Timestamp(d).date() for d in days}
        lo, hi = f"{yrs[0]}-01-01", f"{yrs[-1]}-12-31"
        allw = pd.bdate_range(lo, hi)
        ctrl = [to_broker(str(d.date()), et_h, et_m) for d in allw
                if d.date() not in dayset and not any(abs((d.date() - x).days) <= 2 for x in dayset)]
        for h in [60, 240]:
            r = np.array([x[1] for x in (window(t, h) for t in evs) if x is not None])
            c = np.array([x[1] for x in (window(t, h) for t in ctrl) if x is not None])
            if len(r) < 5 or len(c) < 30:
                continue
            d_ = rng.integers(0, len(c), size=(B, len(r)))
            lower = np.mean(np.median(c[d_], axis=1) > np.median(r)) * 100
            print(f"{era+' H='+str(h):>12} {len(r):>3} | {np.median(r)*100:+8.3f} {r.mean()*100:+7.3f} "
                  f"{np.mean(r < 0)*100:9.1f} | {np.median(c)*100:+11.3f} {len(c):>6} {lower:8.1f}")
