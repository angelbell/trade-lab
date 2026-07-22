"""Is the post-FOMC window special, or special ONLY while gold trends up? And what drives it?

Established: buying gold at the statement and holding 1-8h beats the same clock window on
non-FOMC weekdays at the 100th percentile, median +0.13% to +0.36%, positive median in 8/8
years at H=480. But 2019-2026 is one gold regime, so "the window is special" and "the window
is special in a gold bull market" are not yet separated.

Two tests, both on local data:
  1. REGIME SPLIT -- daily SMA150 state of gold on the day before the meeting (no lookahead).
     If the window only pays while gold is above its 150-day average, it is a trend bet with
     an FOMC trigger, not an FOMC effect.
  2. USDJPY -- if gold rises post-statement because the DOLLAR falls, USDJPY must fall in the
     same window. USDJPY trended dollar-UP over 2018-2026, so a consistent dollar-down print
     there runs AGAINST that instrument's drift = evidence the window itself carries it.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from experiments.fomc_event_study import price_before, candidate_dates

HS = [60, 240, 480]
B = 10000
rng = np.random.default_rng(42)

ev = pd.read_csv("experiments/fomc_stmt_2019.csv", parse_dates=["dt_utc", "dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())


def make_trade(df, cost):
    t_end = df.index.max()
    def trade(ts, h):
        t_in = ts + pd.Timedelta(minutes=1)
        t_out = t_in + pd.Timedelta(minutes=h)
        if t_out > t_end:
            return None
        pe, px = price_before(df, t_in), price_before(df, t_out)
        if pe is None or px is None or not np.isfinite(pe) or not np.isfinite(px) or pe <= 0:
            return None
        return np.log((px - cost) / pe)
    return trade


def med_pct(r, c):
    d = rng.integers(0, len(c), size=(B, len(r)))
    return np.mean(np.median(c[d], axis=1) < np.median(r)) * 100


# ---------------------------------------------------------------- gold regime split
gold = load_mt5_csv("data/vantage_xauusd_m1.csv")
tr_g = make_trade(gold, 0.30)
daily = gold["close"].resample("1D").last().dropna()
sma150 = daily.rolling(150).mean()

def regime(ts):
    """gold's daily trend state using ONLY bars closed before the meeting day."""
    prev = daily.index[daily.index < ts.normalize()]
    if len(prev) == 0:
        return None
    d = prev[-1]
    if not np.isfinite(sma150.get(d, np.nan)):
        return None
    return "上昇(>150日線)" if daily[d] > sma150[d] else "下降(<150日線)"

cand = candidate_dates(gold, events)
clock = events[0].time()
cand_ts = [pd.Timestamp.combine(d.date(), clock).tz_localize("UTC") for d in cand]

print("=== 1. gold: 日足トレンド・レジーム別 ===")
print(f"{'レジーム':>16} {'H':>4} {'n':>3} | {'中央値%':>8} {'勝率%':>6} | {'対照中央値%':>11} {'対照n':>6} {'%ile':>6}")
for reg in ["上昇(>150日線)", "下降(<150日線)"]:
    sel = [e for e in events if regime(e) == reg]
    csel = [t for t in cand_ts if regime(t) == reg]
    for h in HS:
        r = np.array([x for x in (tr_g(t, h) for t in sel) if x is not None])
        c = np.array([x for x in (tr_g(t, h) for t in csel) if x is not None])
        if len(r) < 5 or len(c) < 30:
            print(f"{reg:>16} {h:>4} {len(r):>3} | n不足")
            continue
        print(f"{reg:>16} {h:>4} {len(r):>3} | {np.median(r)*100:+8.3f} {np.mean(r > 0)*100:6.1f} | "
              f"{np.median(c)*100:+11.3f} {len(c):>6} {med_pct(r, c):6.1f}")

# ---------------------------------------------------------------- USDJPY (dollar side)
uj = load_mt5_csv("data/vantage_usdjpy_m1.csv")
tr_u = make_trade(uj, 0.009)
cand_u = candidate_dates(uj, events)
cand_ts_u = [pd.Timestamp.combine(d.date(), clock).tz_localize("UTC") for d in cand_u]

print("\n=== 2. USDJPY: 同じ窓（プラス=ドル高。gold上昇の裏がドル安なら負になるはず）===")
print(f"{'H':>4} {'n':>3} | {'中央値%':>8} {'平均%':>7} {'勝率%':>6} | {'対照中央値%':>11} {'%ile(下側)':>10}")
for h in HS:
    r = np.array([x for x in (tr_u(t, h) for t in events) if x is not None])
    c = np.array([x for x in (tr_u(t, h) for t in cand_ts_u) if x is not None])
    d = rng.integers(0, len(c), size=(B, len(r)))
    lower = np.mean(np.median(c[d], axis=1) > np.median(r)) * 100   # how extreme on the DOWN side
    print(f"{h:>4} {len(r):>3} | {np.median(r)*100:+8.3f} {r.mean()*100:+7.3f} {np.mean(r > 0)*100:6.1f} | "
          f"{np.median(c)*100:+11.3f} {lower:10.1f}")

print("\n=== 3. 同じ会合での gold と USDJPY の符号の対応（H=240）===")
pair = [(tr_g(e, 240), tr_u(e, 240)) for e in events]
pair = [(a, b) for a, b in pair if a is not None and b is not None]
g_, u_ = np.array([p[0] for p in pair]), np.array([p[1] for p in pair])
print(f"n={len(pair)}  相関(順位) = {pd.Series(g_).corr(pd.Series(u_), method='spearman'):+.3f}")
print(f"  gold↑かつドル安(USDJPY↓): {np.mean((g_ > 0) & (u_ < 0))*100:.1f}%   "
      f"gold↓かつドル高: {np.mean((g_ < 0) & (u_ > 0))*100:.1f}%   "
      f"→ 逆相関で説明できる割合 合計 {np.mean(((g_ > 0) & (u_ < 0)) | ((g_ < 0) & (u_ > 0)))*100:.1f}%")
