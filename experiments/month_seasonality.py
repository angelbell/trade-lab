"""month_seasonality.py -- 夏枯れ/month-of-year seasonality. Two layers:

1. MARKET: per calendar month -- relative vol (month ATR%/median of its year, removes the
   secular vol drift), trend efficiency ER (|net|/path on daily closes), share of years the
   month closed green. gold h1 2007->2026 (19yr, n=19/month), BTC h1 2017-> (9yr).
2. LEGS: adopted 3 legs (get_legs). Monthly R sums by (year, month) -> per-month totals,
   mean, green-year share. Nulls by permuting month labels WITHIN each year (preserves
   yearly totals & count structure), 4000 perms:
     - pre-registered (user-named): is Jul+Aug total R LOWER than permutation null? (夏枯れ)
     - exploratory: is the observed WORST month worse than the null's worst month?
       (12-way selection handled by comparing max-deviation to max-deviation)
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from research.portfolio_kama import get_legs

rng = np.random.default_rng(7)


def market(name, csv):
    d = load_mt5_csv(csv)
    dd = d["close"].resample("1D").last().dropna()
    atr = ta.atr(d["high"], d["low"], d["close"], 14) / d["close"] * 100
    ma = atr.resample("ME").mean().dropna()
    rel = ma / ma.groupby(ma.index.year).transform("median")
    per = pd.DataFrame({"relvol": rel})
    per["m"] = per.index.month
    er, ret = {}, {}
    for (y, m), g in dd.groupby([dd.index.year, dd.index.month]):
        if len(g) < 15:
            continue
        path = g.diff().abs().sum()
        er.setdefault(m, []).append(abs(g.iloc[-1] - g.iloc[0]) / path if path > 0 else np.nan)
        ret.setdefault(m, []).append(g.iloc[-1] / g.iloc[0] - 1)
    span = (d.index[-1] - d.index[0]).days / 365.25
    print(f"\n===== {name} 市場の月別プロファイル ({span:.0f}yr) =====")
    print("  月  相対ボラ   ER(トレンド効率)  月リターン緑率(年数)")
    for m in range(1, 13):
        rv = per.relvol[per.m == m]
        e = np.nanmean(er.get(m, [np.nan]))
        r = ret.get(m, [])
        print(f"  {m:>2}  {rv.mean():5.2f}     {e:5.2f}            "
              f"{np.mean([x > 0 for x in r])*100:3.0f}% ({len(r)})")


def leg_seasonality(name, s):
    mon = s.groupby([s.index.year, s.index.month]).sum()
    rows = [(y, m, v) for (y, m), v in mon.items()]
    yrs = sorted({y for y, m, v in rows})
    span = len(yrs)
    obs = {m: sum(v for y, mm, v in rows if mm == m) for m in range(1, 13)}
    grn = {m: (sum(1 for y, mm, v in rows if mm == m and v > 0),
               sum(1 for y, mm, v in rows if mm == m)) for m in range(1, 13)}
    tot = sum(obs.values())
    print(f"\n----- {name} (月別 totR, {span}年) -----")
    print("  月:  " + " ".join(f"{m:>6}" for m in range(1, 13)))
    print("  R :  " + " ".join(f"{obs[m]:+6.1f}" for m in range(1, 13)))
    print("  緑:  " + " ".join(f"{grn[m][0]:>3}/{grn[m][1]:<2}" for m in range(1, 13)))
    # permutation null: shuffle month labels within each year
    ja_obs = obs[7] + obs[8]
    worst_obs = min(obs.values())
    byyear = {}
    for y, m, v in rows:
        byyear.setdefault(y, []).append((m, v))
    ja_null, worst_null = [], []
    for _ in range(4000):
        tot_m = dict.fromkeys(range(1, 13), 0.0)
        for y, lst in byyear.items():
            ms = [m for m, v in lst]
            vs = [v for m, v in lst]
            perm = rng.permutation(len(vs))
            for k, m in enumerate(ms):
                tot_m[m] += vs[perm[k]]
        ja_null.append(tot_m[7] + tot_m[8])
        worst_null.append(min(tot_m.values()))
    ja_null = np.array(ja_null); worst_null = np.array(worst_null)
    p_ja = (ja_null <= ja_obs).mean()
    p_worst = (worst_null <= worst_obs).mean()
    print(f"  夏枯れ(7-8月計): {ja_obs:+.1f}R (全体{tot:+.1f}Rの{ja_obs/tot*100 if tot else 0:.0f}%) "
          f" null中央値{np.median(ja_null):+.1f}  p(これ以下)={p_ja:.3f}")
    print(f"  最悪月: {min(obs, key=obs.get)}月 {worst_obs:+.1f}R  null最悪の中央値{np.median(worst_null):+.1f}"
          f"  p={p_worst:.3f}  (12択選択込みの公平比較)")


def main():
    market("GOLD", "data/vantage_xauusd_h1.csv")
    market("BTC", "data/vantage_btcusd_h1.csv")
    print("\n===== 採用レッグの月別R =====")
    for k, t in get_legs().items():
        s = pd.Series(t.R.values, index=pd.DatetimeIndex(t.time))
        leg_seasonality(k, s)


if __name__ == "__main__":
    main()
