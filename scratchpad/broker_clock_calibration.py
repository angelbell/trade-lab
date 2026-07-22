"""Calibrate the broker's clock per era using landmarks that have nothing to do with FOMC.

The era comparison is only trustworthy if the FOMC anchor is right in every era, and the
learned anchors contradict themselves: 2017-2018 came out "summer 21:00 / winter 20:00" when
the correct conversion gives 21:00 in both (which is exactly what 2019-2026 shows). So the
broker's historical timestamps do not use one convention throughout, and "the effect is absent
before 2019" could be an artifact of anchoring the old eras an hour off.

Fix it with landmarks that exist every single day and do not move: the intraday volatility
profile of USDJPY. Tokyo open, London open and the NY session are fixed in their own local
clocks, so if an era's whole profile is shifted by an hour relative to another era, that is the
broker offset -- measured, not assumed.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv

df = load_mt5_csv("data/vantage_usdjpy_m5.csv").loc["2003-01-01":]
lr = np.log(df["close"]).diff().abs() * 1e4

ERAS = [("2004-2006", "2004-01-01", "2006-12-31"),
        ("2008-2012", "2008-01-01", "2012-12-31"),
        ("2017-2018", "2017-01-01", "2018-12-31"),
        ("2019-2026", "2019-01-01", "2026-06-17")]

# EU summer / winter separately: the broker offset itself may or may not observe DST
prof = {}
for label, lo, hi in ERAS:
    s = lr.loc[lo:hi].dropna()
    s = s[s.index.dayofweek < 5]
    for season, mask in [("夏", (s.index.month >= 4) & (s.index.month <= 10)),
                         ("冬", ~((s.index.month >= 4) & (s.index.month <= 10)))]:
        v = s[mask]
        prof[(label, season)] = v.groupby(v.index.hour).mean()

print("=== USDJPY 1時間ごとの平均|5分リターン| (bp)  ブローカー時計 ===")
print("     時 | " + " | ".join(f"{l[2:]}{sea}" for l, sea in prof))
for h in range(24):
    row = " | ".join(f"{prof[k].get(h, np.nan):7.2f}" for k in prof)
    print(f"     {h:>2} | {row}")

print("\n=== 各プロファイルの山（上位3時間）と、2019-2026(冬)からのずれ ===")
ref = prof[("2019-2026", "冬")]
ref_top = ref.idxmax()
for k, v in prof.items():
    top3 = v.sort_values(ascending=False).head(3).index.tolist()
    print(f"  {k[0]} {k[1]}: 山 {top3}  最大={v.idxmax()}時  → 基準(2019-26冬 {ref_top}時)とのずれ "
          f"{v.idxmax() - ref_top:+d} 時間")

print("\n=== 相互相関でずれを推定（プロファイルを巡回シフトして一致度が最大になる量）===")
for k, v in prof.items():
    if k == ("2019-2026", "冬"):
        continue
    a = (ref - ref.mean()).values
    b = (v.reindex(range(24)) - v.mean()).values
    best = max(range(-4, 5), key=lambda s: np.nansum(a * np.roll(b, s)))
    print(f"  {k[0]} {k[1]}: 最良シフト {best:+d} 時間")
