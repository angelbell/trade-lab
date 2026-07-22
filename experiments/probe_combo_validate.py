"""合成強度スコア(kama_slope+stop_atr+atr_pctile の等重みランク平均)を殺しにかける:
バンド表・巡回ブロックブートストラップ(月ブロックでSpearman)・時代ベータ隔離(年別Q5-Q1)。"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strength_btc15mL as base
import strength_regime_btc15mL as reg
import strength_entryquality_btc15mL as eq
import pandas_ta as ta

d15, raw, args, tL, netR = base.build(smoke=False)
entries, t2 = base.rebuild_entries(d15, args)
i_arr = base.match_entries_to_trades(entries, tL, args.pullback_frac)
R = tL["R"].values - 15.0 / tL["risk"].values
times = pd.DatetimeIndex(tL["time"])

ks, _ = reg.compute_kama_slope(d15, n=args.gate_kama, tf=args.gate_kama_tf)
atr = ta.atr(d15["high"], d15["low"], d15["close"], length=args.atr).values
kama_slope = ks[i_arr]
stop_atr = tL["risk"].values / atr[i_arr]
atr_s = pd.Series(atr)
atr_pctile = atr_s.rolling(500).apply(lambda w: (w[-1] > w[:-1]).mean(), raw=True).values[i_arr]

df = pd.DataFrame({"R": R, "t": times, "ks": kama_slope, "sa": stop_atr, "ap": atr_pctile}).dropna().reset_index(drop=True)
df["combo"] = (df["ks"].rank(pct=True) + df["sa"].rank(pct=True) + df["ap"].rank(pct=True)) / 3
yrs = (df["t"].iloc[-1] - df["t"].iloc[0]).days / 365.25
print(f"n={len(df)}  span {df['t'].iloc[0].date()}→{df['t'].iloc[-1].date()} ({yrs:.1f}yr)  全体meanR={df['R'].mean():+.3f}")

print("\n合成スコア 5分位:")
df["Q"] = pd.qcut(df["combo"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
print(f"{'Q':>2} {'n':>4} {'win%':>6} {'PF':>5} {'meanR':>7} {'totR':>7}")
for q in range(1, 6):
    r = df[df["Q"]==q]["R"]
    pf = r[r>0].sum()/-r[r<0].sum() if (r<0).any() else float('inf')
    print(f"{q:>2} {len(r):>4} {(r>0).mean()*100:>5.1f}% {pf:>5.2f} {r.mean():>+7.3f} {r.sum():>+7.1f}")

rho_all, p_all = spearmanr(df["combo"], df["R"])
print(f"\n合成 Spearman(全体) = {rho_all:+.3f} (p={p_all:.2e})")

# 巡回ブロック・ブートストラップ(月ブロックでSpearman)
def block_boot(months, n_boot=2000, seed=20260718):
    rng = np.random.default_rng(seed)
    d = df.copy(); d["mk"] = d["t"].dt.to_period("M")
    keys = d["mk"].unique(); blk = max(1, months)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), size=int(np.ceil(len(keys)/blk)))
        sel = []
        for p in pick: sel.extend(keys[p:p+blk])
        sub = d[d["mk"].isin(sel)]
        if sub["R"].nunique() > 2:
            out.append(spearmanr(sub["combo"], sub["R"])[0])
    g = np.array(out)
    return np.median(g), np.quantile(g, 0.025), np.quantile(g, 0.975)
print("\n合成スコアの巡回ブロック・ブートストラップ(月ブロック,2000回):")
for mo in (1,3,6,12):
    med, lo, hi = block_boot(mo)
    print(f"  {mo:>2}mo: 中央値 {med:+.3f}  95%CI [{lo:+.3f},{hi:+.3f}]  {'0超' if lo>0 else '0またぎ'}")

# 時代ベータ隔離: グローバル分位で年別 Q5-Q1
print("\n時代ベータ隔離 (グローバル分位・年別 Q5-Q1 meanR):")
gaps=[]
for y in sorted(df["t"].dt.year.unique()):
    s = df[df["t"].dt.year==y]
    q5 = s[s["Q"]==5]["R"]; q1 = s[s["Q"]==1]["R"]
    if len(q5) and len(q1):
        g = q5.mean()-q1.mean(); gaps.append(g)
        print(f"  {y}: Q5(n={len(q5)}) {q5.mean():+.3f} | Q1(n={len(q1)}) {q1.mean():+.3f} | 差 {g:+.3f}")
gaps=np.array(gaps)
print(f"  → 正の年 {(gaps>0).sum()}/{len(gaps)}  中央値 {np.median(gaps):+.3f}")
