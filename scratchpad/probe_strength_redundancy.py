"""冗長性チェック: btc15m_L の3強度候補 kama_slope / stop_atr / atr_pctile が
同じ1軸か独立軸か。(1)変数間Spearman相関 (2)上位20%メンバーの重なり(Jaccard)
(3)片方の上位分位に固定した中で他方がまだEVを並べ替えるか(条件付き)。"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strength_btc15mL as base
import strength_regime_btc15mL as reg
import strength_entryquality_btc15mL as eq

d15, raw, args, tL, netR = base.build(smoke=False)
entries, t2 = base.rebuild_entries(d15, args)
i_arr = base.match_entries_to_trades(entries, tL, args.pullback_frac)
R = tL["R"].values - 15.0 / tL["risk"].values

ks, _ = reg.compute_kama_slope(d15, n=args.gate_kama, tf=args.gate_kama_tf)
atr = eq.compute_atr(d15, args.atr) if hasattr(eq, "compute_atr") else None
# eq のATR計算を流用できなければローカルで（同一シグネチャ）
if atr is None:
    import pandas_ta as ta
    atr = ta.atr(d15["high"], d15["low"], d15["close"], length=args.atr).values

kama_slope = ks[i_arr]
stop_atr = tL["risk"].values / atr[i_arr]
# atr_pctile: trailing 500 percentile at i
atr_s = pd.Series(atr)
atr_pctile_full = atr_s.rolling(500).apply(lambda w: (w[-1] > w[:-1]).mean(), raw=True).values
atr_pctile = atr_pctile_full[i_arr]

df = pd.DataFrame({"R": R, "kama_slope": kama_slope, "stop_atr": stop_atr,
                   "atr_pctile": atr_pctile}).dropna()
print(f"n={len(df)}")
vars3 = ["kama_slope", "stop_atr", "atr_pctile"]
print("\n(1) 変数間 Spearman 相関:")
for a in range(3):
    for b in range(a+1, 3):
        rho, p = spearmanr(df[vars3[a]], df[vars3[b]])
        print(f"  {vars3[a]:>11} × {vars3[b]:<11} rho={rho:+.3f} (p={p:.2e})")

print("\n(2) 上位20%メンバーの重なり (Jaccard):")
top = {v: set(df[df[v] >= df[v].quantile(0.8)].index) for v in vars3}
for a in range(3):
    for b in range(a+1, 3):
        A, B = top[vars3[a]], top[vars3[b]]
        j = len(A & B) / len(A | B)
        print(f"  {vars3[a]:>11} ∩ {vars3[b]:<11} Jaccard={j:.2f} (共通{len(A&B)}本/和集合{len(A|B)})")

print("\n(3) 条件付き: 各変数の上位20%(強)に固定した中で、他変数がまだEVを分けるか")
for cond in vars3:
    sub = df[df[cond] >= df[cond].quantile(0.8)]
    print(f"  [{cond} 強(上位20%, n={len(sub)}) の中で]")
    for other in vars3:
        if other == cond: continue
        rho, p = spearmanr(sub[other], sub["R"])
        print(f"      {other:>11} vs R: rho={rho:+.3f} (p={p:.2e})")

print("\n(4) 単変数 vs 3変数平均ランクスコア の Spearman(対R):")
for v in vars3:
    rho, _ = spearmanr(df[v], df["R"]); print(f"    {v:>11}: {rho:+.3f}")
combo = sum(df[v].rank(pct=True) for v in vars3) / 3
rho, p = spearmanr(combo, df["R"]); print(f"    combo(3平均) : {rho:+.3f} (p={p:.2e})")
# kama_slope+atr_pctile のみ（stop_atr抜き）
combo2 = (df["kama_slope"].rank(pct=True) + df["atr_pctile"].rank(pct=True)) / 2
rho2, _ = spearmanr(combo2, df["R"]); print(f"    combo(ks+atrpct): {rho2:+.3f}")
