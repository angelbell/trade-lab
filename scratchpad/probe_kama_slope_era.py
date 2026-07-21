"""時代ベータ隔離: kama_slope の上位分位(Q5)の優位が特定年に固まっていないか。
土台スクリプトを import し、トレードごとの (R, kama_slope, year) を復元して年別に割る。"""
import os, sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strength_btc15mL as base
import strength_regime_btc15mL as reg

# 土台再構築（照合ゲート通過済みの経路）
d15, raw, args, tL, netR = base.build(smoke=False)
entries, t2 = base.rebuild_entries(d15, args)
i_arr = base.match_entries_to_trades(entries, tL, args.pullback_frac)
R = tL["R"].values - 15.0 / tL["risk"].values  # netコスト（PDH重みは強度の議論に無関係なので素netR）
ks_arr, _ = reg.compute_kama_slope(d15, n=args.gate_kama, tf=args.gate_kama_tf)
ks = ks_arr[i_arr]
yr = pd.DatetimeIndex(tL["time"]).year.values

df = pd.DataFrame({"R": R, "ks": ks, "y": yr})
# 全体の5分位境界でラベル付け（年またぎで一貫した閾値）
df["Q"] = pd.qcut(df["ks"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)

print("=== 年別 Q5(急KAMA) vs Q1(緩KAMA) meanR / n ===")
print(f"{'年':>6} | {'Q5 n':>5} {'Q5 meanR':>9} | {'Q1 n':>5} {'Q1 meanR':>9} | {'Q5-Q1':>7} | {'全体n':>5}")
for y in sorted(df["y"].unique()):
    s = df[df["y"] == y]
    q5, q1 = s[s["Q"]==5], s[s["Q"]==1]
    m5 = q5["R"].mean() if len(q5) else float("nan")
    m1 = q1["R"].mean() if len(q1) else float("nan")
    diff = m5 - m1 if (len(q5) and len(q1)) else float("nan")
    print(f"{y:>6} | {len(q5):>5} {m5:>+9.3f} | {len(q1):>5} {m1:>+9.3f} | {diff:>+7.3f} | {len(s):>5}")

# Q5-Q1 が正の年の数（符号一貫性）
per_year = []
for y in sorted(df["y"].unique()):
    s = df[df["y"] == y]
    q5, q1 = s[s["Q"]==5], s[s["Q"]==1]
    if len(q5) and len(q1):
        per_year.append(q5["R"].mean() - q1["R"].mean())
per_year = np.array(per_year)
print(f"\nQ5-Q1 が正の年: {(per_year>0).sum()}/{len(per_year)}  中央値={np.median(per_year):+.3f}")

# Q5 のトレードが何年に集中しているか
print("\n=== Q5(急KAMA)トレードの年別カウント（集中度）===")
print(df[df["Q"]==5].groupby("y").size().to_string())
print(f"\nQ5総数={ (df['Q']==5).sum() }  上位2年で占める割合="
      f"{df[df['Q']==5].groupby('y').size().nlargest(2).sum()/(df['Q']==5).sum():.1%}")
