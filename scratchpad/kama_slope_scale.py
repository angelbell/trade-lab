"""btc15m_L の kama_slope を実運用の3段スケール(弱/中/強)に落とす。
照合済み土台を再利用。各段: 閾値(slope生値・%/4H足)・n・n/yr・win%・PF・meanR・totR。
強vs弱 meanR ギャップの巡回ブロック・ブートストラップ CI も出す。"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strength_btc15mL as base
import strength_regime_btc15mL as reg

d15, raw, args, tL, netR = base.build(smoke=False)
entries, t2 = base.rebuild_entries(d15, args)
i_arr = base.match_entries_to_trades(entries, tL, args.pullback_frac)
R = (tL["R"].values - 15.0 / tL["risk"].values)  # 素netR(PDH重みは強度議論に無関係)
ks_arr, _ = reg.compute_kama_slope(d15, n=args.gate_kama, tf=args.gate_kama_tf)
ks = ks_arr[i_arr]
times = pd.DatetimeIndex(tL["time"])
yrs = (times[-1] - times[0]).days / 365.25

def band_stats(mask, label, lo, hi):
    r = R[mask]
    n = len(r); wins = (r > 0).sum()
    pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else float("inf")
    print(f"{label:>4} | slope[{lo}] | n={n:>4} ({n/yrs:>4.1f}/yr) | win={wins/n*100:>4.1f}% "
          f"| PF={pf:>4.2f} | meanR={r.mean():>+6.3f} | totR={r.sum():>+7.1f}")
    return r

# 閾値: p40 / p80（弱=下位40%, 中=40-80%, 強=上位20%）
p40, p80 = np.quantile(ks, [0.40, 0.80])
print(f"n={len(ks)}  span {times[0].date()}→{times[-1].date()} ({yrs:.1f}yr)  全体meanR={R.mean():+.3f}")
print(f"閾値(slope生値): p40={p40:.6e}  p80={p80:.6e}")
print(f"閾値(%/4H足=slope*100): p40={p40*100:.4f}%  p80={p80*100:.4f}%\n")

w = band_stats(ks < p40, "弱", "<p40", "")
m = band_stats((ks >= p40) & (ks < p80), "中", "p40-p80", "")
s = band_stats(ks >= p80, "強", ">=p80", "")

gap = s.mean() - w.mean()
print(f"\n強−弱 meanR ギャップ = {gap:+.3f}R")

# 巡回ブロック・ブートストラップ(強band vs 弱band の meanR差, 月ブロック)
def block_boot_gap(months, n_boot=2000, seed=20260718):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"R": R, "ks": ks, "t": times})
    df["mkey"] = df["t"].dt.to_period("M")
    keys = df["mkey"].unique()
    blk = max(1, months)
    gaps = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), size=int(np.ceil(len(keys)/blk)))
        sel = []
        for p in pick:
            sel.extend(keys[p:p+blk])
        sub = df[df["mkey"].isin(sel)]
        sw = sub[sub["ks"] >= p80]["R"]; wk = sub[sub["ks"] < p40]["R"]
        if len(sw) and len(wk):
            gaps.append(sw.mean() - wk.mean())
    g = np.array(gaps)
    return np.median(g), np.quantile(g, 0.025), np.quantile(g, 0.975)
print("\n強−弱ギャップの巡回ブロック・ブートストラップ(月ブロック, 2000回):")
for mo in (1, 3, 6, 12):
    med, lo, hi = block_boot_gap(mo)
    print(f"  {mo:>2}mo: 中央値 {med:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]  {'0超' if lo>0 else '0またぎ'}")
