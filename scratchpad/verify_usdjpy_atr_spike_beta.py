"""【第7段・ベータ検定】USDJPY h1 ATR拡大足ロング（凍結仕様）が、政策乖離期3年（2013/2001/2022）
の集中で持ち上がっているだけの「時代のベータ」か、それを除いても残る「本物」かを判定する。

凍結仕様: k=2.0・ロング・成行(次足始値)・損切り=拡大足の安値・ATR×3トレール・保有上限20本・
前日高値フィルタ（(close[s]-前日高値)/atr_prev>0）。往復コスト0.9pip(=0.009円)。損益=入口価格%。
実行は src.engine.walk.walk() のみ（自前ウォーカー禁止）。

手順:
  1. 通算+30.2%のうち上位3年（2013/2001/2022）を機械的に検出し、除いた残り24年の成績。
  2. 日足SMA200上/下・週足close vs 30週SMA・円ボラ体制（日足ATR14が自分の過去3年ローリング中央値超）
     の3軸で層別（すべて確定後・shiftで先読み回避）。ON/OFF成績を出す。
  3. 素の全陽線帰無（時間帯一致・ランダム間引き300回）を、(a)全期間 (b)上位3年を除いた集合 の両方で。
  4. 時代分割 2000-2012 / 2013-2026。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv          # noqa: E402
from src.engine.walk import walk                  # noqa: E402

COST = 0.009  # 往復0.9pip


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def trigger(d, k, use_pdh):
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    hit = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    s = np.flatnonzero(hit)
    s = s[s + 1 < len(d)]
    if use_pdh:
        s = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    return s


def go(d, s_list, cost_abs):
    o, l = d["open"].to_numpy(), d["low"].to_numpy()
    ent = [(s, o[s + 1], l[s], o[s + 1] + 1000.0 * (o[s + 1] - l[s]), s)
           for s in s_list if o[s + 1] - l[s] > 0]
    if len(ent) < 10:
        return None, None
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=20, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=3.0, trail_n=14)
    t, _ = walk(d, ent, None, a)
    if t is None or len(t) < 10:
        return None, None
    p = ((t["R"] * t["risk"] - cost_abs) / t["e_px"]).to_numpy()
    return t, p


def rep(p):
    if p is None or len(p) == 0:
        return dict(N=0, win=np.nan, pf=np.nan, mean=np.nan, tot=np.nan, dd=np.nan)
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max()) * 100
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    return dict(N=len(p), win=np.mean(p > 0) * 100, pf=w / ls if ls > 0 else np.nan,
                mean=p.mean() * 100, tot=p.sum() * 100, dd=dd)


def null_test(d, s_trig, p_obs, cost_abs, reps=300, seed=23):
    """時間帯一致のランダム間引き帰無（素の全陽線プールから）。"""
    pool = trigger(d, 0.0, False)
    hrs_pool = d.index.hour.to_numpy()[pool]
    hrs_trig = d.index.hour.to_numpy()[s_trig]
    cnt = pd.Series(hrs_trig).value_counts()
    rng = np.random.default_rng(seed)
    npf, nm = [], []
    for _ in range(reps):
        pick = []
        for hh, n in cnt.items():
            cand = pool[hrs_pool == hh]
            if len(cand):
                pick.extend(rng.choice(cand, size=min(int(n), len(cand)), replace=False))
        _, pn = go(d, np.sort(np.array(pick)), cost_abs)
        if pn is None:
            continue
        q = rep(pn)
        npf.append(q["pf"]); nm.append(q["mean"])
    npf, nm = np.array(npf), np.array(nm)
    o = rep(p_obs)
    return dict(pf_med=np.median(npf), pf_std=npf.std(ddof=1),
                m_med=np.median(nm), m_std=nm.std(ddof=1),
                pf_pct=(npf < o["pf"]).mean() * 100, m_pct=(nm < o["mean"]).mean() * 100)


# ============================================================ データ・アンカー再現
d = load_mt5_csv("data/vantage_usdjpy_h1.csv").loc["2000-01-01":]
span = (d.index[-1] - d.index[0]).days / 365.25
s_all = trigger(d, 2.0, True)
t_all, p_all = go(d, s_all, COST)
r_all = rep(p_all)
print("=" * 78)
print("STEP0: アンカー再現")
print(f"  N={r_all['N']} N/年={r_all['N']/span:.1f} 勝率={r_all['win']:.1f}% PF={r_all['pf']:.2f} "
      f"平均={r_all['mean']:+.4f}% 総={r_all['tot']:+.1f}% maxDD={r_all['dd']:.1f}%")
assert r_all["N"] == 636, r_all["N"]
assert abs(r_all["pf"] - 1.30) < 0.005, r_all["pf"]
assert abs(r_all["mean"] - 0.0474) < 0.0005, r_all["mean"]
assert abs(r_all["dd"] - 9.1) < 0.05, r_all["dd"]
print("  OK: N=636 / PF1.30 / 平均+0.0474% / maxDD9.1% を再現")

# ============================================================ STEP1: 上位3年を機械的に検出
yr_pnl = pd.Series(p_all).groupby(t_all["time"].dt.year.values).sum().sort_values(ascending=False)
top3 = list(yr_pnl.index[:3])
print("\n" + "=" * 78)
print("STEP1: 上位3年の機械検出と除外")
print("  年別損益トップ5: " + " ".join(f"{y}:{v*100:+.1f}%" for y, v in yr_pnl.head(5).items()))
print(f"  上位3年 = {sorted(top3)}")
assert sorted(top3) == [2001, 2013, 2022], top3
top3_sum = yr_pnl.iloc[:3].sum() * 100
print(f"  上位3年の合計 = {top3_sum:+.1f}%  (通算 {r_all['tot']:+.1f}% の {top3_sum/r_all['tot']*100:.1f}%)")

mask_ex = ~np.isin(t_all["time"].dt.year.values, top3)
p_ex = p_all[mask_ex]
r_ex = rep(p_ex)
n_years_ex = 27 - 3
pos_years_ex = int((pd.Series(p_ex).groupby(t_all["time"].dt.year.values[mask_ex]).sum() > 0).sum())
print(f"\n  【残り24年（上位3年除外）】")
print(f"  N={r_ex['N']} N/年={r_ex['N']/(span*24/27):.1f} 勝率={r_ex['win']:.1f}% PF={r_ex['pf']:.2f} "
      f"平均={r_ex['mean']:+.4f}% 総={r_ex['tot']:+.1f}% maxDD={r_ex['dd']:.1f}% "
      f"黒字年={pos_years_ex}/{n_years_ex}")
# 整合性チェック: 除外集合の総%が 全体総% - 上位3年合計 と一致すること
resid_check = r_all["tot"] - top3_sum
print(f"  検算: 全体総%({r_all['tot']:+.2f}) − 上位3年合計({top3_sum:+.2f}) = {resid_check:+.2f}% "
      f"vs 除外集合の総% {r_ex['tot']:+.2f}%")
assert abs(resid_check - r_ex["tot"]) < 0.05, (resid_check, r_ex["tot"])
print("  OK: 上位3年除外の集計が全体の残差と整合")

# ============================================================ STEP2: 帰無（全期間 vs 上位3年除外）
print("\n" + "=" * 78)
print("STEP2: 帰無比較（同じ時間帯構成・ランダム間引き300回）")
n_full = null_test(d, s_all, p_all, COST)
print(f"  【全期間】 帰無 PF={n_full['pf_med']:.2f}±{n_full['pf_std']:.2f} "
      f"平均={n_full['m_med']:+.4f}%±{n_full['m_std']:.4f}  "
      f"→ 実測 PF%ile={n_full['pf_pct']:.1f} 平均%ile={n_full['m_pct']:.1f}")

yrs_sig = d.index[s_all].year.to_numpy()
s_ex = s_all[~np.isin(yrs_sig, top3)]  # 信号バーの年で上位3年を除外(トレード配列でなく信号配列側)
n_ex = null_test(d, s_ex, p_ex, COST)
print(f"  【上位3年除外】 帰無 PF={n_ex['pf_med']:.2f}±{n_ex['pf_std']:.2f} "
      f"平均={n_ex['m_med']:+.4f}%±{n_ex['m_std']:.4f}  "
      f"→ 実測 PF%ile={n_ex['pf_pct']:.1f} 平均%ile={n_ex['m_pct']:.1f}")

# ============================================================ STEP3: レジーム層別（先読み点検つき）
print("\n" + "=" * 78)
print("STEP3: レジーム層別（すべて確定後の値・shiftで先読み回避）")

d1 = load_mt5_csv("data/vantage_usdjpy_d1.csv")
w1 = load_mt5_csv("data/vantage_usdjpy_w1.csv")

# 日足SMA200: 直近確定日足のSMA200（当日分は未確定なのでshift(1)）
d1_sma200 = d1["close"].rolling(200).mean().shift(1)
d1_regime = pd.Series(np.where(d1["close"].shift(1) > d1_sma200, 1, -1), index=d1.index)
d1_regime = d1_regime.reindex(d.index.floor("D")).ffill()
d1_regime.index = d.index
daily_up = (d1_regime.to_numpy() > 0)

# 週足: 終値 vs 30週SMA（確定週のみ、当週はshift(1)で除外）
w1_sma30 = w1["close"].rolling(30).mean().shift(1)
w1_regime = pd.Series(np.where(w1["close"].shift(1) > w1_sma30, 1, -1), index=w1.index)
# 週足の値は「その週の月曜0時時点でまだ確定していない」ので、reindexしてffillした上でさらに
# 1週分ラグを取り、当該h1バーの週の"前週末確定値"を使う
w1_regime_shift = w1_regime.shift(1)
w1_regime_h1 = w1_regime_shift.reindex(d.index, method="ffill")
weekly_up = (w1_regime_h1.to_numpy() > 0)

# 円のボラ体制: 日足ATR14が「その時点までの過去3年(756営業日)ローリング中央値」を超えるか
# (拡大窓の中央値=完全に過去のみ・先読み無し)
d1_atr = wilder_atr(d1, 14)
d1_atr_med = d1_atr.rolling(756, min_periods=200).median().shift(1)
vol_regime = pd.Series(np.where(d1_atr.shift(1) > d1_atr_med, 1, -1), index=d1.index)
vol_regime_h1 = vol_regime.reindex(d.index.floor("D")).ffill()
vol_regime_h1.index = d.index
high_vol = (vol_regime_h1.to_numpy() > 0)

# 先読み点検: 末尾を切り落としても過去の値が不変か
d1_trunc = d1.iloc[:-500]
d1_sma200_trunc = d1_trunc["close"].rolling(200).mean().shift(1)
common = d1_sma200.index.intersection(d1_sma200_trunc.index)
common = common[common < d1_trunc.index[-1] - pd.Timedelta(days=5)]  # 端の窓効果を避ける
a_, b_ = d1_sma200.loc[common].to_numpy(), d1_sma200_trunc.loc[common].to_numpy()
match = int(np.isclose(a_, b_, atol=1e-9, equal_nan=True).sum())
print(f"  先読み点検(日足SMA200・末尾500日切断): {match}/{len(common)} 一致")
assert match == len(common), "日足SMA200に先読みあり"

for label, flag in (("日足SMA200 上(1)/下(-1)", daily_up),
                     ("週足close vs 30週MA 上(1)/下(-1)", weekly_up),
                     ("円ボラ体制 高(1)/低(-1)", high_vol)):
    print(f"\n  --- {label}")
    for on in (True, False):
        m = flag[s_all] == on if flag.dtype == bool else None
        sel = s_all[flag[s_all] == on]
        _, pz = go(d, sel, COST)
        rz = rep(pz)
        tag = "ON " if on else "OFF"
        if rz["N"]:
            print(f"    {tag}: N={rz['N']:4d} N/年={rz['N']/span:5.1f} 勝率={rz['win']:5.1f}% "
                  f"PF={rz['pf']:5.2f} 平均={rz['mean']:+.4f}% maxDD={rz['dd']:5.1f}%")
        else:
            print(f"    {tag}: サンプル不足")

# ============================================================ STEP4: 時代分割
print("\n" + "=" * 78)
print("STEP4: 時代分割 2000-2012 / 2013-2026")
for a0, a1, lab in (("2000-01-01", "2012-12-31", "2000-2012"), ("2013-01-01", None, "2013-2026")):
    dd_ = d.loc[a0:a1] if a1 else d.loc[a0:]
    ss = trigger(dd_, 2.0, True)
    _, pz = go(dd_, ss, COST)
    rz = rep(pz)
    sp_ = (dd_.index[-1] - dd_.index[0]).days / 365.25
    nz = null_test(dd_, ss, pz, COST) if rz["N"] else None
    print(f"  {lab}: N={rz['N']} N/年={rz['N']/sp_:.1f} 勝率={rz['win']:.1f}% PF={rz['pf']:.2f} "
          f"平均={rz['mean']:+.4f}% maxDD={rz['dd']:.1f}%"
          + (f"  帰無%ile(PF,平均)={nz['pf_pct']:.1f},{nz['m_pct']:.1f}" if nz else ""))

print("\n" + "=" * 78)
print("完了")
