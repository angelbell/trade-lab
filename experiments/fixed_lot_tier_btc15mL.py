"""btc15m_L の「強度で固定ロットを盛る」検定（凍結仕様カード、2026-07-24）。

背景: ユーザーは小口座・固定ロット手張り（最小0.01、上限0.03）。E1階段の"下削り"（サイズを1未満に
削る側）は床が0.01のため使えない（見送りは法則9bで有害）。ここでは"上に盛る"側だけを固定ロットの
土俵で検定する。

母集団（凍結・変更禁止）: stack_size_btc15mL.build_population()（btc15m_L canonical、
生run tie-back n=763・meanR≈+0.567）をそのまま使う。強度軸は同ファイルの comp1_ladder() が返す
above_pdh / above_hh4 から作る3層（both=両方上 / one=片方 / neither=どちらも下）。

検定1（強度勾配・本命）: 素R（サイズ前・コスト前）で3層の n/win%/meanR/PF/totR を比較し、
  both>one>neither の単調性を、(a) 同サイズ無作為3分割null(2000回)の%ile、(b) 巡回ブロック
  ブートストラップ(1/3/6/12mo)での P(both群meanR > 全体meanR) の block長依存、の2軸で検定する。

検定2（固定ロット$損益）: flat(全部0.01) / 2段(both=0.02,他0.01) / 3段(both=0.03,one=0.02,neither=0.01)
  を同じ非複利加算$曲線（口座1万ドル）で比較する。
  1トレード$損益 = lot × (R × risk) − lot×(15 + swap年30%×e_px×hold/365)。
  これは stack_size_btc15mL.apply_size(t, W) が返す式 W*(R - 15/risk - swap*e_px/risk*hold) に
  risk を掛け戻したものと代数的に同一（W=lot と読み替えるだけ。新規コスト式ではなく既存関数の
  再利用＋単位変換）。したがって実装は apply_size をそのまま import して呼び、その戻り値に
  t["risk"] を掛けるだけ。

流用（車輪の再発明禁止）:
  - stack_size_btc15mL.{build_population, apply_size, comp1_ladder}
  - src.engine.arbiter.{Boot, cd}（block_boot_beat と同じ考え方で both-vs-全体 の勾配版を書く。
    Boot自体は無改変で使う）
  - experiments/verify_e1_tv_window.py の TV窓・curve_stats をそのまま呼んで tie-back 継続を確認
    してから、その同じ窓で新しい固定ロット3方式を評価する（窓の定義=2026-04-01〜2026-07-24 を
    共有するだけで、サイジング方式そのものは別物＝リスク基準%ではなく固定ロット。TV実測値との
    厳密一致は期待しない、方向・オーダーの整合を見る）。

Run: .venv/bin/python experiments/fixed_lot_tier_btc15mL.py [--smoke] 2>&1 | tee experiments/out_fixed_lot_tier_btc15mL.txt
"""
import sys, warnings, argparse
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import numpy as np
import pandas as pd

from stack_size_btc15mL import build_population, apply_size, comp1_ladder, BTC_PCT_YR
from src.engine.arbiter import Boot, cd
from verify_e1_tv_window import curve_stats as tv_curve_stats, W_START, W_END
from stack_size_btc15mL import comp2_daily

NULL_ITERS = 2000
NULL_SEED = 20260724
BOOT_NB = 300


# ============================================================== 検定1: 素R層別統計
def tier_stats(R, mask):
    x = np.asarray(R)[mask]
    n = len(x)
    if n == 0:
        return dict(n=0, win=np.nan, meanR=np.nan, pf=np.nan, totR=np.nan)
    win = 100.0 * np.mean(x > 0)
    meanR = x.mean()
    pos, neg = x[x > 0].sum(), -x[x <= 0].sum()
    pf = pos / neg if neg > 0 else np.inf
    totR = x.sum()
    return dict(n=n, win=win, meanR=meanR, pf=pf, totR=totR)


def print_tier(label, s):
    print(f"    {label:<10} n={s['n']:4d}  win%={s['win']:5.1f}  meanR={s['meanR']:+.3f}  "
          f"PF={s['pf']:5.2f}  totR={s['totR']:+7.1f}")


def null_gradient_test(raw, n_both, n_one, n_neither, real_grad, iters=NULL_ITERS, seed=NULL_SEED):
    """トレードを無作為に(n_both,n_one,n_neither)の3群へ割った時の勾配(meanR_both-meanR_neither)分布。
    グループサイズは実測の3層サイズに固定（置換検定）。"""
    N = len(raw)
    rng = np.random.default_rng(seed)
    grads = np.empty(iters)
    mono = np.zeros(iters, dtype=bool)
    for i in range(iters):
        perm = rng.permutation(N)
        b = raw[perm[:n_both]]
        o = raw[perm[n_both:n_both + n_one]]
        ne = raw[perm[n_both + n_one:n_both + n_one + n_neither]]
        mb, mo, mn = b.mean(), o.mean(), ne.mean()
        grads[i] = mb - mn
        mono[i] = (mb > mo) and (mo > mn)
    pct = 100.0 * np.mean(grads < real_grad)
    return grads, pct, 100.0 * mono.mean()


def block_boot_grad(overall_s, both_s, months, k, nb=BOOT_NB, seed=20260724):
    """block_boot_beat(arb_common/arbiter)と同じ骨格: 同じBoot層で overall と both を
    それぞれ月ブロック連結し、その経路上で mean(both) > mean(overall) かを数える。"""
    boot = Boot(months, nb=nb, k=k, seed=seed)
    mko = overall_s.index.to_period("M")
    mkb = both_s.index.to_period("M")
    byo = {m: overall_s.values[mko == m] for m in months}
    byb = {m: both_s.values[mkb == m] for m in months}
    no = len(overall_s)
    wins, valid = 0, 0
    for seq in boot.layout:
        seqm = [months[j] for j in seq]
        vo = np.concatenate([byo[m] for m in seqm])[:no]
        vb = np.concatenate([byb[m] for m in seqm])
        if len(vb) == 0 or len(vo) == 0:
            continue
        wins += (vb.mean() > vo.mean())
        valid += 1
    return 100.0 * wins / valid if valid else np.nan


# ============================================================== 検定2: 固定ロット$損益
def apply_size_dollars(t, W):
    """固定ロット$損益 = apply_size(t,W)（既存関数、無改変）に t['risk'] を掛け戻したもの。
    代数的に W*(R*risk - 15 - swap*e_px*hold) と同一（=lot×(R×risk)-lot×(15+swap)）。"""
    return apply_size(t, W) * t["risk"].values


def dollar_curve_stats(pnl, ti, start=10000.0):
    order = np.argsort(np.asarray(ti))
    pnl_s = np.asarray(pnl)[order]
    ti_s = np.asarray(ti)[order]
    bal = np.concatenate([[start], start + np.cumsum(pnl_s)])
    peak = np.maximum.accumulate(bal)
    dd = float(np.max((peak - bal) / peak)) * 100.0
    total_pct = (bal[-1] - start) / start * 100.0
    retdd = total_pct / dd if dd > 0 else np.inf
    by_year = pd.Series(pnl_s, index=pd.DatetimeIndex(ti_s)).groupby(pd.DatetimeIndex(ti_s).year).sum()
    return total_pct, dd, retdd, by_year


def report_scheme(label, pnl, ti):
    total, dd, retdd, by_year = dollar_curve_stats(pnl, ti)
    print(f"    {label:<8} total={total:+7.2f}%  maxDD={dd:6.2f}%  return/DD={retdd:6.2f}")
    print("      年別$損益: " + "  ".join(f"{y}:{v:+.0f}" for y, v in by_year.items()))
    return total, dd, retdd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print("#" * 110)
    print("母集団 tie-back（凍結・変更禁止）")
    print("#" * 110)
    d15, t, ii = build_population()
    assert len(t) == 763, f"母体tie-back失敗: n={len(t)} (既知763)"
    assert abs(t["R"].mean() - 0.567) < 0.01, f"母体tie-back失敗: meanR={t['R'].mean():.3f} (既知≈0.567)"
    print(f"  検算OK: n={len(t)}  meanR={t['R'].mean():+.4f}  (既知: n=763, meanR≈+0.567)")

    if args.smoke:
        m = pd.to_datetime(t["time"].values) >= pd.Timestamp("2024-01-01")
        t = t[m].reset_index(drop=True)
        ii = ii[m]
        print(f"  --smoke: 2024-01-01以降に限定 -> n={len(t)}")

    W1, above_pdh, above_hh4 = comp1_ladder(d15, t, ii)
    both = above_pdh & above_hh4
    one = above_pdh ^ above_hh4
    neither = (~above_pdh) & (~above_hh4)
    n_both, n_one, n_neither = int(both.sum()), int(one.sum()), int(neither.sum())
    N = len(t)
    assert n_both + n_one + n_neither == N
    print(f"  3層内訳: both={n_both}({100*n_both/N:.1f}%)  one={n_one}({100*n_one/N:.1f}%)  "
          f"neither={n_neither}({100*n_neither/N:.1f}%)")

    raw = t["R"].values
    ti_all = pd.DatetimeIndex(t["time"])

    # ================================================================ 検定1
    print("\n" + "#" * 110)
    print("検定1: 強度勾配（素R、サイズ前・コスト前）")
    print("#" * 110)
    s_both = tier_stats(raw, both)
    s_one = tier_stats(raw, one)
    s_neither = tier_stats(raw, neither)
    print_tier("both", s_both)
    print_tier("one", s_one)
    print_tier("neither", s_neither)

    mono_meanR = (s_both["meanR"] > s_one["meanR"]) and (s_one["meanR"] > s_neither["meanR"])
    mono_pf = (s_both["pf"] > s_one["pf"]) and (s_one["pf"] > s_neither["pf"])
    mono_totR_density = (s_both["totR"] / s_both["n"] > s_one["totR"] / s_one["n"]
                          > s_neither["totR"] / s_neither["n"])
    print(f"\n  単調性(meanR): both>one>neither = {mono_meanR}")
    print(f"  単調性(PF)   : both>one>neither = {mono_pf}")
    print(f"  単調性(totR/n＝meanRと同値)     = {mono_totR_density}")

    # 年別内訳（both層の年ごとの本数・散らばり確認）
    by_year_n = pd.Series(1, index=ti_all).groupby([ti_all.year,
                          np.select([both, one], ["both", "one"], "neither")]).sum().unstack(fill_value=0)
    print("\n  年別3層本数:")
    print(by_year_n.to_string())

    real_grad = s_both["meanR"] - s_neither["meanR"]
    grads, pct, mono_pct_null = null_gradient_test(raw, n_both, n_one, n_neither, real_grad)
    print(f"\n  null(無作為3分割, {NULL_ITERS}回, 同サイズ): 勾配(meanR_both-meanR_neither)"
          f" 実測={real_grad:+.3f}  null帯=[{np.percentile(grads,2.5):+.3f},{np.percentile(grads,97.5):+.3f}]"
          f"(中央値={np.median(grads):+.3f}, 標準偏差={grads.std():.3f})  -> {pct:.1f}%ile")
    print(f"  参考: null内でboth>one>neitherが偶然そろう率 = {mono_pct_null:.1f}%")

    months = sorted(ti_all.to_period("M").unique())
    overall_s = pd.Series(raw, index=ti_all).sort_index()
    both_s = pd.Series(raw[both], index=ti_all[both]).sort_index()
    print("\n  巡回ブロック・ブートストラップ 1/3/6/12mo: P(both群のmeanR > 全体meanR)")
    boot_ps = {}
    for k in (1, 3, 6, 12):
        p = block_boot_grad(overall_s, both_s, months, k)
        boot_ps[k] = p
        print(f"    k={k:>2}mo: P={p:.1f}%")
    boot_monotone = boot_ps[1] <= boot_ps[3] <= boot_ps[6] <= boot_ps[12]
    print(f"  block長で単調に上昇(1<=3<=6<=12) = {boot_monotone}")

    test1_pass = mono_meanR and (pct >= 90.0) and boot_monotone
    print(f"\n  検定1判定: 単調={mono_meanR}  帰無%ile>=90={pct>=90.0}({pct:.1f}%)  "
          f"ブロック単調上昇={boot_monotone}  => "
          f"{'3段に根拠あり（both>one>neitherは本物の勾配）' if test1_pass else '3段の根拠不十分（both≈one疑い、事前登録の死に方1に該当）'}")

    # ================================================================ 検定2 前半: TV窓 tie-back継続確認
    print("\n" + "#" * 110)
    print(f"検定2 前半: TV窓 tie-back継続確認（{W_START}〜{W_END}）— サイジング方式はTVと別物、"
          f"方向・オーダーの整合のみ確認")
    print("#" * 110)
    tt = pd.to_datetime(t["time"].values)
    m_tv = (tt >= pd.Timestamp(W_START)) & (tt <= pd.Timestamp(W_END))
    print(f"  窓内トレード数: {int(m_tv.sum())}  (TV実測35、既存tie-back実測33。フィード更新差は許容)")
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    w_base = np.where(t["e_px"].values > pdh[ii], 1.0, 0.5)
    W2, _ = comp2_daily(d15, t, ii)
    w_e1 = W1 * W2
    R_base_riskpct = apply_size(t, w_base)
    R_e1_riskpct = apply_size(t, w_e1)
    ti_series = pd.Series(tt)
    for name, R in [("baseline(TV同型,PDHソフト0.5,1%risk曲線)", R_base_riskpct),
                    ("E1(TV同型,階段x日足,1%risk曲線)", R_e1_riskpct)]:
        n, win, winp, pf, total, dd = tv_curve_stats(R[m_tv], ti_series[m_tv])
        print(f"    {name}: n={n} win={win}({winp:.2f}%) PF={pf:.3f} total={total:+.2f}% maxDD={dd:.2f}%")
    print(f"  (参考: TV実測 baseline total=-8.16% maxDD=19.70% / E1 total=-3.83% maxDD=13.73%)")

    # ================================================================ 検定2 本体: 固定ロット$損益
    print("\n" + "#" * 110)
    print("検定2 本体: 固定ロット$損益（口座1万ドル・非複利加算曲線）")
    print("#" * 110)
    W_flat = np.full(N, 0.01)
    W_2 = np.where(both, 0.02, 0.01)               # 切れ目 both|one（当初版）
    W_2b = np.where(both | one, 0.02, 0.01)         # 切れ目 neither|one（濃縮が倍になる境界）
    W_3 = np.where(both, 0.03, np.where(one, 0.02, 0.01))

    pnl_flat = apply_size_dollars(t, W_flat)
    pnl_2 = apply_size_dollars(t, W_2)
    pnl_2b = apply_size_dollars(t, W_2b)
    pnl_3 = apply_size_dollars(t, W_3)

    print(f"\n  -- 全履歴 ({ti_all.min().date()} 〜 {ti_all.max().date()}, n={N}) --")
    res_full = {}
    for label, pnl in [("flat", pnl_flat), ("2段A(both切)", pnl_2),
                       ("2段B(neither切)", pnl_2b), ("3段", pnl_3)]:
        res_full[label] = report_scheme(label, pnl, tt)

    print(f"\n  -- TV窓 ({W_START}〜{W_END}, n={int(m_tv.sum())}) --")
    res_tv = {}
    for label, pnl in [("flat", pnl_flat[m_tv]), ("2段A(both切)", pnl_2[m_tv]),
                       ("2段B(neither切)", pnl_2b[m_tv]), ("3段", pnl_3[m_tv])]:
        res_tv[label] = report_scheme(label, pnl, tt[m_tv])

    print("\n  レバレッジ点検(法則7.5/8): 3段が2段よりreturnもDDも増えるだけならレバレッジ、"
          "return/DDの改善が要件")
    tot_flat, dd_flat, rd_flat = res_full["flat"]
    tot_2a, dd_2a, rd_2a = res_full["2段A(both切)"]
    tot_2b, dd_2b, rd_2b = res_full["2段B(neither切)"]
    tot_3, dd_3, rd_3 = res_full["3段"]
    print(f"    全履歴 return/DD: flat={rd_flat:.2f}  2段A(both切)={rd_2a:.2f}  "
          f"2段B(neither切)={rd_2b:.2f}  3段={rd_3:.2f}")
    print(f"    全履歴 total%   : flat={tot_flat:+.2f}  2段A={tot_2a:+.2f}  2段B={tot_2b:+.2f}  3段={tot_3:+.2f}")

    best2 = "2段B(neither切)" if rd_2b >= rd_2a else "2段A(both切)"
    rd_best2 = max(rd_2a, rd_2b)
    print(f"\n  検定2判定(全履歴, return/DD基準):")
    print(f"    良い方の2段 = {best2} (return/DD={rd_best2:.2f})")
    print(f"    3段 が 良い方の2段 を上回るか = {rd_3 > rd_best2}  ({rd_3:.2f} vs {rd_best2:.2f})")
    print(f"    flat が 良い方の2段 を上回るか = {rd_flat > rd_best2}  ({rd_flat:.2f} vs {rd_best2:.2f})")
    if rd_flat > rd_best2:
        verdict2 = "flatで確定（傾斜そのものに価値なし）"
    elif rd_3 > rd_best2:
        verdict2 = f"3段が最良（{best2}も上回る）"
    else:
        verdict2 = f"{best2}が最良（3段の0.03は濃縮不足でreturn/DD改善せず）"
    print(f"    => {verdict2}")

    print("\n" + "#" * 110)
    print("総合")
    print("#" * 110)
    print(f"  検定1(強度勾配): {'合格' if test1_pass else '不合格'}")
    print(f"  検定2(固定ロットreturn/DD, 全履歴): {verdict2}")
    print(f"  検定2(TV窓, 参考): flat={res_tv['flat'][2]:.2f}  2段A={res_tv['2段A(both切)'][2]:.2f}  "
          f"2段B={res_tv['2段B(neither切)'][2]:.2f}  3段={res_tv['3段'][2]:.2f}  (n少なく参考値)")


if __name__ == "__main__":
    main()
