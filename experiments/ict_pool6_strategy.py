"""改善の本命: 同一機構(C3・E3×S2・目標PDH-5pip)を6ペアに同時に張った「プール」を
1つの単独戦略として評価する。ユーザーはブック作業を棚上げ中＝審判は単独運用の物差し
（PF・N/年・トレード解像度maxDD・年別）。

凍結スペック（全ペア共通・チューニング禁止）:
  母集団=C3(狩り+MSS+FVG ma=0.15,ロングのみ,NYキルゾーン)、入口=E3(浅指値0.25固定リトレース)、
  損切り=S2(狩安値-0.5ATR)、目標=各ペア自身のPDH-5pip、コスト=realistic(ペア別スプレッド)。

流用（車輪の再発明禁止）: ict_lift_decay_test.build_trades（6ペア分のC3×E3×S2トレードを
そのまま生成 --- 母集団構築ロジックの複製はしない）、ict_capture_decomp.{cell_stats, filter_era}、
arb_common.{Boot, cd, months_union}。新規実装はプール合算・ドル因子チェックの集計のみ。

tie-back: EURUSD単体のC3×E3×S2全史が ict_condition_ablation.py の値
(n=340/win42.6/PF1.23/totR+45.6) と一致することを最初に確認。

Run: .venv/bin/python experiments/ict_pool6_strategy.py [--smoke] 2>&1 | tee experiments/out_ict_pool6_strategy.txt
"""
import sys, io, argparse, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from src.data_loader import load_mt5_csv
from ict_capture_decomp import cell_stats, filter_era
from ict_lift_decay_test import build_trades, FX6, ERAS
from arb_common import Boot, cd, months_union

RISK_PCT = 0.01


def trade_series(trades):
    """(date,net,gross,risk)のリスト -> pd.Series(net, index=date) 昇順ソート済み。"""
    if not trades:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex([t[0] for t in trades])
    return pd.Series([t[1] for t in trades], index=idx).sort_index()


def per_pair_report(name, trades):
    print(f"\n  --- {name} ---")
    st_all = cell_stats(trades)
    s = trade_series(trades)
    days = max((s.index[-1] - s.index[0]).days, 1) if len(s) else 1
    dd_all = cd(s.values * RISK_PCT, days)[1] if len(s) else np.nan
    yrs = days / 365.25
    if st_all:
        print(f"    全史    : n={st_all['n']:4d} 本/年={st_all['n']/yrs:5.1f} win%={st_all['win']:5.1f} "
              f"PF={st_all['pf']:5.2f} meanR={st_all['meanR']:+.3f} totR={st_all['totR']:+7.1f} "
              f"maxDD(R)={st_all['maxDD']:6.2f}")
    for lo, hi, elabel in ERAS:
        sub = filter_era(trades, lo, hi)
        cs = cell_stats(sub)
        if cs:
            print(f"    {elabel:8s}: n={cs['n']:4d} 本/年={cs['n']/3.0:5.1f} win%={cs['win']:5.1f} "
                  f"PF={cs['pf']:5.2f} meanR={cs['meanR']:+.3f} totR={cs['totR']:+7.1f} "
                  f"maxDD(R)={cs['maxDD']:6.2f}")
        else:
            print(f"    {elabel:8s}: n不足")
    return st_all, dd_all


def boot_dd_distribution(s, months, k, nb=3000, seed=20260717):
    if len(months) < k + 1:
        return None
    boot = Boot(months, nb=nb, k=k, seed=seed)
    mk = s.index.to_period("M")
    by = {m: s.values[mk == m] for m in months}
    days = max((s.index[-1] - s.index[0]).days, 1)
    n = len(s)
    dd_out = np.full(len(boot.layout), np.nan)
    cagr_out = np.full(len(boot.layout), np.nan)
    for i, seq in enumerate(boot.layout):
        v = np.concatenate([by[months[j]] for j in seq])[:n]
        if len(v) == 0:
            continue
        cagr, dd = cd(v, days)
        dd_out[i] = dd
        cagr_out[i] = cagr
    return dd_out, cagr_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print("#" * 110)
    print("0. tie-back: EURUSD単体 C3×E3×S2 全史 が ict_condition_ablation.py の n=340/win42.6/PF1.23/totR+45.6 を再現するか")
    print("#" * 110)
    trades_by_pair = {}
    for name in FX6:
        out, *_ = build_trades(name, args.smoke)
        trades_by_pair[name] = out["S2"]["C3"]
    st = cell_stats(trades_by_pair["eurusd"])
    print(f"  再現値: n={st['n']} win%={st['win']:.1f} PF={st['pf']:.2f} totR={st['totR']:+.1f}")
    if not args.smoke:
        ok = (st['n'] == 340 and abs(st['win'] - 42.6) < 0.2 and abs(st['pf'] - 1.23) < 0.02)
        print(f"  {'PASS' if ok else 'FAIL --- 要確認'}")

    print("\n" + "#" * 110)
    print("1. ペア別（全史・年代別）")
    print("#" * 110)
    dd_by_pair = {}
    for name in FX6:
        _, dd = per_pair_report(name, trades_by_pair[name])
        dd_by_pair[name] = dd

    # ---------------- 2. プール合算 ----------------
    print("\n" + "#" * 110)
    print("2. プール合算（6ペア同時、1トレード1%リスク）")
    print("#" * 110)
    all_trades = []
    for name in FX6:
        all_trades.extend(trades_by_pair[name])
    all_trades.sort(key=lambda t: t[0])
    pool_s = trade_series(all_trades)
    days = max((pool_s.index[-1] - pool_s.index[0]).days, 1)
    yrs = days / 365.25
    n_total = len(pool_s)
    print(f"\n  n_total={n_total}  本/年={n_total/yrs:.1f}  期間={yrs:.1f}年")

    pool_cagr, pool_dd = cd(pool_s.values * RISK_PCT, days)
    pool_totR = pool_s.values.sum()
    pool_win = 100 * np.mean(pool_s.values > 0)
    pos, neg = pool_s.values[pool_s.values > 0].sum(), -pool_s.values[pool_s.values < 0].sum()
    pool_pf = pos / neg if neg > 0 else np.inf
    print(f"  全史: n={n_total} win%={pool_win:.1f} PF={pool_pf:.2f} totR={pool_totR:+.1f} "
          f"maxDD(R,実測1経路)={cd(pool_s.values, days)[1]:.2f}  "
          f"CAGR(1%risk)={pool_cagr:+.1f}%  maxDD(1%risk,実測1経路)={pool_dd:.2f}%  "
          f"CAGR/DD={pool_cagr/pool_dd if pool_dd>0 else float('inf'):.2f}")

    for lo, hi, elabel in ERAS:
        sub = [t for t in all_trades if lo <= pd.Timestamp(t[0]).year <= hi]
        cs = cell_stats(sub)
        if cs:
            print(f"  {elabel:8s}: n={cs['n']:4d} 本/年={cs['n']/3.0:5.1f} win%={cs['win']:5.1f} "
                  f"PF={cs['pf']:5.2f} meanR={cs['meanR']:+.3f} totR={cs['totR']:+7.1f} "
                  f"maxDD(R)={cs['maxDD']:6.2f}")
        else:
            print(f"  {elabel:8s}: n不足")

    print("\n  年別R(プール合算、非複利):")
    by_year = pool_s.groupby(pool_s.index.year).agg(["sum", "count"])
    for y, row in by_year.iterrows():
        print(f"    {y}: totR={row['sum']:+7.1f}  n={int(row['count']):3d}")

    print("\n  トレード解像度 maxDD の巡回ブロック・ブートストラップ(1/3/6/12mo, 3000回, 1%risk):")
    months_all = sorted(pool_s.index.to_period("M").unique())
    for k in (1, 3, 6, 12):
        res = boot_dd_distribution(pool_s * RISK_PCT, months_all, k)
        if res is None:
            print(f"    k={k:>2}mo: 月数不足でskip"); continue
        dd_out, cagr_out = res
        ok = ~np.isnan(dd_out)
        dd_med = np.nanmedian(dd_out); dd_95 = np.nanpercentile(dd_out[ok], 95)
        rdd = cagr_out[ok] / np.where(dd_out[ok] > 0, dd_out[ok], np.nan)
        print(f"    k={k:>2}mo: maxDD中央値={dd_med:.2f}%  maxDD 95%点={dd_95:.2f}%  "
              f"CAGR中央値={np.nanmedian(cagr_out[ok]):+.1f}%  CAGR/DD中央値={np.nanmedian(rdd):.2f}")

    # ---------------- プール vs EURUSD単独 ----------------
    print("\n" + "#" * 110)
    print("プール vs EURUSD単独（同じ物差し、1%リスク）")
    print("#" * 110)
    eur_s = trade_series(trades_by_pair["eurusd"])
    eur_days = max((eur_s.index[-1] - eur_s.index[0]).days, 1)
    eur_cagr, eur_dd = cd(eur_s.values * RISK_PCT, eur_days)
    print(f"  EURUSD単独: n={len(eur_s)} 本/年={len(eur_s)/(eur_days/365.25):.1f}  "
          f"CAGR={eur_cagr:+.1f}%  maxDD={eur_dd:.2f}%  CAGR/DD={eur_cagr/eur_dd if eur_dd>0 else float('inf'):.2f}")
    print(f"  プール6ペア: n={n_total} 本/年={n_total/yrs:.1f}  "
          f"CAGR={pool_cagr:+.1f}%  maxDD={pool_dd:.2f}%  CAGR/DD={pool_cagr/pool_dd if pool_dd>0 else float('inf'):.2f}")

    # ---------------- 3. ドル因子チェック ----------------
    print("\n" + "#" * 110)
    print("3. ドル因子チェック（構造法則2）")
    print("#" * 110)
    yearly = {}
    for name in FX6:
        s = trade_series(trades_by_pair[name])
        yearly[name] = s.groupby(s.index.year).sum()
    yr_df = pd.DataFrame(yearly).fillna(0.0)
    print("\n  年別Rの相関行列（6ペア）:")
    corr = yr_df.corr()
    print(corr.round(2).to_string())

    print("\n  同日重複率（ペア間、% of 各ペアのトレード日が他ペアと同日）:")
    day_sets = {name: set(pd.DatetimeIndex([t[0] for t in trades_by_pair[name]]).normalize()) for name in FX6}
    for i, a in enumerate(FX6):
        for b in FX6[i + 1:]:
            inter = day_sets[a] & day_sets[b]
            union_a = len(day_sets[a])
            pct = 100 * len(inter) / union_a if union_a else float("nan")
            print(f"    {a}x{b}: {len(inter)}日重複 / {a}の{union_a}日 = {pct:.1f}%")

    all_days = pd.Series(0, index=sorted(set.union(*day_sets.values())))
    for name in FX6:
        for d in day_sets[name]:
            all_days[d] += 1
    multi = 100 * (all_days >= 2).mean()
    print(f"\n  「その日2ペア以上が同時に建った」日の割合 = {multi:.1f}% (n_days={len(all_days)})")

    print("\n  DXY(usdx.r)日次リターンとの相関（2020+のみ、6ペア日次R合計）:")
    with contextlib.redirect_stderr(io.StringIO()):
        dxy = load_mt5_csv("/home/angelbell/dev/auto-trade/data/vantage_usdx.r_m5.csv")
    dxy_daily = dxy["close"].resample("1D").last().dropna()
    dxy_ret = dxy_daily.pct_change().dropna()
    dxy_ret.index = dxy_ret.index.tz_localize(None).normalize()
    pool_daily = pool_s.groupby(pool_s.index.normalize()).sum()
    pool_daily = pool_daily[pool_daily.index >= "2020-02-10"]
    joined = pd.DataFrame({"pool": pool_daily}).join(pd.DataFrame({"dxy": dxy_ret}), how="inner")
    joined = joined.dropna()
    if len(joined) > 20:
        rho = joined["pool"].corr(joined["dxy"])
        print(f"    n_days={len(joined)}  corr(6ペア日次R合計, DXY日次リターン) = {rho:+.3f}")
    else:
        print(f"    n不足(n={len(joined)})でskip")

    # ---------------- 4. 年代安定性 ----------------
    print("\n" + "#" * 110)
    print("4. 年代安定性: プールPF vs EURUSD単独PF（年代別）")
    print("#" * 110)
    for lo, hi, elabel in ERAS:
        pool_sub = [t for t in all_trades if lo <= pd.Timestamp(t[0]).year <= hi]
        eur_sub = filter_era(trades_by_pair["eurusd"], lo, hi)
        cs_pool = cell_stats(pool_sub)
        cs_eur = cell_stats(eur_sub)
        pf_pool = cs_pool['pf'] if cs_pool else float('nan')
        pf_eur = cs_eur['pf'] if cs_eur else float('nan')
        print(f"  {elabel:8s}: プールPF={pf_pool:.2f}  EURUSD単独PF={pf_eur:.2f}")

    print("\n" + "#" * 110)
    print("5. 多重比較の注記: これは新規採掘でなく凍結レシピ(C3/E3xS2/PDH-5pip)の6ペア適用。")
    print("   ロング限定・E3xS2選択の経緯は履歴に記録済み(ict_extliq_target.py 等)。判断はユーザー。")
    print("#" * 110)


if __name__ == "__main__":
    main()
