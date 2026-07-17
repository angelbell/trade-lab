"""follow-up1 で見えた「リフト(C3-C0)の年代減衰」が本物かノイズか、EURUSD固有か6ペア共通かを検定する。
n=34-55/年代・ブートストラップ無しで3点の単調減少は偶然でも1/6で出る --- 台帳に書く前にここを固める。

流用（車輪の再発明禁止）: ict_ground_lift_by_era の母集団構築(calibrate/build_c0)をそのまま呼ぶ
（EURUSD専用だったロジックをsymbol引数で6ペアに一般化しただけ、判定ロジックの複製はしない）。
ict_population.{canonical_setups, load_prepped, build}／ict_exec.walk／ict_extliq_target／
ict_dxy_smt.cost_tiers／ict_capture_decomp.{cell_stats, filter_era, S1_BUF, S2_BUF, run_cell}／
ict_condition_ablation.{calibrate, build_c0}／arb_common.Boot（月ブロック巡回レイアウトをそのまま
2種の検定(年代内CI・年代ラベル置換)に転用）。

主指標: ΔmeanR = meanR(C3)-meanR(C0)（加法的）。ΔPFは参考。E3×S2 primary、E3×S1 併記
（E1×S1はC0がn=0のため除外、E1×S2はC0の意味が薄い=fill率5%前後で不採用、主要2構成のみ）。

検定1: 年代内 巡回ブロック・ブートストラップ(k=3, 3000回) --- 各年代のリフトの中央値・95%CI。
検定2: 年代ラベルの月ブロック置換検定(3000回) --- 全史の月をシャッフルして「2018-20/2021-23/
       2024-26」を偽の年代バケツに割り当て直し(各年代の月数は保存)、統計量 S=リフト(fake2018-20)
       -リフト(fake2024-26) と 3年代の線形トレンドの傾き が、置換nullの中で実測より極端かを見る。
検定3(横断): 6ペアそれぞれで年代別リフト(点推定)を出し、何ペアで単調減衰しているかを数える。

先読み: 母集団生成は既存のまま(shift(1)済み)。ブートストラップ/置換検定は事後統計のみ。

Run: .venv/bin/python scratchpad/ict_lift_decay_test.py [--smoke] 2>&1 | tee scratchpad/out_ict_lift_decay_test.txt
"""
import sys, io, argparse, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import F_CANON, RR_CANON, walk
from ict_population import canonical_setups, load_prepped, build
from ict_extliq_target import make_ext_tgt_fn, EURUSD_LIM_FN, EURUSD_MA
from ict_dxy_smt import cost_tiers
from ict_capture_decomp import cell_stats, filter_era, S1_BUF, S2_BUF, run_cell
from ict_condition_ablation import calibrate, build_c0
from arb_common import Boot

FX6 = ["eurusd", "gbpusd", "usdjpy", "audusd", "nzdusd", "usdcad"]
ERAS = [(2018, 2020, "2018-20"), (2021, 2023, "2021-23"), (2024, 2026, "2024-26")]
STOPS = {"S1": S1_BUF, "S2": S2_BUF}


# ============================================================== 母集団構築(symbol汎用化)
def build_trades(name, smoke=False):
    with contextlib.redirect_stderr(io.StringIO()):
        df, tarr, dates, span = load_prepped(name)
    if smoke:
        dates = dates[-int(len(dates) * 0.3):]
    sp, cost = cost_tiers(name)["realistic"]
    tgt_fn = make_ext_tgt_fn("pdh", 5, name, "long")
    S3 = canonical_setups(df, tarr, dates, 0, use_fvg=True, fvg_min_atr=0.15, use_liq=True, liq_ns=(20, 40))
    atr_by_date = {rec["date"]: rec["long"]["atr"] for rec in S3 if rec["long"] is not None}
    calib = calibrate(df, S3, atr_by_date, sp, cost, tgt_fn)
    off_E3, depth_E3 = calib["E3"]["off_med"], calib["E3"]["depth_med"]
    C0 = build_c0(df, tarr, dates, off_E3, depth_E3, "E3")
    out = {}
    for sk, buf in STOPS.items():
        tr_c0 = walk(df, C0, F_CANON, RR_CANON, buf, sp, cost, "long", lim_fn=None, tgt_fn=tgt_fn)
        tr_c3 = walk(df, S3, F_CANON, RR_CANON, buf, sp, cost, "long", lim_fn=None, tgt_fn=tgt_fn)
        out[sk] = dict(C0=tr_c0, C3=tr_c3)
    return out, df, tarr, dates, span, S3, atr_by_date, sp, cost, tgt_fn


def era_trades(trades, lo, hi):
    return [t for t in trades if lo <= pd.Timestamp(t[0]).year <= hi]


def meanR(trades):
    if not trades:
        return np.nan
    return float(np.mean([t[1] for t in trades]))


def pf(trades):
    if not trades:
        return np.nan
    net = np.array([t[1] for t in trades])
    pos, neg = net[net > 0].sum(), -net[net < 0].sum()
    return pos / neg if neg > 0 else np.inf


def lift_meanR(tr_c0, tr_c3, lo, hi):
    c0 = era_trades(tr_c0, lo, hi)
    c3 = era_trades(tr_c3, lo, hi)
    if not c0 or not c3:
        return np.nan, len(c0), len(c3)
    return meanR(c3) - meanR(c0), len(c0), len(c3)


def lift_pf(tr_c0, tr_c3, lo, hi):
    c0 = era_trades(tr_c0, lo, hi)
    c3 = era_trades(tr_c3, lo, hi)
    if not c0 or not c3:
        return np.nan
    return pf(c3) - pf(c0)


# ============================================================== 検定1: 年代内ブロック・ブートストラップ
def boot_lift_ci(tr_c0_era, tr_c3_era, k=3, nb=3000, seed=20260717):
    if not tr_c0_era or not tr_c3_era:
        return None
    idx0 = pd.DatetimeIndex([t[0] for t in tr_c0_era])
    idx3 = pd.DatetimeIndex([t[0] for t in tr_c3_era])
    v0all = np.array([t[1] for t in tr_c0_era])
    v3all = np.array([t[1] for t in tr_c3_era])
    months = sorted(set(idx0.to_period("M")) | set(idx3.to_period("M")))
    if len(months) < k + 1:
        return None
    boot = Boot(months, nb=nb, k=k, seed=seed)
    mk0 = idx0.to_period("M"); mk3 = idx3.to_period("M")
    by0 = {m: v0all[mk0 == m] for m in months}
    by3 = {m: v3all[mk3 == m] for m in months}
    n0, n3 = len(v0all), len(v3all)
    out = np.full(len(boot.layout), np.nan)
    for i, seq in enumerate(boot.layout):
        a = np.concatenate([by0[months[j]] for j in seq])
        b = np.concatenate([by3[months[j]] for j in seq])
        if len(a) == 0 or len(b) == 0:
            continue
        out[i] = b[:n3].mean() - a[:n0].mean() if n3 <= len(b) and n0 <= len(a) else np.nan
    return out


# ============================================================== 検定2: 年代ラベルの月ブロック置換
def permutation_test(tr_c0_full, tr_c3_full, real_by_era, nb=3000, k=3, seed=20260717):
    """全史の月を巡回ブロックでシャッフルし、元の年代ごとの月数を保つ形で偽の3年代バケツに
    割り当て直す。統計量 S=lift(fake1)-lift(fake3) と 3点の線形トレンド傾き の null 分布を作る。"""
    idx0 = pd.DatetimeIndex([t[0] for t in tr_c0_full])
    idx3 = pd.DatetimeIndex([t[0] for t in tr_c3_full])
    v0all = np.array([t[1] for t in tr_c0_full])
    v3all = np.array([t[1] for t in tr_c3_full])
    months = sorted(set(idx0.to_period("M")) | set(idx3.to_period("M")))
    mk0 = idx0.to_period("M"); mk3 = idx3.to_period("M")
    by0 = {m: v0all[mk0 == m] for m in months}
    by3 = {m: v3all[mk3 == m] for m in months}

    n_months_era = []
    for lo, hi, elabel in ERAS:
        n_months_era.append(sum(1 for m in months if lo <= m.year <= hi))
    n1, n2, n3 = n_months_era
    need = n1 + n2 + n3
    boot = Boot(months, nb=nb, k=k, seed=seed)

    S_null = np.full(nb, np.nan)
    trend_null = np.full(nb, np.nan)
    xs = np.array([0, 1, 2])
    for i, seq in enumerate(boot.layout):
        if len(seq) < need:
            continue
        m1 = seq[:n1]; m2 = seq[n1:n1 + n2]; m3 = seq[n1 + n2:n1 + n2 + n3]
        lifts = []
        for mm in (m1, m2, m3):
            a = np.concatenate([by0[months[j]] for j in mm]) if len(mm) else np.array([])
            b = np.concatenate([by3[months[j]] for j in mm]) if len(mm) else np.array([])
            if len(a) == 0 or len(b) == 0:
                lifts.append(np.nan)
            else:
                lifts.append(b.mean() - a.mean())
        lifts = np.array(lifts)
        if np.any(np.isnan(lifts)):
            continue
        S_null[i] = lifts[0] - lifts[2]
        trend_null[i] = np.polyfit(xs, lifts, 1)[0]

    S_real = real_by_era[0] - real_by_era[2]
    trend_real = np.polyfit(xs, real_by_era, 1)[0]
    ok = ~np.isnan(S_null)
    pct_S = 100 * np.mean(S_null[ok] < S_real)     # 実測Sが小さいほど「減衰が大きい」= 下側パーセンタイルで見る
    ok_t = ~np.isnan(trend_null)
    pct_trend = 100 * np.mean(trend_null[ok_t] < trend_real)
    return dict(S_real=S_real, pct_S=pct_S, n_valid_S=int(ok.sum()),
                trend_real=trend_real, pct_trend=pct_trend, n_valid_trend=int(ok_t.sum()),
                S_null=S_null[ok], trend_null=trend_null[ok_t])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print("#" * 110)
    print("0. tie-back（EURUSD, C3×E1×S1 が n=313/win34.5/PF1.41 を再現するか）")
    print("#" * 110)
    trades_e, df_e, tarr_e, dates_e, span_e, S3_e, atr_e, sp_e, cost_e, tgt_e = build_trades("eurusd", args.smoke)
    tr_check, _ = run_cell(df_e, S3_e, S3_e, atr_e, "E1", "S1", sp_e, cost_e, tgt_e)
    net = np.array([t[1] for t in tr_check])
    print(f"  n={len(tr_check)}  win%={100*np.mean(net>0):.1f}  meanR={net.mean():+.3f}")
    if not args.smoke:
        print(f"  {'PASS' if len(tr_check)==313 else 'FAIL --- 要確認'}")

    print("\n" + "#" * 110)
    print("1+2. EURUSD: 年代別リフト(ΔmeanR・ΔPF)・年代内ブロックブートストラップCI・年代ラベル置換検定")
    print("#" * 110)

    perm_summ = {}
    for sk in ("S2", "S1"):
        label = "E3xS2 (primary)" if sk == "S2" else "E3xS1"
        print(f"\n  === {label} ===")
        tr_c0 = trades_e[sk]["C0"]; tr_c3 = trades_e[sk]["C3"]
        real_lifts = []
        for lo, hi, elabel in ERAS:
            dR, n0, n3 = lift_meanR(tr_c0, tr_c3, lo, hi)
            dPF = lift_pf(tr_c0, tr_c3, lo, hi)
            real_lifts.append(dR)
            c0e = era_trades(tr_c0, lo, hi); c3e = era_trades(tr_c3, lo, hi)
            boot_out = boot_lift_ci(c0e, c3e)
            if boot_out is not None:
                ok = ~np.isnan(boot_out)
                med = np.nanmedian(boot_out)
                lo95, hi95 = np.nanpercentile(boot_out[ok], [2.5, 97.5])
                print(f"    {elabel:10s}: n(C0)={n0:4d} n(C3)={n3:4d}  ΔmeanR(実測)={dR:+.3f}  ΔPF(実測)={dPF:+.2f}  "
                      f"ブートストラップ中央値={med:+.3f}  95%CI=[{lo95:+.3f},{hi95:+.3f}]  "
                      f"(sd={np.nanstd(boot_out[ok]):.3f})")
            else:
                print(f"    {elabel:10s}: n(C0)={n0:4d} n(C3)={n3:4d}  ΔmeanR(実測)={dR:+.3f}  ΔPF(実測)={dPF:+.2f}  "
                      f"ブートストラップ不能(月数不足)")

        if not any(np.isnan(real_lifts)):
            print(f"\n    年代ラベル置換検定(3000回, k=3ヶ月ブロック):")
            perm = permutation_test(tr_c0, tr_c3, real_lifts)
            print(f"      実測 S=lift(2018-20)-lift(2024-26) = {perm['S_real']:+.3f}  "
                  f"-> null下側 {perm['pct_S']:.1f}%ile (有効n={perm['n_valid_S']}/3000)")
            print(f"      実測 線形トレンド傾き = {perm['trend_real']:+.3f}/era  "
                  f"-> null下側 {perm['pct_trend']:.1f}%ile (有効n={perm['n_valid_trend']}/3000)")
            print(f"      null S帯: [{np.percentile(perm['S_null'],2.5):+.3f}, {np.percentile(perm['S_null'],97.5):+.3f}]  "
                  f"null trend帯: [{np.percentile(perm['trend_null'],2.5):+.3f}, {np.percentile(perm['trend_null'],97.5):+.3f}]")
            perm_summ[sk] = perm
        else:
            print("    年代のいずれかで n不足のため置換検定skip")

    print("\n" + "#" * 110)
    print("3. 横断（6ペア）: 年代別リフト(ΔmeanR点推定) --- 何ペアで単調減衰しているか")
    print("#" * 110)
    cross = {}
    for name in FX6:
        if name == "eurusd":
            trades_x = trades_e
        else:
            trades_x, *_ = build_trades(name, args.smoke)
        print(f"\n  --- {name} ---")
        for sk in ("S2", "S1"):
            label = "E3xS2" if sk == "S2" else "E3xS1"
            tr_c0 = trades_x[sk]["C0"]; tr_c3 = trades_x[sk]["C3"]
            lifts = []
            for lo, hi, elabel in ERAS:
                dR, n0, n3 = lift_meanR(tr_c0, tr_c3, lo, hi)
                lifts.append(dR)
            mono = "単調減衰" if (not any(np.isnan(lifts))) and lifts[0] > lifts[1] > lifts[2] else (
                   "n不足" if any(np.isnan(lifts)) else "非単調")
            print(f"    {label}: " + "  ".join(f"{ERAS[i][2]}:{lifts[i]:+.3f}" for i in range(3)) + f"   [{mono}]")
            cross[(name, sk)] = lifts

    n_mono_s2 = sum(1 for name in FX6 if not any(np.isnan(cross[(name, "S2")]))
                    and cross[(name, "S2")][0] > cross[(name, "S2")][1] > cross[(name, "S2")][2])
    n_mono_s1 = sum(1 for name in FX6 if not any(np.isnan(cross[(name, "S1")]))
                    and cross[(name, "S1")][0] > cross[(name, "S1")][1] > cross[(name, "S1")][2])
    print(f"\n  単調減衰ペア数: E3xS2 = {n_mono_s2}/6   E3xS1 = {n_mono_s1}/6")

    print("\n" + "#" * 110)
    print("判定(事前登録通り): 置換検定Pと年代内CIの重なりを見て判定すること。")
    print("P<=5%かつCI非重複 -> 減衰は台帳に書ける。P>10% -> n不足で判定不能、案1に委ねて閉じる。")
    print("#" * 110)


if __name__ == "__main__":
    main()
