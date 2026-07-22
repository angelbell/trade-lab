"""ICT 案1 の続き: なぜ EURUSD の「地面」(C0=条件なしのKZ指値) が 2024-26 で沈んだのか。
後付けの物語を禁じる設計 --- 6ペア×年代のパネルで「地面」と各特性を突き合わせ、
特性が地面を横断的に予言するかを見る（予言するならレジーム・ゲート候補＝構造法則3、
しないなら「なぜ沈んだか」は特定不能として閉じる）。

流用（車輪の再発明禁止）:
  - ict_condition_ablation.{calibrate, build_c0}     … C0(条件なし・E3浅指値×S2)母集団の生成
  - ict_population.{canonical_setups, load_prepped}  … C3(旗艦型)母集団
  - ict_exec.{walk, BUF, F_CANON, RR_CANON}
  - ict_extliq_target.make_ext_tgt_fn / ict_dxy_smt.cost_tiers
  - ict_capture_decomp.{cell_stats, filter_era}       … 統計・era切り出し
  - ict_alpha_decay.scan_mfe_mae                       … MAE中央値・損切り率(H2特性)
  - ict_audit.block_boot                               … ゲート候補の12ヶ月ブロック検定
  - research.regime_gate_lab.er / research.regime_statedet.variance_ratio … ER/VR（トレンド性）

新規実装: ドリフト(KZ窓・日足)・PDH距離/ATR・コスト/R の3特性の集計のみ（後段の統計は既存を再利用）。

パネル: 6ペア(eurusd/gbpusd/usdjpy/audusd/nzdusd/usdcad) × 3年代(2018-20/2021-23/2024-26) = 18セル。
応答変数 y = C0(E3×S2, 目標PDH-5pip, realistic) の PF・win%。

先読み: 全て既存関数のまま(build()のsweep/MSS/FVG判定・PDHのshift(1)は不変)。ER/VR/ドリフトは
era内の実現値集計であり、C0/C3のシグナル生成自体には使わない(事後の特性診断のみ、ゲート化する
場合のみ trailing 値を使う --- 後半の(c)ゲート検定で明記)。

Run: .venv/bin/python experiments/ict_ground_panel.py [--smoke] 2>&1 | tee experiments/out_ict_ground_panel.txt
"""
import sys, io, argparse, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import BUF, F_CANON, RR_CANON, walk, stats
from ict_population import canonical_setups, load_prepped
from ict_extliq_target import make_ext_tgt_fn
from ict_dxy_smt import cost_tiers
from ict_capture_decomp import cell_stats, filter_era, run_cell, S1_BUF, S2_BUF
from ict_condition_ablation import calibrate, build_c0
from ict_alpha_decay import scan_mfe_mae
from ict_audit import block_boot
from ict_fvg_anchor import fvg_anchor_fn
from research.regime_gate_lab import er
from research.regime_statedet import variance_ratio

FX6 = ["eurusd", "gbpusd", "usdjpy", "audusd", "nzdusd", "usdcad"]
ERAS = [(2018, 2020), (2021, 2023), (2024, 2026)]


# ============================================================== (a) 応答変数: C0(E3xS2) PF/win%
def ground_pf(df, tarr, dates, name, sp, cost, tgt_fn):
    S3 = canonical_setups(df, tarr, dates, 0, use_fvg=True, fvg_min_atr=0.15, use_liq=True, liq_ns=(20, 40))
    atr_by_date = {rec["date"]: rec["long"]["atr"] for rec in S3 if rec["long"] is not None}
    calib = calibrate(df, S3, atr_by_date, sp, cost, tgt_fn)
    off_E3, depth_E3 = calib["E3"]["off_med"], calib["E3"]["depth_med"]
    C0 = build_c0(df, tarr, dates, off_E3, depth_E3, "E3")
    trades = walk(df, C0, F_CANON, RR_CANON, S2_BUF, sp, cost, "long", lim_fn=None, tgt_fn=tgt_fn)
    out = {}
    for a, b in ERAS:
        sub = filter_era(trades, a, b)
        out[(a, b)] = cell_stats(sub)
    return out, S3, atr_by_date, trades


# ============================================================== (b) 特性群
def daily_series(df):
    g = df.groupby(df["_t"].dt.normalize())["close"].last()
    return g


def drift_features(df, S3):
    """KZ窓リターン平均・日足リターン平均（年代別、bps）。"""
    o, c = df["open"].values, df["close"].values
    kz_by_date = {rec["date"]: rec["long"]["kz"] for rec in S3 if rec["long"] is not None}
    dly = daily_series(df)
    dly_ret = dly.pct_change().dropna()
    out = {}
    for a, b in ERAS:
        kz_rets = []
        for d, (k0, k1) in kz_by_date.items():
            if a <= pd.Timestamp(d).year <= b and k1 - 1 < len(c):
                kz_rets.append((c[k1 - 1] - o[k0]) / o[k0])
        dr = dly_ret[(dly_ret.index.year >= a) & (dly_ret.index.year <= b)]
        out[(a, b)] = dict(kz_ret_bps=1e4 * np.mean(kz_rets) if kz_rets else np.nan,
                           daily_ret_bps=1e4 * dr.mean() if len(dr) else np.nan, n_kz=len(kz_rets))
    return out


def trend_features(df):
    """ER(2.5h=10本 / 1日=96本)・VR(q=10/96) を era内の15分終値から集計。"""
    c = df["close"]
    er10 = er(c, 10).values
    er96 = er(c, 96).values
    yr = df["_t"].dt.year.values
    logc = np.log(c.values)
    ret = np.diff(logc)                 # ret[i] = logc[i+1]-logc[i], aligned to yr[1:]
    yr_ret = yr[1:]
    out = {}
    for a, b in ERAS:
        m = (yr >= a) & (yr <= b)
        e10 = er10[m]; e10 = e10[~np.isnan(e10)]
        e96 = er96[m]; e96 = e96[~np.isnan(e96)]
        mr = (yr_ret >= a) & (yr_ret <= b)
        r = ret[mr]
        vr10 = variance_ratio(r, 10) if len(r) > 20 else np.nan
        vr96 = variance_ratio(r, 96) if len(r) > 200 else np.nan
        out[(a, b)] = dict(ER_2h5=e10.mean() if len(e10) else np.nan,
                           ER_1d=e96.mean() if len(e96) else np.nan, VR_2h5=vr10, VR_1d=vr96)
    return out


def h2_features(df, S3, name, sp, cost):
    """MAE中央値/平均・損切り率（年代別）--- C3×E1(真のFVG-CE)母集団で、ict_alpha_decay.scan_mfe_mae を流用。
    ⚠️ MAEは損切り到達時に1.0で打ち切る定義のため、損切り率が高い(>50%)とMAE"中央値"は1.00に張り付く
    （ふるまいとして正しいが差別化力が無い）。年代を跨いだ比較には mae_mean（打ち切り無しトレードの
    寄与を残す）を使う。"""
    lim_fn = fvg_anchor_fn("mid", "long")
    scans = scan_mfe_mae(df, S3, "long", lim_fn, sp, cost)
    out = {}
    for a, b in ERAS:
        sub = [x for x in scans if a <= pd.Timestamp(x["date"]).year <= b]
        if len(sub) < 5:
            out[(a, b)] = dict(mae_med=np.nan, mae_mean=np.nan, stop_rate=np.nan, n=len(sub))
            continue
        mae = np.array([x["mae"] for x in sub])
        out[(a, b)] = dict(mae_med=np.median(mae), mae_mean=mae.mean(),
                           stop_rate=100 * np.mean([x["stopped"] for x in sub]), n=len(sub))
    return out


def dist_cost_features(df, S3, atr_by_date, name, sp, cost, tgt_fn):
    """PDH距離/ATR中央値・コスト/リスク中央値・ATR%(価格比)平均 --- C3×E1×S1 trade_log から。"""
    tr, tlog = run_cell(df, S3, S3, atr_by_date, "E1", "S1", sp, cost, tgt_fn)
    out = {}
    for a, b in ERAS:
        rows = [r for r in tlog if a <= pd.Timestamp(r["date"]).year <= b]
        if not rows:
            out[(a, b)] = dict(dist_atr_med=np.nan, cost_risk_med=np.nan, atr_pct_mean=np.nan, n=0)
            continue
        dist_atr, cost_risk, atr_pct = [], [], []
        for r in rows:
            A = atr_by_date.get(r["date"])
            if not A or A <= 0:
                continue
            risk = r["entry"] - r["stop"]
            dist_atr.append(r["r_rr"] * risk / A)
            cost_risk.append(sp / risk if risk > 0 else np.nan)
            atr_pct.append(100 * A / r["entry"])
        out[(a, b)] = dict(dist_atr_med=np.median(dist_atr) if dist_atr else np.nan,
                           cost_risk_med=np.median(cost_risk) if cost_risk else np.nan,
                           atr_pct_mean=np.mean(atr_pct) if atr_pct else np.nan, n=len(rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    panel = {}   # (sym, era) -> dict of all features + y
    print("#" * 110)
    print("パネル構築: 6ペア x 3年代（y=C0(E3xS2,目標PDH-5pip,realistic)のPF/win% + 5特性群）")
    print("#" * 110)
    for name in FX6:
        with contextlib.redirect_stderr(io.StringIO()):
            df, tarr, dates, span = load_prepped(name)
        if args.smoke:
            dates = dates[-int(len(dates) * 0.3):]
        sp, cost = cost_tiers(name)["realistic"]
        tgt_fn = make_ext_tgt_fn("pdh", 5, name, "long")

        y, S3, atr_by_date, c0_trades = ground_pf(df, tarr, dates, name, sp, cost, tgt_fn)
        drift = drift_features(df, S3)
        trend = trend_features(df)
        h2 = h2_features(df, S3, name, sp, cost)
        dc = dist_cost_features(df, S3, atr_by_date, name, sp, cost, tgt_fn)

        print(f"\n  --- {name} ---")
        for a, b in ERAS:
            cs = y[(a, b)]
            pf = cs["pf"] if cs else np.nan
            win = cs["win"] if cs else np.nan
            n = cs["n"] if cs else 0
            d, t, h, c = drift[(a, b)], trend[(a, b)], h2[(a, b)], dc[(a, b)]
            print(f"    {a}-{b}: y[C0 n={n:4d} PF={pf:.2f} win%={win:.1f}]  "
                  f"drift[kz={d['kz_ret_bps']:+.1f}bps daily={d['daily_ret_bps']:+.1f}bps]  "
                  f"trend[ER2.5h={t['ER_2h5']:.3f} ER1d={t['ER_1d']:.3f} VR2.5h={t['VR_2h5']:.2f} "
                  f"VR1d={t['VR_1d']:.2f}]  h2[MAE中央={h['mae_med']:.2f} MAE平均={h['mae_mean']:.2f} "
                  f"損切率={h['stop_rate']:.1f}%]  "
                  f"dist/cost[PDH距離/ATR={c['dist_atr_med']:.2f} cost/risk={c['cost_risk_med']:.3f} "
                  f"ATR%={c['atr_pct_mean']:.3f}]")
            h_clean = {k: v for k, v in h.items() if k != "n"}
            c_clean = {k: v for k, v in c.items() if k != "n"}
            panel[(name, a, b)] = dict(pf=pf, win=win, n=n, **d, **t, **h_clean, **c_clean)

    # ---------------- (c) 横断予言力: Spearman(feature, y=PF) across 18 cells ----------------
    print("\n" + "#" * 110)
    print("横断予言力: 18セル(6ペアx3年代)で 各特性 vs C0のPF/win% の順位相関(Spearman)")
    print("#" * 110)
    rows = [v for v in panel.values() if not np.isnan(v["pf"])]
    feat_keys = ["kz_ret_bps", "daily_ret_bps", "ER_2h5", "ER_1d", "VR_2h5", "VR_1d",
                "mae_med", "mae_mean", "stop_rate", "dist_atr_med", "cost_risk_med", "atr_pct_mean"]
    pf_arr = np.array([v["pf"] for v in rows])
    win_arr = np.array([v["win"] for v in rows])
    print(f"  有効セル数 n={len(rows)}")
    print(f"  {'特性':<16}{'rho(PF)':>10}{'rho(win%)':>10}")
    for k in feat_keys:
        x = np.array([v[k] for v in rows])
        ok = ~np.isnan(x) & ~np.isnan(pf_arr)
        if ok.sum() < 6:
            print(f"  {k:<16}{'n不足':>10}")
            continue
        rho_pf = pd.Series(x[ok]).corr(pd.Series(pf_arr[ok]), method="spearman")
        rho_win = pd.Series(x[ok]).corr(pd.Series(win_arr[ok]), method="spearman")
        print(f"  {k:<16}{rho_pf:>+10.2f}{rho_win:>+10.2f}")

    # ---------------- EURUSD固有か市場全体か: 特性の年代推移を6ペア並べる ----------------
    print("\n" + "#" * 110)
    print("EURUSD固有か市場全体か: 主要特性の年代推移(6ペア並記)")
    print("#" * 110)
    for k in ["kz_ret_bps", "daily_ret_bps", "mae_mean", "stop_rate", "dist_atr_med", "ER_2h5", "VR_2h5"]:
        print(f"\n  {k}:")
        for name in FX6:
            vals = "  ".join(f"{a}-{b}:{panel[(name,a,b)][k]:+.2f}" if not np.isnan(panel[(name,a,b)][k])
                             else f"{a}-{b}:n/a" for a, b in ERAS)
            print(f"    {name:<8}{vals}")

    print("\n" + "#" * 110)
    print("判定に必要な一言: 上のrho(特に|rho|>=0.5)を見て、どの特性が『地面』を横断的に予言するか判定すること。")
    print("予言する特性が無ければ「なぜ沈んだか」は特定不能＝後付け禁止、案1(Pine A/B前向き)に委ねて閉じる。")
    print("#" * 110)


if __name__ == "__main__":
    main()
