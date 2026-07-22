"""E1: 直交サイズ変数の統合スタック（btc15m_L単独レッグ）。
凍結カード: PDH×HH4H階段 x 日足レジーム x ICTラベル(A∧B,X=48) を乗算合成し、現行(PDHソフト0.5のみ)
と比べる。エントリーは1本も変えない、サイズ写像だけを変える。裁定器は ict_size_transplant.py と同一
(Boot.equal_dd_cagr による同ブートストラップ中央値DD揃えCAGR比較・40seedランダムnull・巡回ブロック・
逆ダミー)。

母集団（凍結・変更禁止）: btc15m_L canonical = BASE+gate_kama14/240min+pullback_frac0.3+rr4.5+
  fill_win200、cost=$15、swap=年30%（ict_size_transplant.build_btc15m_L と同じ式）。
  tie-back: 生run(サイズ・コスト前) n=763・meanR≈+0.567 を最初に確認。

流用（車輪の再発明禁止）:
  - breakout_wave.{run, resample, kama_adaptive, swings_zigzag}／radar_gate_race.BASE
  - btc15m_htf_size.hh4h_series と同じ考え方（4hスイング高値、confirm後にshift(1)、15分へffill展開）
    --- ただし母集団は breakout_wave.run() の canonical trade（本タスクの母集団固定）に合わせるため
    ATR計算を pandas_ta.atr に差し替えて再実装（アルゴリズムは同一、依存モジュールだけ差し替え）。
  - A_daily_regime.py と同じ日足SMA150判定（終値[確定]<SMA150、shift(1)+ffill）。
  - ict_size_transplant.{compute_labels, RISK_PCT, block_boot_beat}（ICTラベルA∧B・裁定器を再利用）。
  - arb_common.{Boot, cd}。

サイズ写像（乗算合成、ラベルカードの定義通り）:
  1. PDH×HH4H階段: 終値が前日高値の上 かつ 直近確定4Hスイング高値の上 = 1.0 / 片方のみ = 0.5 /
     どちらも無し = 0.25。（現行PDHソフト0.5をこの階段に置換 --- 成分1を使うセルはPDHソフトを外す）
  2. 日足レジーム: 日足終値[確定]<SMA150 なら x0.75。
  3. ICTラベル: A∧B(X=48)無しの玉は x0.5。

コスト式: risk=t.risk/W, R=t.R*W - 15/risk - swap(年30%)*(e_px/risk)*hold
（build_btc15m_L の式の W を PDHソフト固定値から任意の重み配列へ一般化しただけ、新規ロジックなし）。

Run: .venv/bin/python experiments/stack_size_btc15mL.py [--smoke] 2>&1 | tee experiments/out_stack_size_btc15mL.txt
"""
import sys, io, contextlib, argparse, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import numpy as np
import pandas as pd
import pandas_ta as ta

from arb_common import Boot, cd
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample, kama_adaptive, swings_zigzag
from radar_gate_race import BASE
from ict_size_transplant import compute_labels, RISK_PCT, block_boot_beat

ROOT = "/home/angelbell/dev/auto-trade"
BTC_PCT_YR = 30.0
NB_MAIN = 200
NREP_NULL = 40
RNG = np.random.default_rng(20260717)


# ============================================================== 母集団(凍結)
def build_population():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    CFG = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3, "rr": 4.5,
           "fill_win": 200, "fwd": 500}
    with contextlib.redirect_stderr(io.StringIO()):
        t = run(d15, SimpleNamespace(**CFG))
    print(f"  tie-back(生run): n={len(t)}  meanR={t['R'].mean():+.3f}  (既知: n=763, meanR≈+0.567)")
    ii = d15.index.get_indexer(t["time"])
    return d15, t, ii


def apply_size(t, W):
    """build_btc15m_L の式の重みを PDHソフト固定値(1.0/0.5)から任意の配列 W へ一般化しただけ。"""
    risk = t["risk"].values / W
    R = (t["R"].values * W - 15.0 / risk
         - (BTC_PCT_YR / 365.0 / 100.0) * (t["e_px"].values / risk) * t["hold"].values)
    return R


# ============================================================== 成分1: PDH x HH4H 階段
def hh4h_series(d15):
    """直近 確定 4Hスイング高値。btc15m_htf_size.hh4h_series と同じアルゴリズム
    （swings_zigzag+shift(1)）、ATRだけ pandas_ta.atr に差し替え。"""
    h4 = d15.resample("4h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    a4 = ta.atr(h4["high"], h4["low"], h4["close"], 14).values
    sw = swings_zigzag(h4["high"].values, h4["low"].values, a4, 2.0)
    s = pd.Series(np.nan, index=h4.index)
    for (ci, pi, px, kind) in sw:
        if kind == +1:
            s.iloc[ci] = px
    return s.ffill().shift(1).reindex(d15.index, method="ffill").values


def comp1_ladder(d15, t, ii):
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    hh4 = hh4h_series(d15)
    e_px = t["e_px"].values
    above_pdh = e_px > pdh[ii]
    above_hh4 = np.where(np.isfinite(hh4[ii]), e_px > hh4[ii], False)
    both = above_pdh & above_hh4
    one = above_pdh ^ above_hh4
    W = np.where(both, 1.0, np.where(one, 0.5, 0.25))
    return W, above_pdh, above_hh4


def comp1_ladder_reverse(above_pdh, above_hh4):
    """逆ダミー: 階段の 1.0<->0.25 を入れ替え(0.5はそのまま)。"""
    both = above_pdh & above_hh4
    one = above_pdh ^ above_hh4
    return np.where(both, 0.25, np.where(one, 0.5, 1.0))


# ============================================================== 成分2: 日足レジーム
def comp2_daily(d15, t, ii):
    dly = d15["close"].resample("1D").last().dropna()
    sma150 = dly.rolling(150).mean()
    down = (dly < sma150).shift(1)
    down_15 = down.reindex(d15.index, method="ffill").fillna(False).values
    down_at_b = down_15[ii]
    W = np.where(down_at_b, 0.75, 1.0)
    return W, down_at_b


def comp2_daily_reverse(down_at_b):
    return np.where(down_at_b, 1.25, 1.0)


# ============================================================== 成分3: ICTラベル A∧B, X=48
def comp3_ict(d15, t, ii, x=48):
    labelA, labelB = compute_labels(d15, t, ii, x)
    AB = labelA & labelB
    W = np.where(AB, 1.0, 0.5)
    return W, AB


def comp3_ict_variant(AB, weak_w):
    return np.where(AB, 1.0, weak_w)


# ============================================================== 統計/裁定
def cell_report(label, R, ti):
    """R は生のR倍率(risk単位)。cd()/Boot に渡す口座複利曲線は RISK_PCT(1%)で縮尺してから作る
    【修正済みバグ: 以前は生Rをそのままcd()に渡し(=1トレードで口座を(1+R)倍)、DDが100%超に
    破綻していた。ict_size_transplant.py で踏んだのと同じ穴】。win%/PF/totR は生R基準(縮尺非依存)。"""
    order = np.argsort(ti.values)
    ti_s = ti[order]; R_s = np.asarray(R)[order]
    s_scaled = pd.Series(R_s * RISK_PCT, index=ti_s)
    days = max((ti_s[-1] - ti_s[0]).days, 1)
    cagr, dd = cd(s_scaled.values, days)
    net = R_s
    pos, neg = net[net > 0].sum(), -net[net <= 0].sum()
    pf = pos / neg if neg > 0 else np.inf
    win = 100 * np.mean(net > 0)
    totR = net.sum()
    print(f"\n  --- {label} ---")
    print(f"    n={len(net)}  win%={win:.1f}  PF={pf:.2f}  totR(R単位)={totR:+.1f}  "
          f"実測CAGR(1%risk複利)={cagr:+.1f}%  実測maxDD={dd:.2f}%  "
          f"実測CAGR/DD={cagr/dd if dd>0 else float('inf'):.2f}")
    by_year = pd.Series(net, index=ti_s).groupby(ti_s.year).sum()
    print("    年別totR(R単位): " + "  ".join(f"{y}:{v:+.1f}" for y, v in by_year.items()))
    return s_scaled


def equal_dd_result(boot, s, D0):
    cagr, scale = boot.equal_dd_cagr(s, D0)
    return cagr, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print("#" * 110)
    print("母集団 tie-back")
    print("#" * 110)
    d15, t, ii = build_population()
    if args.smoke:
        m = pd.DatetimeIndex(t["time"]) >= pd.Timestamp("2024-01-01", tz=pd.DatetimeIndex(t["time"]).tz)
        t = t[m].reset_index(drop=True); ii = ii[m]

    # ---- baseline: 現行形(PDHソフト0.5のみ) ----
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    w_pdh = np.where(t["e_px"].values > pdh[ii], 1.0, 0.5)
    R_base = apply_size(t, w_pdh)
    ti = pd.DatetimeIndex(t["time"])

    print("\n" + "#" * 110)
    print("ベースライン（現行形=PDHソフト0.5のみ）")
    print("#" * 110)
    s_base = cell_report("baseline (PDH soft 0.5)", R_base, ti)

    months = sorted(s_base.index.to_period("M").unique())
    boot = Boot(months, nb=NB_MAIN, k=3, seed=20260717)
    D0 = boot.dd_median(s_base)
    cagr_base, _ = boot.equal_dd_cagr(s_base, D0)
    print(f"    同DD({D0:.2f}%)でのbaseline CAGR = {cagr_base:+.1f}%  (以降はこのD0に揃えて比較)")

    # ---- 成分計算 ----
    W1, above_pdh, above_hh4 = comp1_ladder(d15, t, ii)
    W2, down_at_b = comp2_daily(d15, t, ii)
    W3, AB = comp3_ict(d15, t, ii, x=48)
    print(f"\n  成分あり率: HTF階段(both)={100*np.mean(above_pdh&above_hh4):.1f}%  "
          f"日足↓={100*np.mean(down_at_b):.1f}%  ICT(A∧B)あり={100*np.mean(AB):.1f}%")

    def eval_and_report(label, W):
        R = apply_size(t, W)
        s = cell_report(label, R, ti)
        cagr, scale = boot.equal_dd_cagr(s, D0)
        print(f"    同DD({D0:.2f}%)でのCAGR = {cagr:+.1f}%  (対baseline差={cagr-cagr_base:+.1f}pt)  "
              f"スケール={scale:.3f}")
        return s, cagr

    print("\n" + "#" * 110)
    print("成分単独x3（各成分だけ、PDHソフトは外す）")
    print("#" * 110)
    s1, cagr1 = eval_and_report("成分1単独: HTF階段(PDH x HH4H)", W1)
    s2, cagr2 = eval_and_report("成分2単独: 日足レジーム(SMA150)", W2)
    s3, cagr3 = eval_and_report("成分3単独: ICTラベル(A∧B,X=48)", W3)

    print("\n" + "#" * 110)
    print("フルスタック（1x2x3）")
    print("#" * 110)
    Wfull = W1 * W2 * W3
    s_full, cagr_full = eval_and_report("フルスタック", Wfull)

    print("\n" + "#" * 110)
    print("フルから1成分ずつ抜いた3形（限界利得の分解）")
    print("#" * 110)
    s_no1, cagr_no1 = eval_and_report("フル - 成分1(HTF階段抜き)", W2 * W3)
    s_no2, cagr_no2 = eval_and_report("フル - 成分2(日足抜き)", W1 * W3)
    s_no3, cagr_no3 = eval_and_report("フル - 成分3(ICT抜き)", W1 * W2)
    print(f"\n    限界利得(フル - (フル-成分)): 成分1={cagr_full-cagr_no1:+.1f}pt  "
          f"成分2={cagr_full-cagr_no2:+.1f}pt  成分3={cagr_full-cagr_no3:+.1f}pt")

    print("\n" + "#" * 110)
    print("外挿: フルスタックでICT成分だけ x0.25 / x0 に振った2形")
    print("#" * 110)
    W_ict025 = comp3_ict_variant(AB, 0.25)
    W_ict0 = comp3_ict_variant(AB, 0.0)
    s_e025, cagr_e025 = eval_and_report("フル、ICT弱=0.25", W1 * W2 * W_ict025)
    if np.all(W_ict0[~AB] == 0):
        # W=0 だと risk=inf/div-by-zero になるため、それらのトレードは寄与ゼロとして除外して評価
        mask = W_ict0 > 0
        Wz = W1 * W2 * np.where(AB, 1.0, 1e-9)
        s_e0, cagr_e0 = eval_and_report("フル、ICT弱=0(実質フィルタ)", Wz)

    print("\n" + "#" * 110)
    print("同数ランダムサイズnull(40回): フルスタックの重みの多重集合をシャッフルして張り直す")
    print("#" * 110)
    null_diffs = []
    for seed in range(NREP_NULL):
        perm = np.random.default_rng(20260717 + seed).permutation(len(Wfull))
        W_shuf = Wfull[perm]
        R_shuf = apply_size(t, W_shuf)
        s_shuf = pd.Series(R_shuf * RISK_PCT, index=ti).sort_index()
        cagr_shuf, _ = boot.equal_dd_cagr(s_shuf, D0)
        null_diffs.append(cagr_shuf - cagr_base)
    null_diffs = np.array(null_diffs)
    real_diff = cagr_full - cagr_base
    pct = 100 * np.mean(null_diffs < real_diff)
    print(f"    実測差(フル-baseline)={real_diff:+.1f}pt  null帯=[{np.percentile(null_diffs,2.5):+.1f},"
          f"{np.percentile(null_diffs,97.5):+.1f}]pt (中央値={np.median(null_diffs):+.1f})  -> {pct:.0f}%ile")

    print("\n" + "#" * 110)
    print("巡回ブロック・ブートストラップ 1/3/6/12mo: P(フルスタックがbaselineを同DD-CAGRで上回る)")
    print("#" * 110)
    for k in (1, 3, 6, 12):
        p = block_boot_beat(s_base, s_full, months, k, nb=300)
        print(f"    k={k:>2}mo: P={p:.0f}%")

    print("\n" + "#" * 110)
    print("逆ダミー: 写像を反転（階段0.25<->1.0、日足↓でx1.25、ラベル無しでx1.5）")
    print("#" * 110)
    W1_rev = comp1_ladder_reverse(above_pdh, above_hh4)
    W2_rev = comp2_daily_reverse(down_at_b)
    W3_rev = comp3_ict_variant(AB, 1.5)
    s_rev, cagr_rev = eval_and_report("逆ダミー フルスタック", W1_rev * W2_rev * W3_rev)
    print(f"\n    逆ダミー vs baseline: {'機構整合(悪化,OK)' if cagr_rev < cagr_base else '⚠️機構不整合(悪化せず)'}")
    print(f"    逆ダミー vs フルスタック: {'機構整合(フルが優位,OK)' if cagr_full > cagr_rev else '⚠️機構不整合'}")

    print("\n" + "#" * 110)
    print("追加tie-back: HH4H階段単独セルが s07既知値(PF1.62->1.75, maxDD20.4%->10.9%相当の方向)と整合するか")
    print("#" * 110)
    s_base_stats = pd.Series(R_base, index=ti)
    s1_stats = pd.Series(apply_size(t, W1), index=ti)
    pf_base = (s_base_stats[s_base_stats > 0].sum() / -s_base_stats[s_base_stats <= 0].sum())
    pf_1 = (s1_stats[s1_stats > 0].sum() / -s1_stats[s1_stats <= 0].sum())
    dd_base = cd(s_base_stats.sort_index().values * RISK_PCT, days=(ti.max() - ti.min()).days)[1]
    dd_1 = cd(s1_stats.sort_index().values * RISK_PCT, days=(ti.max() - ti.min()).days)[1]
    print(f"    baseline(PDHソフト): PF={pf_base:.2f}  maxDD={dd_base:.2f}%")
    print(f"    HTF階段単独        : PF={pf_1:.2f}  maxDD={dd_1:.2f}%")
    print(f"    方向: PF{'改善' if pf_1>pf_base else '非改善'}  DD{'改善(縮小)' if dd_1<dd_base else '非改善'}")


if __name__ == "__main__":
    main()
