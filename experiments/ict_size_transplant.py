"""ICT改善案3: 生き残った検出器(狩り→reclaim / FVG-displacement)を、N・コスト検証済みの
btc15m_L・gold15m に「サイズ変数」として移植する（フィルタでなくサイズ＝法則9系・9b）。
台帳 s01:30/41 の next-lever。エントリーは1本も増減させない、ラベルを付けてサイズを変えるだけ。

母集団（凍結・変更禁止、既存の正典スクリプトの数式をそのまま複製 = 車輪の再発明ではなく
"検算可能な形での転記"。tie-back で数値一致を検査する）:
  btc15m_L: L_daily_size_standalone.py と同じ式（BASE+gate_kama14/240min+pullback0.3+rr4.5+
            fill_win200、PDHソフト0.5・$15コスト・年30%swap込み）。tie-back: n≈763・meanR≈+0.567
            (コスト/swap抜きの生run値。PDHソフト+コスト+swap込みの最終meanRは別途照合)。
  gold15m : book_deployed_spec.py と同じ式（m5→15min resample・pullback0.25・ext-cap8%・rr4・
            日足SMA150、fill_win200、$0.3コスト）。

ICTラベル（先読み厳禁・シグナル足 b=約定足の直前確定足のみ。b自体は使わない）:
  ラベルA(狩り→reclaim): 確定フラクタル安値(n=2、breakout_wave.swings_pivotをそのまま流用) または
    前日安値(PDL, shift(1)+ffill) を、[b-X, b) で low が下抜け、かつ close[b-1](直前確定足)で
    その水準の上に回復していること。
  ラベルB(FVG/displacement): ブレイク脚[leg_start, b) 内に bullish FVG >= 0.15*ATR が存在。
    leg_start = b - base_bars（run()が返す base_bars 列を流用。厳密には base_bars は"真のブレイク
    確定足"までの本数で b=約定足とは異なるバー基準 --- 近似である旨を明記、FVG検出は
    ict_population.bullish_fvg_size をそのまま使う）。
  ラベルA∧B = 両方。MSSはブレイク自体が兼ねるため別判定しない。

サイズ・オーバーレイ: ラベル無し側の risk を m in {0.5,0.75,1.0,1.25} 倍。判定は「同じ
ブートストラップ中央値maxDDに揃えたCAGR」(arb_common.Boot.equal_dd_cagr)。同数ランダムラベルnull
(40 seed)・巡回ブロック・ブートストラップ(1/3/6/12mo, P(治療がbaselineを上回る))・逆ダミー(m=1.25)
も出す。X in {48,96}, 両銘柄。

Run: .venv/bin/python experiments/ict_size_transplant.py [--smoke] 2>&1 | tee experiments/out_ict_size_transplant.txt
"""
import sys, io, contextlib, argparse, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import numpy as np
import pandas as pd
import pandas_ta as ta

from arb_common import Boot, cd, months_union
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample, kama_adaptive, swings_pivot
from radar_gate_race import BASE
from ict_population import bullish_fvg_size

ROOT = "/home/angelbell/dev/auto-trade"
X_LIST = [48, 96]
M_LIST = [0.5, 0.75, 1.0, 1.25]
NB_MAIN = 200          # equal_dd_cagr は24回の二分探索xこの回数だけdd_medianを回すため軽量化必須
NREP_NULL = 40
RNG = np.random.default_rng(20260717)


# ============================================================== 母集団: btc15m_L (canonical, 凍結)
def build_btc15m_L():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    CFG = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3, "rr": 4.5,
           "fill_win": 200, "fwd": 500}
    with contextlib.redirect_stderr(io.StringIO()):
        t = run(d15, SimpleNamespace(**CFG))
    print(f"  [btc15m_L] 生run tie-back: n={len(t)}  meanR={t['R'].mean():+.3f}  (既知: n≈763, meanR≈+0.567)")

    ii = d15.index.get_indexer(t["time"])
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    w = np.where(t["e_px"].values > pdh[ii], 1.0, 0.5)
    risk = t["risk"].values / w
    BTC_PCT_YR = 30.0
    Rfinal = (t["R"].values * w - 15.0 / risk
             - (BTC_PCT_YR / 365.0 / 100.0) * (t["e_px"].values / risk) * t["hold"].values)
    ti = pd.DatetimeIndex(t["time"])
    print(f"  [btc15m_L] サイズ+コスト+swap込み: n={len(Rfinal)}  meanR={Rfinal.mean():+.3f}")
    return d15, t, ii, Rfinal, ti


# ============================================================== 母集団: gold15m (canonical, 凍結)
def build_gold15m():
    with contextlib.redirect_stderr(io.StringIO()):
        g15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
    CFG = {**BASE, "daily_sma": 150, "daily_slope_k": 10, "ext_cap": 8.0, "pullback_frac": 0.25,
           "fill_win": 200, "fwd": 500}
    with contextlib.redirect_stderr(io.StringIO()):
        t = run(g15, SimpleNamespace(**CFG))
    print(f"  [gold15m] 生run tie-back: n={len(t)}  meanR={t['R'].mean():+.3f}")
    Rfinal = t["R"].values - 0.3 / t["risk"].values
    ti = pd.DatetimeIndex(t["time"])
    print(f"  [gold15m] コスト込み: n={len(Rfinal)}  meanR={Rfinal.mean():+.3f}")
    ii = g15.index.get_indexer(t["time"])
    return g15, t, ii, Rfinal, ti


# ============================================================== ICTラベル
def compute_labels(d, t, ii, X):
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    atr = ta.atr(d["high"], d["low"], d["close"], 14).values
    n = len(d)

    # フラクタル安値(n=2) --- breakout_wave.swings_pivot をそのまま流用
    sw = swings_pivot(h, l, 2)
    frac_lo = [(ci, price) for (ci, pi, price, kind) in sw if kind == -1]
    frac_lo.sort()
    confirms = np.array([x[0] for x in frac_lo])
    prices = np.array([x[1] for x in frac_lo])

    pdl = pd.Series(l, index=d.index).groupby(d.index.normalize()).transform("min")
    pdl_daily = pd.Series(l, index=d.index).resample("1D").min().shift(1)
    pdl_arr = pdl_daily.reindex(d.index, method="ffill").values

    base_bars = t["base_bars"].values

    labelA = np.zeros(len(ii), dtype=bool)
    labelB = np.zeros(len(ii), dtype=bool)
    for k, b in enumerate(ii):
        lo_win = max(0, b - X)
        if lo_win >= b:
            continue
        # ---- Label A: hunt + reclaim ----
        candidates = []
        idx = np.searchsorted(confirms, b) - 1
        while idx >= 0 and confirms[idx] >= lo_win:
            candidates.append(prices[idx]); idx -= 1
            if len(candidates) >= 3:
                break
        if np.isfinite(pdl_arr[b]):
            candidates.append(pdl_arr[b])
        hit = False
        for lvl in candidates:
            if (l[lo_win:b] < lvl).any() and b >= 1 and c[b - 1] > lvl:
                hit = True; break
        labelA[k] = hit
        # ---- Label B: FVG in break leg ----
        bb = base_bars[k]
        if np.isfinite(bb) and bb > 2:
            leg_start = max(0, int(b - bb))
            A = atr[b - 1] if b >= 1 and np.isfinite(atr[b - 1]) else np.nan
            if np.isfinite(A) and A > 0 and leg_start < b - 1:
                sz, edges = bullish_fvg_size(h, l, A, leg_start, min(b, n - 2), 0.15)
                labelB[k] = sz is not None
    return labelA, labelB


# ============================================================== 統計・サイジング
def layer_stats(R, flag):
    x = R[flag]
    n = len(x)
    if n == 0:
        return None
    pos, neg = x[x > 0].sum(), -x[x <= 0].sum()
    return dict(n=n, win=100 * np.mean(x > 0), meanR=x.mean(), pf=(pos / neg if neg > 0 else np.inf))


def print_layer(name, s):
    if s is None:
        print(f"      {name:<12} n=0")
        return
    print(f"      {name:<12} n={s['n']:5d}  win%={s['win']:5.1f}  meanR={s['meanR']:+.3f}  PF={s['pf']:5.2f}")


RISK_PCT = 0.01   # 1トレード1%リスク。R(倍率)をこれで縮尺してからCAGR/DDへ(=口座の複利曲線)。
                  # 【修正済みバグ】旧版はこの縮尺を欠いておりcd()に生Rを渡していた＝1トレードで
                  # 口座を(1+R)倍する計算になり、D0(ブートストラップ中央値maxDD)が100%超という
                  # 破綻値になっていた(全セルNaN/-100%の原因)。


def sized_series(R, ti, flag_labeled, m):
    """ラベル無し側(flag_labeled=False)の risk を m 倍 -> R も m 倍(risk比例)。ラベルあり側は1.0固定。
    RISK_PCTで口座複利スケールに変換する。"""
    mult = np.where(flag_labeled, 1.0, m)
    return pd.Series(R * mult * RISK_PCT, index=ti)


def block_boot_beat(base_s, treat_s, months, k, nb=NB_MAIN, seed=20260717):
    boot = Boot(months, nb=nb, k=k, seed=seed)
    mkb = base_s.index.to_period("M"); mkt = treat_s.index.to_period("M")
    byb = {m: base_s.values[mkb == m] for m in months}
    byt = {m: treat_s.values[mkt == m] for m in months}
    nb_, nt_ = len(base_s), len(treat_s)
    days = max((base_s.index[-1] - base_s.index[0]).days, 1)
    wins = 0; valid = 0
    for seq in boot.layout:
        vb = np.concatenate([byb[m] for m in [months[j] for j in seq]])[:nb_]
        vt = np.concatenate([byt[m] for m in [months[j] for j in seq]])[:nt_]
        if len(vb) == 0 or len(vt) == 0:
            continue
        rb = cd(vb, days); rt = cd(vt, days)
        rdd_b = rb[0] / rb[1] if rb[1] > 0 else -999
        rdd_t = rt[0] / rt[1] if rt[1] > 0 else -999
        wins += (rdd_t > rdd_b); valid += 1
    return 100 * wins / valid if valid else np.nan


def evaluate_label(inst_name, label_name, R, ti, flag, X):
    n_total = len(R)
    n_lab = int(flag.sum())
    yrs = pd.Series(ti[flag]).dt.year.value_counts().sort_index()
    spread = "/".join(f"{y}:{v}" for y, v in yrs.items())
    print(f"\n    --- {inst_name} label={label_name} X={X} --- ラベルあり率={100*n_lab/n_total:.1f}% "
          f"(n={n_lab}/{n_total})")
    print(f"      年別散らばり(ラベルあり): {spread}")
    print_layer("ラベルあり", layer_stats(R, flag))
    print_layer("ラベルなし", layer_stats(R, ~flag))

    base_s = pd.Series(R * RISK_PCT, index=ti).sort_index()
    months = sorted(base_s.index.to_period("M").unique())
    if len(months) < 13 or n_lab < 15 or (n_total - n_lab) < 15:
        print("      n不足でサイズ評価skip")
        return

    boot_full = Boot(months, nb=NB_MAIN, k=3, seed=20260717)
    D0 = boot_full.dd_median(base_s)
    cagr_base, m_base = boot_full.equal_dd_cagr(base_s, D0)

    print(f"      同DD({D0:.2f}%)でのCAGR比較 (baseline m=1.0 のCAGR={cagr_base:+.1f}%):")
    results_m = {}
    for m in M_LIST:
        s_m = sized_series(R, ti, flag, m)
        cagr_m, scale_m = boot_full.equal_dd_cagr(s_m, D0)
        results_m[m] = cagr_m
        tag = "  <- 現行" if m == 1.0 else ("  <- 逆ダミー" if m == 1.25 else "")
        print(f"        m={m:.2f}: CAGR={cagr_m:+.1f}%  (差={cagr_m-cagr_base:+.1f}pt){tag}")

    weak_m = 0.5   # nullとの比較は凍結スペックどおり m=0.5 固定（argmin選択は事後選択になるため廃止）
    print(f"      同数ランダムラベルnull({NREP_NULL}回, m={weak_m}での比較):")
    real_gain = results_m[weak_m] - cagr_base
    null_gains = []
    for rep in range(NREP_NULL):
        rflag = np.zeros(n_total, dtype=bool)
        idx = RNG.choice(n_total, size=n_lab, replace=False)
        rflag[idx] = True
        s_rand = sized_series(R, ti, rflag, weak_m)
        cagr_rand, _ = boot_full.equal_dd_cagr(s_rand, D0)
        null_gains.append(cagr_rand - cagr_base)
    null_gains = np.array(null_gains)
    pct = 100 * np.mean(null_gains < real_gain)
    print(f"        実測差={real_gain:+.1f}pt  null帯=[{np.percentile(null_gains,2.5):+.1f},"
          f"{np.percentile(null_gains,97.5):+.1f}]pt  -> {pct:.0f}%ile")

    print(f"      巡回ブロック・ブートストラップ P(m={weak_m}がbaselineを上回る、CAGR/DD基準):")
    s_weak = sized_series(R, ti, flag, weak_m)
    for k in (1, 3, 6, 12):
        p = block_boot_beat(base_s, s_weak, months, k, nb=300)
        print(f"        k={k:>2}mo: P={p:.0f}%")

    if results_m[1.25] < cagr_base:
        print(f"      逆ダミー(m=1.25)確認: CAGR={results_m[1.25]:+.1f}% < baseline{cagr_base:+.1f}% -> 機構整合(OK)")
    else:
        print(f"      逆ダミー(m=1.25)確認: CAGR={results_m[1.25]:+.1f}% >= baseline{cagr_base:+.1f}% -> ⚠️機構不整合")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print("#" * 110)
    print("btc15m_L")
    print("#" * 110)
    d_b, t_b, ii_b, R_b, ti_b = build_btc15m_L()
    if args.smoke:
        mask = ti_b >= pd.Timestamp("2024-01-01", tz=ti_b.tz)
        d_b_s, t_b_s, ii_b_s, R_b_s, ti_b_s = d_b, t_b[mask].reset_index(drop=True), ii_b[mask], R_b[mask], ti_b[mask]
    else:
        t_b_s, ii_b_s, R_b_s, ti_b_s = t_b, ii_b, R_b, ti_b
    for X in X_LIST:
        labelA, labelB = compute_labels(d_b, t_b_s, ii_b_s, X)
        evaluate_label("btc15m_L", "A(狩り)", R_b_s, ti_b_s, labelA, X)
        evaluate_label("btc15m_L", "B(FVG)", R_b_s, ti_b_s, labelB, X)
        evaluate_label("btc15m_L", "A∧B", R_b_s, ti_b_s, labelA & labelB, X)

    print("\n" + "#" * 110)
    print("gold15m")
    print("#" * 110)
    d_g, t_g, ii_g, R_g, ti_g = build_gold15m()
    if args.smoke:
        mask = ti_g >= pd.Timestamp("2024-01-01", tz=ti_g.tz)
        t_g_s, ii_g_s, R_g_s, ti_g_s = t_g[mask].reset_index(drop=True), ii_g[mask], R_g[mask], ti_g[mask]
    else:
        t_g_s, ii_g_s, R_g_s, ti_g_s = t_g, ii_g, R_g, ti_g
    for X in X_LIST:
        labelA, labelB = compute_labels(d_g, t_g_s, ii_g_s, X)
        evaluate_label("gold15m", "A(狩り)", R_g_s, ti_g_s, labelA, X)
        evaluate_label("gold15m", "B(FVG)", R_g_s, ti_g_s, labelB, X)
        evaluate_label("gold15m", "A∧B", R_g_s, ti_g_s, labelA & labelB, X)


if __name__ == "__main__":
    main()
