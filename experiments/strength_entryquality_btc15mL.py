"""仕様カード experiments/spec_entryquality_btc15mL.md の実装（パートA + パートB）。

btc15m_L 1本を experiments/strength_btc15mL.py の build()/rebuild_entries()/
match_entries_to_trades()（照合ゲート1-3 済み・再発明禁止で import）で再構築し、

パートA「エントリー質＝実効リスクの小ささ」の3強度候補を測る:
  1. stop_atr      = tL.risk / ATR(14)[i]            -- 損切りがボラ対比でタイトか
  2. support_dist   = (e_px - rolling_min(low, K=48)[i]) / ATR(14)[i]
                      -- 直近スイング安値からの距離(ATR単位)。小さいほど支えの直上で入った
  3. atr_pctile     = ATR(14)[i] の trailing 500本 percentile
                      -- 低いほど静かな相場

すべて確定足 i（ブレイク確定足、entries経由で復元・照合ゲート3で対応付け済み）で評価。
先読みなし: rolling min / rolling percentile は i を含む過去窓のみを使う
(i時点で計算済みの量を使うのは、戦略本体の detect/plan がすでに a[i] をこの流儀で
使っているのと同じ扱い)。ATR(14)は rebuild_entries() 内部の 'a' と同一の呼び出しで
d15 から作り直す（決定的関数なので土台のtL/entriesとの整合は自動的に保たれる。土台
自体は変更しない）。

パートB「stop$ 可否フィルタ」(サイジングでなく take/skip、ユーザーの実制約に直結):
  BTC 0.01ロットの契約仕様(1ロット=何BTCか)は未確認。仕様カードの推奨前提「Vantage BTC CFD
  1ロット=1BTC」を採用し、stop$ = 0.01 x tL.risk（price距離[USD]をそのまま口座通貨とみなす、
  BTC/USD建て契約の標準的な想定）とする。前提が外れても risk_price(=tL.risk)自体の分位で
  X を決めているため、掃引の相対比較(take/skipの効果)への影響は無い（絶対値スケールのみ動く）。

R は仕様カード5の指示どおり 素netR = tL.R - 15/tL.risk（PDH重み(WL)は別軸なので外す。
build() が返す netR は WL 込みなので、ここでは tL の列から独立に計算し直す。ただし照合
ゲート1自体は土台と同じ「WL込みnetR」でbook.get_book_legs()と突き合わせる — 仕様カードの
netR定義とは別に、土台の tie-back 照合はWL込みが正しい対象なので、そこだけ使い分ける）。

Run:
  .venv/bin/python experiments/strength_entryquality_btc15mL.py --smoke 2>&1 | \\
      tee experiments/out_entryquality_btc15mL_smoke.txt
  .venv/bin/python experiments/strength_entryquality_btc15mL.py 2>&1 | \\
      tee experiments/out_entryquality_btc15mL.txt
"""
import argparse
import contextlib
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.strength_btc15mL import (
    build, rebuild_entries, match_entries_to_trades,
    quintile_table, monotone_flag, block_bootstrap_spearman,
    random_drop_null, report_candidate,
)
from research.portfolio_alloc import cagr_dd_trades   # 既存関数を流用（車輪の再発明禁止）

RISK_FIXED = 0.01   # パートB掃引の固定ベット。全行(baseline含む)で同じ値＝サイズ再最適化なし


# ---------------------------------------------------------------- パートA candidate計算

def support_dist_at(low_arr, e_px, i_arr, K=48):
    """(e_px - 過去K本の最安値[i]) / ATR(14)[i] の分子部分。ATR除算は呼び出し側。
    rolling min は i を含む過去K本 [i-K+1, i] のみ使用 -- 先読みなし。"""
    low_s = pd.Series(low_arr)
    roll_min = low_s.rolling(K, min_periods=K).min().values
    swing_low = roll_min[i_arr]
    return e_px - swing_low, swing_low


def atr_percentile_at(atr_arr, i_arr, window=500):
    """ATR(14)[i] の trailing window本percentile。窓は [i-window+1, i] のみ(先読みなし)。"""
    atr_arr = np.asarray(atr_arr, dtype=float)
    out = np.full(len(i_arr), np.nan)
    for j, i in enumerate(i_arr):
        if i < window - 1 or np.isnan(atr_arr[i - window + 1:i + 1]).any():
            continue
        w = atr_arr[i - window + 1:i + 1]
        out[j] = 100.0 * (w <= atr_arr[i]).mean()
    return out


def era_beta_check(times, x, R, label):
    """年別 Q5-Q1 meanR（時代ベータ隔離）。年ごとに"その年のデータだけで"5分位を作り、
    年内Q5-Q1を出す(グローバル分位だと年によってQ1/Q5が空になりやすいため、年内再分位で
    「そのeraの中でもこの軸は並べ替えているか」を直接見る)。"""
    s = pd.DataFrame({"x": x, "R": R}, index=pd.DatetimeIndex(times))
    s["year"] = s.index.year
    print(f"\n  [時代ベータ隔離] 年別 Q5-Q1 meanR ({label}):")
    rows = []
    for yr, grp in s.groupby("year"):
        grp = grp.dropna()
        if len(grp) < 10:
            print(f"    {yr}: n={len(grp)} (<10, 分位化スキップ)")
            continue
        ranks = grp["x"].rank(method="first")
        q = pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5])
        q1 = grp.loc[q == 1, "R"]
        q5 = grp.loc[q == 5, "R"]
        diff = q5.mean() - q1.mean()
        rows.append((yr, len(grp), q1.mean(), q5.mean(), diff))
        print(f"    {yr}: n={len(grp):4d}  Q1meanR={q1.mean():+.3f}  Q5meanR={q5.mean():+.3f}"
              f"  Q5-Q1={diff:+.3f}")
    if rows:
        diffs = np.array([r[4] for r in rows])
        print(f"    年別 Q5-Q1 の符号: {sum(diffs > 0)}勝{sum(diffs <= 0)}敗 / {len(diffs)}年"
              f"  平均={diffs.mean():+.3f}  中央値={np.median(diffs):+.3f}  標準偏差={diffs.std():.3f}")
    return rows


# ---------------------------------------------------------------- パートB: stop$ 可否フィルタ

def part_b(tL, plain_netR, times, span_years):
    print(f"\n{'='*78}\nパートB: stop$ 可否フィルタ（サイジングでなく take/skip）\n{'='*78}")
    print("  前提: Vantage BTC CFD の1ロット=1BTC と仮定（契約仕様は未確認のため仕様カードの推奨前提"
          "を採用）。\n"
          "        0.01ロットの stop$ = 0.01 x tL.risk（price距離[USD]をそのまま使用）。\n"
          "        前提が外れても risk_price(=tL.risk) 自体の分位でXを決めているため、"
          "掃引の相対比較\n"
          "        （take/skipの効果）は前提に依存しない。ずれるのはstop$の絶対値スケールのみ。")

    risk_price = tL["risk"].values
    stopusd = 0.01 * risk_price
    years = pd.DatetimeIndex(times).year

    s = pd.Series(stopusd)
    p50, p70, p90, p95, mx = (s.quantile(.5), s.quantile(.7), s.quantile(.9),
                               s.quantile(.95), s.max())
    print(f"\n  stop$ 分布 (0.01ロット, n={len(s)}):")
    print(f"    中央値(p50)=${p50:.3f}  p70=${p70:.3f}  p90=${p90:.3f}  p95=${p95:.3f}  "
          f"max=${mx:.3f}  mean=${s.mean():.3f}")
    print("  年別 stop$ 中央値:")
    for yr, g in pd.DataFrame({"stopusd": stopusd, "year": years}).groupby("year"):
        print(f"    {yr}: n={len(g):>3}  中央値=${g['stopusd'].median():.3f}")

    def row(mask, label, X):
        n = int(mask.sum())
        if n == 0:
            print(f"  {label:<20} X=${X:>8.3f}  n=0 -- 空集合")
            return
        Rm = plain_netR[mask]
        win = 100.0 * (Rm > 0).mean()
        pos = Rm[Rm > 0].sum(); neg = abs(Rm[Rm <= 0].sum())
        pf = pos / neg if neg > 0 else np.inf
        pf_s = f"{pf:.2f}" if np.isfinite(pf) else "inf"
        ser = pd.Series(Rm * RISK_FIXED, index=pd.DatetimeIndex(times)[mask]).sort_index()
        cagr, dd, cdd = cagr_dd_trades(ser)
        print(f"  {label:<20} X=${X:>8.3f}  n={n:>4} ({n/span_years:5.1f}/yr)  win={win:5.1f}%  "
              f"PF={pf_s:>5}  meanR={Rm.mean():+.3f}  totR={Rm.sum():+7.1f}  "
              f"CAGR={cagr:+6.1f}%  maxDD={dd:5.1f}%  CAGR/DD={cdd:5.2f}")

    print(f"\n  掃引（固定ベット risk={RISK_FIXED*100:.0f}%/trade、全行同一＝サイズ再最適化なし。"
          f"span={span_years:.2f}yr、R=素netR）:")
    row(np.full(len(plain_netR), True), "baseline(全採用)", mx)
    for label, X in [("stop$<=p50", p50), ("stop$<=p70", p70), ("stop$<=p90", p90)]:
        mask = stopusd <= X
        row(mask, label, X)

    print(f"\n  参考: 除外された「遠stop」側のトレード自身のEV（可否 vs フィルタ利得を直接見る）:")
    for label, X in [("stop$>p50", p50), ("stop$>p70", p70), ("stop$>p90", p90)]:
        mask = stopusd > X
        row(mask, label, X)

    print("\n  読み方: baseline比で meanR/PF/CAGR-DDが上回れば「遠stopが足を引っ張っている="
          "フィルタとして有効」。\n"
          "  下回れば「遠stopトレード自体もEVを持っている＝この制約は"
          "取れないから諦める(可否)であって\n"
          "  フィルタの利得ではない」と読む（指値化/自動化での復活余地は別軸、法則6参照）。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    d15, raw, args, tL, _ = build(cli.smoke)
    print(f"btc15m_L 再構築: n={len(tL)}  span={tL['time'].iloc[0]} -> {tL['time'].iloc[-1]}"
          f"  (smoke={cli.smoke})")

    # ---- 照合ゲート1: build() の netR(PDH重み込み) vs research.book.get_book_legs() ----
    if cli.smoke:
        print("\n[照合ゲート1] --smoke のためスキップ (get_book_legs()はフルデータ前提)")
    else:
        import research.book as book_mod
        with contextlib.redirect_stderr(io.StringIO()):
            legs = book_mod.get_book_legs()
        ref = legs["btc15m_L"]
        from src.engine.size import pdh_soft
        WL, _ = pdh_soft(d15, tL)
        netR_pdh = (tL["R"].values - 15.0 / tL["risk"].values) * WL
        mine = pd.Series(netR_pdh, index=pd.DatetimeIndex(tL["time"]))
        same_len = len(ref) == len(mine)
        same_idx = same_len and ref.index.equals(mine.index)
        same_val = same_idx and np.allclose(ref.values, mine.values, rtol=0, atol=1e-12)
        gate1 = same_len and same_idx and same_val
        print(f"\n[照合ゲート1] netR(PDH込) vs book.get_book_legs()['btc15m_L']: "
              f"len {len(ref)}=={len(mine)} -> {same_len} | idx一致 -> {same_idx} | "
              f"値一致(atol=1e-12) -> {same_val}  => {'PASS' if gate1 else 'FAIL'}")
        if not gate1:
            print("!!! 照合ゲート1 FAIL -- 以降の数字は信用しないこと。ここで停止する。")
            return

    # ---- entries 復元 + 照合ゲート2 (t2 が tL と bit一致するか) ----
    entries, t2 = rebuild_entries(d15, args)
    same_n = len(t2) == len(tL)
    cols = ["time", "R", "hold", "risk", "e_px", "r_mkt", "filled", "base_bars"]
    same_vals = same_n and all(
        (np.allclose(t2[c].values.astype(float), tL[c].values.astype(float),
                      rtol=0, atol=1e-9) if c != "time" else
         (t2[c].values == tL[c].values).all())
        for c in cols
    )
    gate2 = same_n and same_vals
    print(f"\n[照合ゲート2] entries直呼び再構築 t2 vs run()の tL: n {len(t2)}=={len(tL)} -> {same_n} | "
          f"列一致({cols}) -> {same_vals}  => {'PASS' if gate2 else 'FAIL'}")
    if not gate2:
        print("!!! 照合ゲート2 FAIL -- i の復元(entries対応付け)を信用できない。ここで停止する。")
        return

    i_arr = match_entries_to_trades(entries, tL, args.pullback_frac)
    print(f"[照合ゲート3] entries<->trades 対応付け: {len(i_arr)}/{len(tL)} 本すべて一意対応 => PASS")

    # ---- 仕様カード5指定の R: 素netR = tL.R - 15/tL.risk (PDH重み外す) ----
    plain_netR = tL["R"].values - 15.0 / tL["risk"].values
    times = tL["time"].values
    e_px = tL["e_px"].values
    risk = tL["risk"].values
    span_years = max((pd.DatetimeIndex(times).max() - pd.DatetimeIndex(times).min()).days / 365.25, 0.5)
    print(f"\n[R定義] plain_netR = tL.R - 15/tL.risk (WL=PDHソフト重み対象外)。"
          f"mean={plain_netR.mean():+.4f}  n={len(plain_netR)}")

    # ATR(14)、entries復元と同じ手順で(args.atr=14, rebuild_entries内部と同一呼び出し)
    atr14 = ta.atr(d15["high"], d15["low"], d15["close"], length=args.atr).values
    print(f"[ATR] length={args.atr} (BASE.atr、rebuild_entries()内の 'a' と同一定義)")

    # ================================================================== パートA
    print(f"\n{'#'*78}\nパートA: エントリー質＝実効リスクの小ささ（強度候補3つ）\n{'#'*78}")

    # ---- 候補1: stop_atr = risk / ATR(14)[i] ----
    atr_at_i = atr14[i_arr]
    stop_atr = risk / atr_at_i
    rows1, rho1 = report_candidate(
        "stop_atr (= tL.risk / ATR(14)[i]、損切りのボラ対比タイトさ)",
        stop_atr, plain_netR, times)
    era_beta_check(times, stop_atr, plain_netR, "stop_atr")

    # ---- 候補2: support_dist = (e_px - swing_low[i]) / ATR(14)[i] ----
    num, swing_low = support_dist_at(d15["low"].values, e_px, i_arr, K=48)
    n_nan_sl = np.isnan(swing_low).sum()
    support_dist = num / atr_at_i
    print(f"\n[support_dist] K=48本rolling最安値、window不足でNaNの本数: {n_nan_sl}/{len(i_arr)}")
    mask2 = ~np.isnan(support_dist)
    rows2, rho2 = report_candidate(
        "support_dist (= (fill価格e_px - 過去48本安値[i]) / ATR(14)[i]、支えからの距離)",
        support_dist[mask2], plain_netR[mask2], times[mask2],
        tag=f"[有効n={mask2.sum()}, NaN除外={n_nan_sl}]")
    era_beta_check(times[mask2], support_dist[mask2], plain_netR[mask2], "support_dist")

    # ---- 候補3: atr_pctile = ATR(14)[i] の trailing 500本 percentile ----
    atr_pctile = atr_percentile_at(atr14, i_arr, window=500)
    n_nan_ap = np.isnan(atr_pctile).sum()
    print(f"\n[atr_pctile] trailing 500本percentile、window不足でNaNの本数: {n_nan_ap}/{len(i_arr)}")
    mask3 = ~np.isnan(atr_pctile)
    rows3, rho3 = report_candidate(
        "atr_pctile (= ATR(14)[i] の trailing 500本percentile、静かさの水準)",
        atr_pctile[mask3], plain_netR[mask3], times[mask3],
        tag=f"[有効n={mask3.sum()}, NaN除外={n_nan_ap}]")
    era_beta_check(times[mask3], atr_pctile[mask3], plain_netR[mask3], "atr_pctile")

    # ================================================================== パートB
    part_b(tL, plain_netR, times, span_years)

    print(f"\n実行コマンド: .venv/bin/python experiments/strength_entryquality_btc15mL.py"
          f"{' --smoke' if cli.smoke else ''}")


if __name__ == "__main__":
    main()
