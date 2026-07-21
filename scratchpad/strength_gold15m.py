"""仕様カード7 scratchpad/spec_strength_gold15m.md の実装。

問い: btc15m_L で生き残った合成強度旗 combo2=(rank_pct(stop_atr)+rank_pct(atr_pctile))/2
(トップ20%が PF2.35/+0.94R vs 残りPF1.36/+0.33R・ブロックCI全0超) が、同型レッグ gold15m
(15m・押し目指値・ロング・確定足breakout — btc15m_L と同じ型)でも効くか。他BTCレッグ
(btc15m_S/btc_bo_kama/btc_pull、仕様カード6=strength_transplant_btc.py)へは移植しなかった
＝型が違った(ショート/4H成行/別エンジン)。gold15m は型が同じなので見込みが相対的に高い。

土台（再発明禁止・import流用）:
  scratchpad/strength_gateslope_generalize.py -- build_gold15m()（照合ゲート1-3 済みの
                                          gold15m 構築経路。netR定義もbook.pyと一致確認済み）・
                                          compute_sma_slope（日足SMA150傾き, combo3用）
  scratchpad/strength_btc15mL.py       -- rebuild_entries/match_entries_to_trades/
                                          quintile_table/monotone_flag/block_bootstrap_spearman/
                                          random_drop_null/report_candidate
  scratchpad/strength_entryquality_btc15mL.py -- atr_percentile_at（trailing500分位）
  scratchpad/strength_transplant_btc.py -- build_combo/topgap_table/block_bootstrap_topgap/
                                          report_topgap_bootstrap/era_topgap/spearman_report/
                                          pf_stat/print_baseline（合成旗の測り方一式）

対象: gold15m 1本。research/book.py get_book_legs() L94-98 と厳密一致（gg.build_gold15m()
が既にこの経路で照合ゲート1-3 PASS 済み）。R は gold15m の book 定義
(netR = t.R - 0.3/t.risk)。n≈325 (2019-05〜2026-05, 約44本/年)。ATRは g15=15m gold。

強度候補（確定足 i で・no-lookahead）:
  stop_atr   = t.risk / ATR(14)[i]           (ATRは15m gold)
  atr_pctile = ATR(14)[i] の trailing 500本percentile (同TF)
  combo2     = (rank_pct(stop_atr) + rank_pct(atr_pctile)) / 2
  combo3(参考) = 上記 + 日足SMA150傾き rank
    (gateslope_generalize.compute_sma_slope: 前日確定+ffill、1本差分の変化率。
     gold15mのゲート自体はSMA150上向き+10本前より高いだが、強度候補はそのslope自体で
     測る — strength_gateslope_generalize.py のカード3所見(単体フラット)を、合成での
     寄与として参考的に見る)。

測り方（btc15m_L/strength_transplant_btc.py と同一）:
  5分位表 n/win%/PF/meanR/totR + 単調性 + Spearman(combo2 vs R, ブロックbootstrap CI)。
  トップ20% vs 残り(n=325なのでトップ65)の meanRギャップ + 巡回ブロックbootstrap
  (1/3/6/12mo, 3000回) 95%CI + 年別 top-rest。combo3も同様に併記。

Run:
  .venv/bin/python scratchpad/strength_gold15m.py --smoke 2>&1 | \\
      tee scratchpad/out_strength_gold15m_smoke.txt
  .venv/bin/python scratchpad/strength_gold15m.py 2>&1 | \\
      tee scratchpad/out_strength_gold15m.txt
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import strength_btc15mL as base                  # rebuild_entries/match_entries_to_trades/
                                                   # quintile_table/monotone_flag/
                                                   # block_bootstrap_spearman/random_drop_null/
                                                   # report_candidate
import strength_gateslope_generalize as gg        # build_gold15m/gate1_check/gate2_check/
                                                   # compute_sma_slope
import strength_entryquality_btc15mL as eq        # atr_percentile_at
import strength_transplant_btc as tr              # build_combo/topgap_table/
                                                   # block_bootstrap_topgap/report_topgap_bootstrap/
                                                   # era_topgap/spearman_report/pf_stat/print_baseline


def run_gold15m(cli_smoke):
    print(f"\n{'#'*78}\n# gold15m -- 合成強度旗(combo2=stop_atr+atr_pctile)の移植検定 "
          f"(同型レッグ: 15m・押し目指値・ロング・breakout, n≈325, トップ20% vs 残り)\n{'#'*78}")
    g15, args, t, netR = gg.build_gold15m(cli_smoke)
    print(f"gold15m 再構築: n={len(t)}  span={t['time'].iloc[0]} -> {t['time'].iloc[-1]}"
          f"  (smoke={cli_smoke})")

    mine = pd.Series(netR, index=pd.DatetimeIndex(t["time"]))
    gate1 = gg.gate1_check("gold15m", mine, cli_smoke)
    if gate1 is False:
        print("!!! gold15m 照合ゲート1 FAIL -- 以降の数字は信用しないこと。ここで停止する。")
        return None
    if gate1 is None:
        print("(smoke run: ゲート1はフルデータでのみ判定される)")

    entries, t2 = base.rebuild_entries(g15, args)
    gate2 = gg.gate2_check("gold15m", t2, t)
    if not gate2:
        print("!!! gold15m 照合ゲート2 FAIL -- entries復元を信用できない。ここで停止する。")
        return None

    i_arr = base.match_entries_to_trades(entries, t, args.pullback_frac)
    print(f"[gold15m 照合ゲート3] entries<->trades 対応付け: {len(i_arr)}/{len(t)} 本すべて一意対応 => PASS")

    times = t["time"].values
    R = netR  # gold15m の book 定義の netR で測る(仕様カード7指定)
    span_years = max((pd.DatetimeIndex(times).max() - pd.DatetimeIndex(times).min()).days / 365.25, 0.1)
    tr.print_baseline("gold15m baseline", R, span_years)

    # ATR: g15(15m gold), rebuild_entries内部の 'a' と同一定義(BASE.atr=14)
    atr_g15 = ta.atr(g15["high"], g15["low"], g15["close"], length=args.atr).values

    stop_atr = t["risk"].values / atr_g15[i_arr]
    atr_pctile = eq.atr_percentile_at(atr_g15, i_arr, window=500)

    # combo3参考軸: 日足SMA150傾き(前日確定+ffill、1本差分)。strength_gateslope_generalize と同じ関数。
    sma_slope_arr = gg.compute_sma_slope(g15, sma_n=args.daily_sma, tf="1D")
    sma_slope = sma_slope_arr[i_arr]

    df = pd.DataFrame({"R": R, "t": times, "sa": stop_atr, "ap": atr_pctile, "ss": sma_slope})
    n_nan = df[["sa", "ap"]].isna().any(axis=1).sum()
    df2 = df.dropna(subset=["sa", "ap"]).reset_index(drop=True)
    print(f"[stop_atr/atr_pctile] window不足等でNaN除外: {n_nan}/{len(df)}  有効n={len(df2)}")

    combo2, _ = tr.build_combo(df2["sa"].values, df2["ap"].values)
    df2["combo2"] = combo2

    print("\n  === 5分位表・単調性・Spearman・ブロックbootstrap・random-drop null (combo2) ===")
    base.report_candidate("combo2 (stop_atr+atr_pctile, gold15m, n=325目安の5分位)",
                           df2["combo2"].values, df2["R"].values, df2["t"].values)

    top_mask = (df2["combo2"] >= df2["combo2"].quantile(0.8)).values
    print(f"\n  === 判定本体: トップ20% (combo2>=p80, n={top_mask.sum()}) vs 残り (n={(~top_mask).sum()}) ===")
    tr.topgap_table(df2["R"].values, top_mask)
    tr.report_topgap_bootstrap(df2["t"].values, top_mask, df2["R"].values)
    tr.era_topgap(df2["t"].values, top_mask, df2["R"].values)

    # 分解: stop_atr単体 / atr_pctile単体でもトップ20%ギャップを見る(combo2が崩れた場合にどちらが
    # 病巣かを機構的に説明するため)
    for colname, label in [("sa", "stop_atr単体"), ("ap", "atr_pctile単体")]:
        tm = (df2[colname] >= df2[colname].quantile(0.8)).values
        print(f"\n  --- 分解: {label} のみでトップ20% (n={tm.sum()}) vs 残り (n={(~tm).sum()}) ---")
        tr.topgap_table(df2["R"].values, tm)

    # combo3(参考)
    df3 = df.dropna(subset=["sa", "ap", "ss"]).reset_index(drop=True)
    print(f"\n[combo3参考軸: 日足SMA150傾き] NaN(SMA150ウォームアップ等)除外後 有効n={len(df3)}/{len(df)}")
    combo2_3, combo3 = tr.build_combo(df3["sa"].values, df3["ap"].values, extra=df3["ss"].values)
    tr.spearman_report("combo3 (stop_atr+atr_pctile+日足SMA150傾き, 参考)", combo3, df3["R"].values,
                        df3["t"].values)
    top_mask3 = (pd.Series(combo3) >= pd.Series(combo3).quantile(0.8)).values
    print(f"\n  --- combo3参考: トップ20% (n={top_mask3.sum()}) vs 残り (n={(~top_mask3).sum()}) ---")
    tr.topgap_table(df3["R"].values, top_mask3)
    tr.report_topgap_bootstrap(df3["t"].values, top_mask3, df3["R"].values)

    return dict(leg="gold15m", n=len(df2), top_mask=top_mask, R=df2["R"].values, t=df2["t"].values,
                combo=df2["combo2"].values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    run_gold15m(cli.smoke)

    print(f"\n実行コマンド: .venv/bin/python scratchpad/strength_gold15m.py"
          f"{' --smoke' if cli.smoke else ''}")


if __name__ == "__main__":
    main()
