"""仕様カード experiments/spec_gold15m_fast_accel.md の実装。

問い: btc15m_L で効いた kama_slope(4H KAMA(14) の傾きの急さ) は、gold15m に「自分のゲート=日足SMA150
(鈍い)」の傾きを当てると消えた(カード3=strength_gateslope_generalize.py)。病巣候補=指標の速さ。
今回は btc15m_L の勝ち信号(4H KAMA(14)の傾き)を銘柄だけ替えて gold15m にそのまま移植し、
「4Hトレンド加速というプリミティブが銘柄横断か、btc15m_L 脚固有か」を切り分ける。

土台(再発明禁止):
  - gold15m のレッグ再構築(entries直呼び＋トレード<->確定足i対応、照合ゲート1/2/3) は
    experiments/strength_gateslope_generalize.py の build_gold15m/gate1_check/gate2_check/
    era_report をそのまま import して使う(book.get_book_legs()['gold15m'] と一致済み)。
  - quintile_table/monotone_flag/block_bootstrap_spearman/random_drop_null/report_candidate は
    experiments/strength_btc15mL.py の汎用実装をそのまま import。
  - h4_kama_slope は experiments/strength_regime_btc15mL.py の compute_kama_slope を
    g15(gold 15min フレーム) に対して n=14, tf="240min" で呼ぶだけ(btc15m_L と一字一句同じ関数・
    同じ shift(1)+ffill 規約 -- 完全移植)。

強度候補3つ(単体・確定足i・no-lookahead):
  1. h4_kama_slope(本命): gold 4H(240min) KAMA(14) の1本あたり傾き。btc15m_L の勝ち信号を銘柄だけ
     替えた完全移植。reg.compute_kama_slope(g15, n=14, tf="240min")。
  2. atr_expansion: 確定足iでの ATR(14)[i]/ATR(14)[i-20](15m足ATR、過去のみ)。ボラ拡大=加速の代理。
  3. h1_ema_slope: gold 1H(60min) EMA(20) の1本あたり傾き。日足SMA150(鈍)と4H(速)の中間速度。

Run:
  .venv/bin/python experiments/strength_gold15m_fastaccel.py --smoke 2>&1 | tee experiments/out_strength_gold15m_fastaccel_smoke.txt
  .venv/bin/python experiments/strength_gold15m_fastaccel.py 2>&1 | tee experiments/out_strength_gold15m_fastaccel.txt
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

import strength_btc15mL as base                    # quintile_table/monotone_flag/
                                                     # block_bootstrap_spearman/random_drop_null/
                                                     # report_candidate
import strength_regime_btc15mL as reg               # compute_kama_slope (h4_kama_slope の本体)
import strength_gateslope_generalize as gsg         # build_gold15m/gate1_check/gate2_check/era_report


# ---------------------------------------------------------------- 強度候補2/3

def compute_atr_expansion(g15, atr_n=14, lookback=20):
    """候補2: 確定足iでの ATR(14)[i]/ATR(14)[i-20]。15m足ATR、過去のみ参照でno-lookahead
    (ATR[i]はbar iのH/L/Cまでで確定するのでbar i確定時点で既知)。"""
    a = ta.atr(g15["high"], g15["low"], g15["close"], length=atr_n).values.astype(float)
    ratio = np.full(len(a), np.nan)
    ratio[lookback:] = a[lookback:] / a[:-lookback]
    return ratio


def compute_ema_slope(g15, span=20, tf="60min"):
    """候補3: 1H(60min) EMA(20) の1本あたり変化率。gate_kama/compute_kama_slope と全く同じ
    shift(1)+ffill 規約(HTF確定後に使用)。"""
    dc = g15["close"].resample(tf).last().dropna()
    ema = dc.ewm(span=span, adjust=False).mean()
    slope_pct = (ema - ema.shift(1)) / ema.shift(1)
    slope_shifted = slope_pct.shift(1)
    arr = slope_shifted.reindex(g15.index, method="ffill").values
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    print(f"\n{'#'*78}\n# gold15m -- 強度候補: h4_kama_slope(本命・btc15m_L移植) / "
          f"atr_expansion / h1_ema_slope\n{'#'*78}")

    g15, args, t, netR = gsg.build_gold15m(cli.smoke)
    print(f"gold15m 再構築: n={len(t)}  span={t['time'].iloc[0]} -> {t['time'].iloc[-1]}  "
          f"(smoke={cli.smoke})")

    # ---- 照合ゲート1: 自作netR vs book.get_book_legs()['gold15m'] ----
    mine = pd.Series(netR, index=pd.DatetimeIndex(t["time"]))
    gate1 = gsg.gate1_check("gold15m", mine, cli.smoke)
    if gate1 is False:
        print("!!! gold15m 照合ゲート1 FAIL -- 以降の数字は信用しないこと。停止する。")
        return

    # ---- 照合ゲート2: entries直呼び再構築 t2 vs run()の生トレード表 ----
    entries, t2 = base.rebuild_entries(g15, args)
    gate2 = gsg.gate2_check("gold15m", t2, t)
    if not gate2:
        print("!!! gold15m 照合ゲート2 FAIL -- entries復元を信用できない。停止する。")
        return

    # ---- 照合ゲート3: entries<->trades 対応付け ----
    i_arr = base.match_entries_to_trades(entries, t, args.pullback_frac)
    print(f"[gold15m 照合ゲート3] entries<->trades 対応付け: {len(i_arr)}/{len(t)} 本すべて一意対応 => PASS")

    R = t["R"].values
    times = t["time"].values

    # ================================================================ 候補1: h4_kama_slope (本命)
    h4_arr, h4_median = reg.compute_kama_slope(g15, n=14, tf="240min")
    h4 = h4_arr[i_arr]
    n_nan1 = int(np.isnan(h4).sum())
    n_neg1 = int((h4[~np.isnan(h4)] < 0).sum())
    print(f"\n[h4_kama_slope 自己点検] トレード{len(h4)}本中、4H KAMA(14)のウォームアップ等でNaN={n_nan1}本、"
          f"負(下降)の本数={n_neg1}/{len(h4)-n_nan1}  (gold15m のゲートは日足SMA150であり4H KAMAの向きは"
          f"要求しない。ここは btc15m_L の勝ち信号を銘柄だけ替えた完全移植の強度候補であり、gold15mの"
          f"エントリー自体に4H KAMA上向き制約は無いので負の値が相当数出ても矛盾ではない。"
          f"240min粒度の全履歴中央値={h4_median:+.6f})")
    mask1 = ~np.isnan(h4)
    rows1, rho1 = base.report_candidate(
        "h4_kama_slope (= gold 4H KAMA(14) の1本あたり変化率, btc15m_L の勝ち信号を銘柄だけ移植)",
        h4[mask1], R[mask1], times[mask1], tag=f"[有効n={mask1.sum()}, NaN除外={n_nan1}]")
    gsg.era_report(h4[mask1], R[mask1], times[mask1])

    # ================================================================ 候補2: atr_expansion
    atrexp_arr = compute_atr_expansion(g15, atr_n=14, lookback=20)
    atrexp = atrexp_arr[i_arr]
    n_nan2 = int(np.isnan(atrexp).sum())
    print(f"\n[atr_expansion 自己点検] トレード{len(atrexp)}本中、ATR(14)ウォームアップ/lookback20不足で"
          f"NaN={n_nan2}本  中央値(NaN除く)={np.nanmedian(atrexp):+.4f}")
    mask2 = ~np.isnan(atrexp)
    rows2, rho2 = base.report_candidate(
        "atr_expansion (= ATR(14)[i]/ATR(14)[i-20], 15m足, 過去のみ)",
        atrexp[mask2], R[mask2], times[mask2], tag=f"[有効n={mask2.sum()}, NaN除外={n_nan2}]")
    gsg.era_report(atrexp[mask2], R[mask2], times[mask2])

    # ================================================================ 候補3: h1_ema_slope
    h1_arr = compute_ema_slope(g15, span=20, tf="60min")
    h1 = h1_arr[i_arr]
    n_nan3 = int(np.isnan(h1).sum())
    n_neg3 = int((h1[~np.isnan(h1)] < 0).sum())
    print(f"\n[h1_ema_slope 自己点検] トレード{len(h1)}本中、1H EMA(20)のウォームアップ等でNaN={n_nan3}本、"
          f"負(下降)の本数={n_neg3}/{len(h1)-n_nan3}")
    mask3 = ~np.isnan(h1)
    rows3, rho3 = base.report_candidate(
        "h1_ema_slope (= gold 1H EMA(20) の1本あたり変化率, 日足SMA150と4Hの中間速度)",
        h1[mask3], R[mask3], times[mask3], tag=f"[有効n={mask3.sum()}, NaN除外={n_nan3}]")
    gsg.era_report(h1[mask3], R[mask3], times[mask3])

    # ================================================================ 判定サマリ
    print(f"\n{'='*78}\n判定サマリ\n{'='*78}")
    print(f"  h4_kama_slope: Spearman={rho1:+.4f}  (btc15m_L と同じ向き/プラスなら横断プリミティブ候補、"
          f"フラット/負なら脚固有)")
    print(f"  atr_expansion: Spearman={rho2:+.4f}")
    print(f"  h1_ema_slope : Spearman={rho3:+.4f}")

    print(f"\n実行コマンド: .venv/bin/python experiments/strength_gold15m_fastaccel.py"
          f"{' --smoke' if cli.smoke else ''}")


if __name__ == "__main__":
    main()
