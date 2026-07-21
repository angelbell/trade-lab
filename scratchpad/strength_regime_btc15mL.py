"""仕様カード2 scratchpad/spec_strength_regime_btc15mL.md の実装。

btc15m_L を scratchpad/strength_btc15mL.py の土台（entries直呼び再構築＋トレード<->確定足i
対応付け、照合ゲート3本）でそのまま再構築し、3つの「レジーム強度」候補を確定足iで評価する。
候補1・2は既存の quintile_table/monotone_flag/block_bootstrap_spearman/random_drop_null を
そのまま import して使う（車輪の再発明禁止）。候補3(mtf_agree)は0-3の順序尺度なので
専用の bucket_table を用意し、同じ統計関数群に食わせる。

候補の中身（確定足iで評価・no-lookahead、既存gateと同じ shift(1)+ffill 規約）:
  1. kama_slope: 4H(240min) KAMA(14) の1本あたり変化率 (kmg[p]-kmg[p-1])/kmg[p-1]。
     src.engine.gates.gate_kama の krise = (kmg>kmg.shift(1)).shift(1) と全く同じ足付け・
     shift/ffill 規約で「符号(bool)」の代わりに「値そのもの」を持ち回る。gate は上向きのみ
     通すので、この値は btc15m_L の全トレードでほぼ正（符号反転しているのは境界のズレのみ
     ―― 下の実行結果で最小値を明示して自己点検する）。価格正規化＝KAMA自身の前値で割る。
  2. daily_trend: (close_daily - SMA150_daily)/SMA150_daily を「前日までの確定日足」で。
     src.engine.gates.gate_sma の ext_arr と全く同じ式・shift(1)+ffill 規約（gate_sma自体は
     btc15m_L で daily_sma=0 のため未使用＝このレッグの学習に一切混ざっていない独立変数）。
  3. mtf_agree（0-3）: 以下3条件の真の数。
       (a) 4H KAMAの傾き(=候補1の値) が「全履歴の240min確定値の中央値」より急
           （spec: ゲート条件そのもの(上向き二値)は全trueで強度の足しにならないため、
           指示どおり中央値超えに置換）
       (b) 日足SMA150が上向き（sma > sma.shift(1), 前日までの確定日足・shift(1)+ffill）
       (c) 週足終値 > 30週MA（research/portfolio_kama.cycle_gate_pull と同じ "1W" 足付け・
           shift(1週)+ffill 規約）
     年別のバケット分布・ON率(b>=2)も併記し「合流=上昇相場ベータ」の疑いを検算する。

Run:
  .venv/bin/python scratchpad/strength_regime_btc15mL.py --smoke 2>&1 | tee scratchpad/out_strength_regime_btc15mL_smoke.txt
  .venv/bin/python scratchpad/strength_regime_btc15mL.py 2>&1 | tee scratchpad/out_strength_regime_btc15mL.txt
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
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import strength_btc15mL as base  # 土台: build/rebuild_entries/match_entries_to_trades +
                                  # quintile_table/monotone_flag/block_bootstrap_spearman/random_drop_null
from breakout_wave import kama_adaptive


# ---------------------------------------------------------------- 強度候補の計算
# 3つとも「shift(1) してから d15.index に ffill で再展開」という、
# src/engine/gates.py の gate_sma / gate_kama と全く同じ確定足規約に揃える。

def compute_kama_slope(d15, n=14, tf="240min"):
    """候補1: 4H KAMA(n) の1本あたり変化率。gate_kama と同じ足付け・shift(1)+ffill。
    戻り値: (d15.index に展開した配列, shift後・240min粒度でdropnaした中央値)"""
    dck = d15["close"].resample(tf).last().dropna()
    kmg = kama_adaptive(dck, n)
    slope_pct = (kmg - kmg.shift(1)) / kmg.shift(1)
    slope_shifted = slope_pct.shift(1)          # gate_kama の krise=(...).shift(1) と同じ1本ずらし
    arr = slope_shifted.reindex(d15.index, method="ffill").values
    median_240 = slope_shifted.dropna().median()  # 240min粒度(重複なし)の中央値
    return arr, median_240


def compute_daily_trend(d15, sma_n=150):
    """候補2: 前日までの確定日足で (close-SMA150)/SMA150。gate_sma の ext_arr と同じ式・規約。"""
    dc = d15["close"].resample("1D").last().dropna()
    sma = dc.rolling(sma_n).mean()
    ext = (dc - sma) / sma
    arr = ext.shift(1).reindex(d15.index, method="ffill").values
    return arr


def compute_daily_sma_up(d15, sma_n=150, k=1):
    """mtf_agree条件(b): 前日までの確定日足でSMA150が上向き。"""
    dc = d15["close"].resample("1D").last().dropna()
    sma = dc.rolling(sma_n).mean()
    up = (sma > sma.shift(k)).shift(1)
    return up.reindex(d15.index, method="ffill").fillna(False).values


def compute_weekly_trend_up(d15, w_n=30):
    """mtf_agree条件(c): 週足終値 > 30週MA。research/portfolio_kama.cycle_gate_pull と同じ
    "1W" 足付け・shift(1週)+ffill。"""
    dw = d15["close"].resample("1W").last().dropna()
    wma = dw.rolling(w_n).mean()
    up = (dw > wma).shift(1)
    return up.reindex(d15.index, method="ffill").fillna(False).values


# ---------------------------------------------------------------- mtf_agree専用バケット表

def bucket_table(x, R):
    df = pd.DataFrame({"x": np.asarray(x, dtype=float), "R": np.asarray(R, dtype=float)}).dropna()
    rows = []
    for b in (0, 1, 2, 3):
        sub = df.loc[df["x"] == b, "R"]
        n = len(sub)
        win = 100.0 * (sub > 0).mean() if n else np.nan
        pos = sub[sub > 0].sum()
        neg = abs(sub[sub <= 0].sum())
        pf = (pos / neg) if neg > 0 else (np.inf if pos > 0 else np.nan)
        rows.append(dict(q=b, n=n, win=win, pf=pf,
                          meanR=(sub.mean() if n else np.nan), totR=sub.sum()))
    return rows, df


def report_bucket_candidate(name, x, R, times):
    print(f"\n{'='*78}\n候補: {name}\n{'='*78}")
    rows, df = bucket_table(x, R)
    print(f"  {'bkt':>3}{'n':>6}{'win%':>8}{'PF':>8}{'meanR':>9}{'totR':>9}")
    for r in rows:
        if r["n"] == 0:
            print(f"  {r['q']:>3}{r['n']:>6}   (n=0)")
            continue
        pf_s = f"{r['pf']:.2f}" if np.isfinite(r["pf"]) else "inf"
        print(f"  {r['q']:>3}{r['n']:>6}{r['win']:>7.1f}%{pf_s:>8}{r['meanR']:>+9.3f}{r['totR']:>+9.1f}")

    present = [r for r in rows if r["n"] > 0]
    means = [r["meanR"] for r in present]
    nondecr = all(means[i] <= means[i + 1] + 1e-12 for i in range(len(means) - 1))
    up = means[-1] > means[0] if len(means) > 1 else False
    print(f"  meanR系列(bkt昇順, n>0のみ): {[round(m,3) for m in means]}")
    print(f"  単調非減少: {'YES' if nondecr else 'NO'}   最高bkt>最低bkt: {'YES' if up else 'NO'}")

    rho, p = spearmanr(df["x"], df["R"])
    print(f"  Spearman(x,R) = {rho:+.4f}  (p={p:.4g}, n={len(df)})")
    for k in (1, 3, 6, 12):
        med, lo, hi, nvalid = base.block_bootstrap_spearman(times, x, R, k, n_boot=1000)
        print(f"    循環ブロック({k}mo) bootstrap median rho={med:+.4f}  95%CI=[{lo:+.4f}, {hi:+.4f}]"
              f"  (有効draw={nvalid}/1000)")

    top = rows[3]  # bkt=3
    if top["n"] > 0:
        pct, null_mean, null_std = base.random_drop_null(df["R"].values, top["meanR"], top["n"])
        print(f"  ランダム除去null: bkt=3と同数(n={top['n']})をランダム抽出した meanR の分布 "
              f"(平均{null_mean:+.3f}±{null_std:.3f}) に対し 実測bkt=3 meanR={top['meanR']:+.3f} は "
              f"{pct:.1f}パーセンタイル")
    else:
        print("  ランダム除去null: bkt=3 の n=0 のためスキップ")
    return rows, rho


def year_distribution(bucket, times):
    yrs = pd.DatetimeIndex(times).year
    ct = pd.crosstab(yrs, bucket)
    on_rate = pd.Series(bucket, index=yrs).groupby(level=0).apply(lambda s: 100.0 * (s >= 2).mean())
    print("\n  [年別] mtf_agree バケット分布 (行=年, 列=bkt0..3):")
    print(ct.to_string().replace("\n", "\n  "))
    print("\n  [年別] ON率 (bkt>=2 の割合, %):")
    print(("  " + on_rate.round(1).to_string()).replace("\n", "\n  "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    d15, raw, args, tL, netR = base.build(cli.smoke)
    print(f"btc15m_L 再構築: n={len(tL)}  span={tL['time'].iloc[0]} -> {tL['time'].iloc[-1]}"
          f"  (smoke={cli.smoke})")

    # ---- 照合ゲート1 ----
    if cli.smoke:
        print("\n[照合ゲート1] --smoke のためスキップ (get_book_legs()はフルデータ前提)")
    else:
        import research.book as book_mod
        with contextlib.redirect_stderr(io.StringIO()):
            legs = book_mod.get_book_legs()
        ref = legs["btc15m_L"]
        mine = pd.Series(netR, index=pd.DatetimeIndex(tL["time"]))
        same_len = len(ref) == len(mine)
        same_idx = same_len and ref.index.equals(mine.index)
        same_val = same_idx and np.allclose(ref.values, mine.values, rtol=0, atol=1e-12)
        gate1 = same_len and same_idx and same_val
        print(f"\n[照合ゲート1] netR vs book.get_book_legs()['btc15m_L']: "
              f"len {len(ref)}=={len(mine)} -> {same_len} | idx一致 -> {same_idx} | "
              f"値一致(atol=1e-12) -> {same_val}  => {'PASS' if gate1 else 'FAIL'}")
        if not gate1:
            print("!!! 照合ゲート1 FAIL -- 以降の数字は信用しないこと。ここで停止する。")
            return

    # ---- entries復元 + 照合ゲート2 ----
    entries, t2 = base.rebuild_entries(d15, args)
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

    i_arr = base.match_entries_to_trades(entries, tL, args.pullback_frac)
    print(f"[照合ゲート3] entries<->trades 対応付け: {len(i_arr)}/{len(tL)} 本すべて一意対応 => PASS")

    R = tL["R"].values
    times = tL["time"].values

    # ---- 候補1: kama_slope ----
    kama_slope_arr, kama_median = compute_kama_slope(d15, n=args.gate_kama, tf=args.gate_kama_tf)
    ks = kama_slope_arr[i_arr]
    n_neg = int((ks < 0).sum())
    print(f"\n[kama_slope 自己点検] トレード{len(ks)}本中、値が負(=gate_kamaの上向き判定と矛盾しうる境界)"
          f"の本数: {n_neg}  最小値={np.nanmin(ks):+.6f}  (240min粒度の全履歴中央値={kama_median:+.6f})")
    base.report_candidate("kama_slope (4H KAMA(14) の1本あたり変化率, 価格正規化)", ks, R, times)

    # ---- 候補2: daily_trend ----
    daily_trend_arr = compute_daily_trend(d15, sma_n=150)
    dt = daily_trend_arr[i_arr]
    n_nan = int(np.isnan(dt).sum())
    print(f"\n[daily_trend] SMA150ウォームアップ等でNaNの本数: {n_nan}/{len(dt)}")
    mask = ~np.isnan(dt)
    base.report_candidate("daily_trend (= (close_daily-SMA150)/SMA150, 前日確定)",
                           dt[mask], R[mask], times[mask],
                           tag=f"[有効n={mask.sum()}, NaN除外={n_nan}]")

    # ---- 候補3: mtf_agree (0-3) ----
    cond_a_arr = (kama_slope_arr > kama_median)
    cond_b_arr = compute_daily_sma_up(d15, sma_n=150, k=1)
    cond_c_arr = compute_weekly_trend_up(d15, w_n=30)
    mtf_arr = cond_a_arr.astype(int) + cond_b_arr.astype(int) + cond_c_arr.astype(int)
    mtf = mtf_arr[i_arr]
    a_at, b_at, c_at = cond_a_arr[i_arr], cond_b_arr[i_arr], cond_c_arr[i_arr]
    print(f"\n[mtf_agree 内訳] ON率: (a)KAMA急={100*a_at.mean():.1f}%  "
          f"(b)日足SMA150↑={100*b_at.mean():.1f}%  (c)週足>30wMA={100*c_at.mean():.1f}%")
    rows3, rho3 = report_bucket_candidate("mtf_agree (0-3: KAMA急+日足SMA150↑+週足>30wMA)", mtf, R, times)
    year_distribution(mtf, times)

    print(f"\n実行コマンド: .venv/bin/python scratchpad/strength_regime_btc15mL.py"
          f"{' --smoke' if cli.smoke else ''}")


if __name__ == "__main__":
    main()
