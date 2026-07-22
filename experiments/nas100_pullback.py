"""nas100_pullback.py -- spec card `experiments/spec_nas100_pullback.md` (仕様カード12).

nas100 (NAS100.r, Vantage H1, 2016-) を ema_pullback.py (btc_pull と同じ機構: 80MA-slope
トレンド + 20MA押し目 + 確定足リクレイム・エントリー) の long のみで測る、装置由来の新規
単体レッグの「第一歩」。自前ウォーカーは実装しない -- ema_pullback.run()/resample() を
そのまま import して使う (src/engine/ 経由、breakout族と同じ分解エンジン)。

第一歩A: 全シグナルの stop距離 (指数ポイント) 分布。契約仕様(pt->$ 換算)がCSVから
  不明なので、0.01ロットの $/pt = v をパラメタに v in {0.01, 0.1, 1.0} で $ 換算表を出す
  (実際の値はユーザーのMT5確認が必要 -- ここではどのvでも「取れる/取れない」を判断できる
  ように3段を並べるだけ)。

第一歩B: PBプリセット (btc_pull の既定パラメタ, research/portfolio_kama.py の PB dict)
  を出発点に、TFラダー (1h/2h/4h/8h/1d; 15m/5mはデータ無し) で長さそのまま(チューニング
  なし)実行。cost=0.0 (素の gross R, コストは後段で別途 pt 建てで乗せる)。

ベータnull (最重要の反証): 2016-以降はほぼ一本調子の強気=ロング必勝時代なので、素の
  PF>1 だけでは何の情報にもならない。構成 (このスクリプト独自の実装, 既存の
  null 生成関数の直接流用ではない -- 明記):
    各TFで実測トレード数ぶんだけランダムな新規エントリー足(重複可・ウォームアップ
    100足以降)を引き、そのエントリー足の"stop距離"は実測トレード群の risk(pt) 分布から
    復元抽出でひとつ割り当てる。ema_pullback と同じ規約 (rr=PBのRR3, 確定足の終値で
    エントリー、翌足から fwd 本先まで安値がstopを割ったら-1R/高値がtargetを超えたら+RR/
    どちちも起きなければ fwd 本目の終値で mark) で再生し、そのトライアルのPFを記録。
    trials=2000 回繰り返し、実測PFがそのnull分布の何%ileかを出す。
    -- edge_harness.py の _beta_pct (同数ランダム・同機構での再生) と同じ発想だが、
    ema_pullback 側の risk 定義 (構造ピアース幅) をそのまま流用するため専用に書いた。
    n<12 のTFはnull計算をスキップ (edge_harnessの足切りに合わせる)。

判定: null percentile > 90 (edge_harness の "beta% >90 = real selection" 基準を流用) の
  TFがあれば次段へ、全TFで届かなければ「指数プルバックもベータの上澄み」と記録。

Run:
  .venv/bin/python experiments/nas100_pullback.py --smoke
  .venv/bin/python experiments/nas100_pullback.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from types import SimpleNamespace

from src.data_loader import load_mt5_csv
from ema_pullback import run as run_pb, resample

CSV = "data/vantage_nas100.r_h1.csv"
TFS = ["1h", "2h", "4h", "8h", "1d"]

# btc_pull の既定 (research/portfolio_kama.py PB dict) -- チューニングしない、そのまま。
PB = dict(ema_fast=20, ema_slow=80, slope_k=6, filter="slope", er_period=14,
          swap_pct=0.0, gate_tf="", gate_type="ema-slope", gate_n=14,
          exit_sma=0, exit_ma_type="sma", peryear=False,
          no_overlap=True, entry_trigger="close", fill_at_close=True, rr=3.0,
          min_stop_atr=0.5, atr=14, fwd=90, cost=0.0,
          trend_ma_type="sma", fast_ma_type="ema")


def get_trades(d_tf, tf):
    args = SimpleNamespace(**{**PB, "csv": "x", "tf": tf})
    t = run_pb(d_tf, "long", args, 0.0)
    return t


def pf(r):
    r = np.asarray(r, dtype=float)
    pos, neg = r[r > 0].sum(), r[r <= 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else float("nan")
    return pos / abs(neg)


def is_oos(t):
    yrs = sorted(t["y"].unique())
    half = yrs[len(yrs) // 2] if len(yrs) > 1 else None
    isr = t[t["y"] < half]["R"] if half else t["R"]
    oosr = t[t["y"] >= half]["R"] if half else t["R"]
    return isr.mean(), oosr.mean()


def beta_null(t, d_tf, trials, seed=0):
    """同数ランダム建て(risk分布は実測から復元抽出)・同機構(rr/fwd)の null PF 分布。
    docstring 冒頭の「構成」参照。n<12 なら (nan, nan) を返す。"""
    if t is None or len(t) < 12:
        return float("nan"), float("nan"), np.array([])
    rng = np.random.default_rng(seed)
    c, h, l = d_tf["close"].values, d_tf["high"].values, d_tf["low"].values
    n_bars = len(c)
    n_trades = len(t)
    risks = t["risk"].values
    rr = PB["rr"]
    fwd = PB["fwd"]
    warmup = 100
    valid = np.arange(warmup, n_bars - 1)
    real_pf = pf(t["R"].values)
    pfs = []
    for _ in range(trials):
        idx = rng.choice(valid, size=n_trades, replace=True)
        rk = rng.choice(risks, size=n_trades, replace=True)
        Rs = np.empty(n_trades)
        for k in range(n_trades):
            i, risk = idx[k], rk[k]
            e = c[i]
            stop = e - risk
            tgt = e + rr * risk
            end = min(i + 1 + fwd, n_bars)
            R = None
            for j in range(i + 1, end):
                if l[j] <= stop:
                    R = -1.0
                    break
                if h[j] >= tgt:
                    R = rr
                    break
            if R is None:
                exit_j = end - 1
                R = (c[exit_j] - e) / risk
            Rs[k] = R
        pfs.append(pf(Rs))
    pfs = np.array(pfs)
    pfs_finite = pfs[np.isfinite(pfs)]
    pct = (pfs_finite < real_pf).mean() * 100 if len(pfs_finite) else float("nan")
    return real_pf, pct, pfs_finite


def stepA(df_h1, tfs, quiet_years=False):
    print("\n=== 第一歩A: stop距離 (指数pt) 分布 + $換算表 (v=0.01ロットの$/pt) ===")
    print(f"  現在価格帯: 最終H1終値={df_h1['close'].iloc[-1]:.2f}  "
          f"最古H1終値(2016年初)={df_h1['close'].iloc[0]:.2f}")
    for tf in tfs:
        d = resample(df_h1, tf)
        t = get_trades(d, tf)
        if t is None or len(t) == 0:
            print(f"\n  -- TF={tf}: エントリー無し --")
            continue
        stop = t["risk"].values
        med, p25, p75, p90 = (np.median(stop), np.percentile(stop, 25),
                               np.percentile(stop, 75), np.percentile(stop, 90))
        print(f"\n  -- TF={tf}  n={len(t)} --")
        print(f"     stop(pt): median={med:.1f}  p25={p25:.1f}  p75={p75:.1f}  p90={p90:.1f}")
        if not quiet_years:
            per_y = t.groupby("y")["risk"].median()
            print("     年別 median stop(pt): " + " ".join(f"{y}:{v:.0f}" for y, v in per_y.items()))
        print("     $換算 (0.01ロット, v=$/pt):")
        for v in (0.01, 0.1, 1.0):
            print(f"       v={v:<4}  median stop$={med*v:>8.2f}  p90 stop$={p90*v:>8.2f}")


def stepB(df_h1, tfs, trials):
    print("\n=== 第一歩B: 素の測定 (全シグナル, long only, gate=off, cost=0 gross) ===")
    print(f"  {'TF':<4}{'n':>6}{'n/yr':>7}{'win%':>7}{'PF':>7}{'meanR':>8}{'totR':>8}"
          f"{'IS':>7}{'OOS':>7}{'nullPF%ile':>12}")
    results = {}
    for tf in tfs:
        d = resample(df_h1, tf)
        t = get_trades(d, tf)
        if t is None or len(t) == 0:
            print(f"  {tf:<4}  no entries")
            continue
        n = len(t)
        span_yrs = max((t["time"].iloc[-1] - t["time"].iloc[0]).days / 365.25, 0.5)
        npy = n / span_yrs
        win = (t["R"] > 0).mean() * 100
        PF = pf(t["R"].values)
        meanR = t["R"].mean()
        totR = t["R"].sum()
        isr, oosr = is_oos(t)
        real_pf, pct, nullpfs = beta_null(t, d, trials)
        pf_str = f"{PF:.2f}" if np.isfinite(PF) else "inf"
        pct_str = f"{pct:.0f}%" if np.isfinite(pct) else "n/a(<12)"
        print(f"  {tf:<4}{n:>6}{npy:>7.1f}{win:>6.0f}%{pf_str:>7}{meanR:>+8.3f}{totR:>+8.1f}"
              f"{isr:>+7.2f}{oosr:>+7.2f}{pct_str:>12}")
        print("       per-year totR: " + " ".join(
            f"{y}:{g['R'].sum():+.1f}(n{len(g)})" for y, g in t.groupby("y")))
        if len(nullpfs):
            print(f"       null PF dist (n={len(nullpfs)} finite trials): "
                  f"median={np.median(nullpfs):.2f}  p10={np.percentile(nullpfs,10):.2f}  "
                  f"p90={np.percentile(nullpfs,90):.2f}")
        results[tf] = dict(t=t, n=n, npy=npy, win=win, PF=PF, meanR=meanR, totR=totR,
                            IS=isr, OOS=oosr, null_pct=pct)
    return results


def cost_sensitivity(results):
    print("\n=== コスト感度 (0 / 2pt / 5pt, 素のgross Rから後掛け; 実測スプレッドは別途) ===")
    print(f"  {'TF':<4}{'cost':>6}{'PF':>8}{'meanR':>9}{'totR':>9}")
    for tf, r in results.items():
        t = r["t"]
        for cpt in (0, 2, 5):
            Rc = t["R"].values - cpt / t["risk"].values
            PFc = pf(Rc)
            pf_str = f"{PFc:.2f}" if np.isfinite(PFc) else "inf"
            print(f"  {tf:<4}{cpt:>5}pt{pf_str:>7}{Rc.mean():>+9.3f}{Rc.sum():>+9.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="直近2年だけ (配線確認)")
    p.add_argument("--trials", type=int, default=2000, help="null bootstrap trials")
    p.add_argument("--start", default=None)
    args = p.parse_args()

    df = load_mt5_csv(CSV)
    if args.smoke:
        start = df.index[-1] - pd.Timedelta(days=365 * 2)
        df = df.loc[start:]
        print(f"[SMOKE] using last ~2yr: {df.index[0]} -> {df.index[-1]}  ({len(df)} H1 bars)")
    elif args.start:
        df = df.loc[args.start:]

    print(f"nas100 H1 loaded: {df.index[0]} -> {df.index[-1]}  ({len(df):,} bars)  src={CSV}")

    stepA(df, TFS, quiet_years=args.smoke)
    trials = 200 if args.smoke else args.trials
    results = stepB(df, TFS, trials)
    cost_sensitivity(results)

    print("\n=== 判定 (null%ile>90 の TF の有無; >90 = beta_pctのedge_harness基準を流用) ===")
    any_pass = False
    for tf, r in results.items():
        flag = np.isfinite(r["null_pct"]) and r["null_pct"] > 90
        any_pass = any_pass or flag
        print(f"  {tf}: null%ile={r['null_pct']:.0f}%  {'PASS(>90)' if flag else 'fail/na'}"
              if np.isfinite(r["null_pct"]) else f"  {tf}: null%ile=n/a (n<12)")
    print(f"  -> {'null超えTFあり: 次段へ' if any_pass else '全TFでnull以下(または判定不能): ベータの上澄み疑い'}")
    print("  (注) 2016->は8.5年=1レジームのみのサンプル。時代分散は測れない。")


if __name__ == "__main__":
    main()
