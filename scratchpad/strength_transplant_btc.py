"""仕様カード6 scratchpad/spec_strength_transplant_btc.md の実装。

問い: btc15m_L で生き残った2軸 stop_atr(=risk/ATR)＋atr_pctile(ATRのtrailing500分位) の
合成強度旗 combo=(rank_pct(stop_atr)+rank_pct(atr_pctile))/2 が、他のBTCレッグ
(btc15m_S / btc_bo_kama / btc_pull) でも効くか＝BTC横断プリミティブか btc15m_L固有か。

土台（再発明禁止・import流用）:
  scratchpad/strength_btc15mL.py       -- build/rebuild_entries/match_entries_to_trades/
                                           quintile_table/block_bootstrap_spearman/report_candidate
  scratchpad/strength_gateslope_generalize.py -- btc15m_S の構築(build_btc15mS)・
                                           gate1_check/gate2_check（照合ゲート済みの経路）
  scratchpad/strength_entryquality_btc15mL.py -- atr_percentile_at（trailing500分位）
  scratchpad/strength_regime_btc15mL.py       -- compute_kama_slope（4H/日足KAMA傾き、
                                           gate_kama と同じ shift(1)+ffill 規約）
  research/portfolio_kama.py  -- kama_gate_btc / cycle_gate_pull（book.py と字句一致で再利用、
                                  boolean行マスクなので全列を素通しできる＝手書きマスク不要）
  research/book.py            -- get_book_legs()（照合ゲート1の正解データ）

対象3レッグの構築（research/book.py get_book_legs L86-92, L107-112 と厳密一致）:
  1. btc15m_S (L107-112): gg.build_btc15mS() をそのまま使う（gg内部で既に gate1/2/3 照合済みの
     経路と同一）。ATRは d15(15m)。invert()はTrue-Range各成分を並べ替えるだけで値は保存する
     （TR(inv,i)==TR(real,i)。ATRはinv/dのどちらで計算しても同じ値）ので、spec指定どおり d15 で計算。
  2. btc_bo_kama (L86-89): b4=resample(btc_h1,"4h") -> t_full=run(b4,{CFG,rr2,fwd300}) ->
     kama_gate_btc(t_full) で行マスク（全列保持）。pullback_frac=0(市場成行、CFGにこの属性は
     無いので match_entries_to_trades には明示的に 0.0 を渡す)＝エントリー足=確定足i
     （walk()の市場分岐は e_bar=i をそのままtimeに使うので、entries直呼び再構築での
     照合が可能）。ATRはb4(4h)。
  3. btc_pull (L90-92): b4 (同上) -> t_full=run_pb(b4,"long",{PB},0.0) -> cycle_gate_pull(t_full)
     で行マスク。walk_ema() は entries=(i,e,stop) の i をそのまま time=d.index[i] にする
     （fillバーのズレが無い＝ pullback-limit系のような entries<->trades 再対応付けが不要）ので、
     time -> b4.index.get_indexer() で直接 i を復元する（spec の指示どおり）。ATRはb4(4h)。

照合ゲート:
  ゲート1(全レッグ): 自作netR(book.py と同じ関数・同じマスク適用) vs
                      research.book.get_book_legs()[leg] の時刻・値一致。
  ゲート2(btc15m_S/btc_bo_kama): entries直呼び再構築 t2 が run()の生トレード表と bit一致。
  ゲート3(btc15m_S/btc_bo_kama): entries<->trades の全数一意対応（base.match_entries_to_trades）。
  btc_pull: walk_ema の time が d.index[i] と厳密一致することの自己点検（fill-bar再構築が
  不要な構造なので、上記2/3に相当するチェックはこの一致確認そのもの）。

強度候補（各レッグ自身のTFで・no-lookahead）:
  stop_atr   = leg.risk / ATR(14)[entry]
  atr_pctile = ATR(14)[entry] の trailing 500本percentile（同TF）
  combo(2軸) = (rank_pct(stop_atr) + rank_pct(atr_pctile)) / 2
  combo(3軸, 参考) = 上記 + そのレッグのゲート指標の傾き（kama_slope。btc15m_Sは日足KAMA(14)
    下向きの急さ[gateslope_generalize と同じ符号反転]、btc_bo_kamaは日足KAMA(14)上向きの急さ、
    btc_pullはKAMAゲートが無いので「そのレッグのゲートTF傾き」の指示どおり、ゲートの元系列＝
    週足30週SMA(cycle_gate_pullと同じ shift(1)+ffill 規約)の1本あたり変化率を代用する
    （注記: 文字通りのKAMAではなく、そのレッグの実ゲート指標のslope。参考のみ・判定の主軸ではない）。

測り方: btc15m_S(n≈100)はトップ20% vs 残り、btc_bo_kama/btc_pull(n≈70)はトップ1/3 vs 残り2/3。
巡回ブロックbootstrap(1/3/6/12mo, 3000回)のmeanRギャップ95%CI＋年別符号＋合成Spearman(block
bootstrap CI付き)。5分位表はbtc15m_Sのみ(参考、base.report_candidateで一括: quintile+monotone+
spearman+block-bootstrap-spearman+random-drop-null)。他2レッグは薄いため5分位を作らない。

Run:
  .venv/bin/python scratchpad/strength_transplant_btc.py --smoke 2>&1 | \\
      tee scratchpad/out_strength_transplant_btc_smoke.txt
  .venv/bin/python scratchpad/strength_transplant_btc.py 2>&1 | \\
      tee scratchpad/out_strength_transplant_btc.txt
"""
import argparse
import contextlib
import io
import os
import sys
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import strength_btc15mL as base                  # build/rebuild_entries/match_entries_to_trades/
                                                   # quintile_table/monotone_flag/
                                                   # block_bootstrap_spearman/random_drop_null/
                                                   # report_candidate
import strength_gateslope_generalize as gg        # build_btc15mS/gate1_check/gate2_check
import strength_regime_btc15mL as reg             # compute_kama_slope
import strength_entryquality_btc15mL as eq        # atr_percentile_at

from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from ema_pullback import run as run_pb
from research.regime_gate_lab import CFG
from research.portfolio_kama import kama_gate_btc, cycle_gate_pull, PB

BTC_H1 = f"{ROOT}/data/vantage_btcusd_h1.csv"


# ==================================================================== 共通ヘルパ

def pf_stat(R):
    R = np.asarray(R, dtype=float)
    n = len(R)
    if n == 0:
        return n, float("nan"), float("nan"), float("nan"), 0.0
    win = 100.0 * (R > 0).mean()
    pos = R[R > 0].sum(); neg = abs(R[R <= 0].sum())
    pf = pos / neg if neg > 0 else float("inf")
    return n, win, pf, R.mean(), R.sum()


def print_baseline(label, R, span_years):
    n, win, pf, mr, tr = pf_stat(R)
    pf_s = f"{pf:.2f}" if np.isfinite(pf) else "inf"
    print(f"  [{label}] n={n} ({n/span_years:.1f}/yr)  win={win:.1f}%  PF={pf_s}  "
          f"meanR={mr:+.4f}  totR={tr:+.1f}")


def topgap_table(R, top_mask):
    rows = []
    for name, mask in [("top", top_mask), ("rest", ~top_mask)]:
        n, win, pf, mr, tr = pf_stat(R[mask])
        rows.append((name, n, win, pf, mr, tr))
    print(f"  {'group':<6}{'n':>6}{'win%':>8}{'PF':>8}{'meanR':>9}{'totR':>9}")
    for name, n, win, pf, mr, tr in rows:
        pf_s = f"{pf:.2f}" if np.isfinite(pf) else "inf"
        print(f"  {name:<6}{n:>6}{win:>7.1f}%{pf_s:>8}{mr:>+9.3f}{tr:>+9.1f}")
    gap = rows[0][4] - rows[1][4]
    print(f"  gap(top-rest) meanR = {gap:+.4f}")
    return gap


def block_bootstrap_topgap(times, top_mask, R, k_months, n_boot=3000, seed=20260718):
    """top群 - 残り群 の meanR差を、巡回月ブロック(k_months本連続)でブートストラップする。
    base.block_bootstrap_spearman と同じ月ブロック抽出方式を、統計量だけ差し替えて流用。"""
    s = pd.DataFrame({"top": np.asarray(top_mask, dtype=bool), "R": np.asarray(R, dtype=float)},
                      index=pd.DatetimeIndex(times))
    months = sorted(s.index.to_period("M").unique())
    nm = len(months)
    by_month = {m: s[s.index.to_period("M") == m] for m in months}
    rng = np.random.default_rng(seed)
    nblk = int(np.ceil(nm / k_months))
    diffs = []
    for _ in range(n_boot):
        starts = rng.integers(0, nm, size=nblk)
        seq = np.concatenate([[(st + j) % nm for j in range(k_months)] for st in starts])
        samp = pd.concat([by_month[months[j]] for j in seq])
        top = samp.loc[samp["top"], "R"]; rest = samp.loc[~samp["top"], "R"]
        if len(top) == 0 or len(rest) == 0:
            continue
        diffs.append(top.mean() - rest.mean())
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.median(diffs)), float(lo), float(hi), len(diffs)


def report_topgap_bootstrap(times, top_mask, R, n_boot=3000):
    print(f"\n  トップvs残り meanRギャップの巡回ブロックbootstrap({n_boot}回):")
    for k in (1, 3, 6, 12):
        med, lo, hi, nv = block_bootstrap_topgap(times, top_mask, R, k, n_boot=n_boot)
        tag = "0超" if (np.isfinite(lo) and lo > 0) else ("0未満" if (np.isfinite(hi) and hi < 0) else "0またぎ")
        print(f"    {k:>2}mo: 中央値 {med:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  ({tag})  "
              f"(有効draw={nv}/{n_boot})")


def era_topgap(times, top_mask, R):
    s = pd.DataFrame({"top": np.asarray(top_mask, dtype=bool), "R": np.asarray(R, dtype=float)},
                      index=pd.DatetimeIndex(times))
    s["year"] = s.index.year
    print("  年別 top vs rest meanR:")
    diffs = []
    for yr, g in s.groupby("year"):
        t = g.loc[g["top"], "R"]; r = g.loc[~g["top"], "R"]
        if len(t) == 0 or len(r) == 0:
            print(f"    {yr}: top n={len(t)}  rest n={len(r)}  (どちらか0本のため差スキップ)")
            continue
        d = t.mean() - r.mean()
        diffs.append(d)
        print(f"    {yr}: top n={len(t):>3} meanR={t.mean():+.3f} | rest n={len(r):>3} meanR={r.mean():+.3f} "
              f"| 差={d:+.3f}")
    diffs = np.array(diffs)
    if len(diffs):
        print(f"    符号: 正{int((diffs>0).sum())}/{len(diffs)}年  中央値={np.median(diffs):+.3f}  "
              f"標準偏差={diffs.std():.3f}")
    else:
        print("    年別比較不能（top/restどちらかが0本の年しかない）")
    return diffs


def spearman_report(name, x, R, times, n_boot=1000):
    x = np.asarray(x, dtype=float); R = np.asarray(R, dtype=float)
    rho, p = spearmanr(x, R)
    print(f"\n  [{name}] Spearman(x,R) = {rho:+.4f} (p={p:.4g}, n={len(x)})")
    for k in (1, 3, 6, 12):
        med, lo, hi, nv = base.block_bootstrap_spearman(times, x, R, k, n_boot=n_boot)
        print(f"    循環ブロック({k}mo) bootstrap median rho={med:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  "
              f"(有効draw={nv}/{n_boot})")
    return rho


def build_combo(stop_atr, atr_pctile, extra=None):
    """combo = (rank_pct(stop_atr)+rank_pct(atr_pctile))/2。extra があれば3軸平均も返す。
    NaN を含む行は事前に呼び出し側で落とす前提（純粋にrank_pctするだけ）。"""
    d = {"stop_atr": pd.Series(stop_atr), "atr_pctile": pd.Series(atr_pctile)}
    combo2 = (d["stop_atr"].rank(pct=True) + d["atr_pctile"].rank(pct=True)) / 2
    combo3 = None
    if extra is not None:
        e = pd.Series(extra)
        combo3 = (d["stop_atr"].rank(pct=True) + d["atr_pctile"].rank(pct=True) + e.rank(pct=True)) / 3
    return combo2.values, (combo3.values if combo3 is not None else None)


def build_b4(smoke):
    with contextlib.redirect_stderr(io.StringIO()):
        raw = load_mt5_csv(BTC_H1)
        if smoke:
            raw = raw.loc[:"2019-12-31"]
        b4 = resample(raw, "4h")
    return b4


# ==================================================================== レッグ1: btc15m_S

def run_btc15mS(cli_smoke):
    print(f"\n{'#'*78}\n# btc15m_S -- 合成強度旗の移植検定 (n≈100, トップ20% vs 残り)\n{'#'*78}")
    d15, inv, C, args, ts, mS, netR = gg.build_btc15mS(cli_smoke)
    print(f"btc15m_S 再構築(マスク後): n={mS.sum()}/{len(ts)}  (smoke={cli_smoke})")

    mine = pd.Series(netR, index=pd.DatetimeIndex(ts["time"])[mS])
    gate1 = gg.gate1_check("btc15m_S", mine, cli_smoke)
    if gate1 is False:
        print("!!! btc15m_S 照合ゲート1 FAIL -- 以降の数字は信用しないこと。btc15m_Sをスキップする。")
        return

    entries, t2 = base.rebuild_entries(inv, args)
    gate2 = gg.gate2_check("btc15m_S", t2, ts)
    if not gate2:
        print("!!! btc15m_S 照合ゲート2 FAIL -- entries復元を信用できない。btc15m_Sをスキップする。")
        return

    i_arr_full = base.match_entries_to_trades(entries, ts, args.pullback_frac)
    print(f"[btc15m_S 照合ゲート3] entries<->trades 対応付け(マスク前): "
          f"{len(i_arr_full)}/{len(ts)} 本すべて一意対応 => PASS")

    i_arr = i_arr_full[mS]
    times = ts["time"].values[mS]
    R = netR
    span_years = max((pd.DatetimeIndex(times).max() - pd.DatetimeIndex(times).min()).days / 365.25, 0.1)
    print_baseline("btc15m_S baseline", R, span_years)

    # ATR: d15(実価格, 反転前)。TR(inv)==TR(実)なので inv で計算しても同値(TR成分の入替のみ)。
    atr_d15 = ta.atr(d15["high"], d15["low"], d15["close"], length=args.atr).values

    stop_atr = ts["risk"].values[mS] / atr_d15[i_arr]
    atr_pctile = eq.atr_percentile_at(atr_d15, i_arr, window=500)

    # 3軸参考: btc15m_Sのゲートは日足KAMA(14)下向き(=inv上昇)。実価格の下向きの急さを正で表現
    # (strength_gateslope_generalize.py と同じ符号反転)。
    ks_real_arr, _ = reg.compute_kama_slope(d15, n=args.gate_kama, tf="1D")
    kama_slope_down = -ks_real_arr[i_arr]

    df = pd.DataFrame({"R": R, "t": times, "sa": stop_atr, "ap": atr_pctile, "ks": kama_slope_down})
    n_nan = df[["sa", "ap"]].isna().any(axis=1).sum()
    df2 = df.dropna(subset=["sa", "ap"]).reset_index(drop=True)
    print(f"[stop_atr/atr_pctile] window不足等でNaN除外: {n_nan}/{len(df)}  有効n={len(df2)}")

    combo2, _ = build_combo(df2["sa"].values, df2["ap"].values)
    df2["combo2"] = combo2

    print("\n  [参考] combo(2軸) の5分位表・Spearman・巡回ブロックbootstrap(spearman)・random-drop null:")
    base.report_candidate("combo2 (stop_atr+atr_pctile, btc15m_S, 参考5分位)", df2["combo2"].values,
                           df2["R"].values, df2["t"].values)

    top_mask = (df2["combo2"] >= df2["combo2"].quantile(0.8)).values
    print(f"\n  === 判定本体: トップ20% (combo2>=p80, n={top_mask.sum()}) vs 残り (n={(~top_mask).sum()}) ===")
    topgap_table(df2["R"].values, top_mask)
    report_topgap_bootstrap(df2["t"].values, top_mask, df2["R"].values)
    era_topgap(df2["t"].values, top_mask, df2["R"].values)

    # 3軸版(参考)
    df3 = df.dropna(subset=["sa", "ap", "ks"]).reset_index(drop=True)
    combo2_3, combo3 = build_combo(df3["sa"].values, df3["ap"].values, extra=df3["ks"].values)
    spearman_report("combo3 (stop_atr+atr_pctile+kama_slope_down, 参考)", combo3, df3["R"].values,
                     df3["t"].values)

    return dict(leg="btc15m_S", n=len(df2), top_mask=top_mask, R=df2["R"].values, t=df2["t"].values,
                combo=df2["combo2"].values)


# ==================================================================== レッグ2: btc_bo_kama

def run_btc_bo_kama(cli_smoke):
    print(f"\n{'#'*78}\n# btc_bo_kama -- 合成強度旗の移植検定 (n≈70, トップ1/3 vs 残り2/3)\n{'#'*78}")
    b4 = build_b4(cli_smoke)
    args = SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0, "fwd": 300})
    with contextlib.redirect_stderr(io.StringIO()):
        t_full = run(b4, args)
    if t_full is None:
        raise SystemExit("btc_bo_kama: no entries (smoke過ぎ?)")
    print(f"btc_bo_kama 再構築(マスク前): n={len(t_full)}  span={t_full['time'].iloc[0]} -> "
          f"{t_full['time'].iloc[-1]}  (smoke={cli_smoke})")

    tK = kama_gate_btc(t_full)  # research/portfolio_kama.py と字句一致で再利用（全列を行マスク）
    print(f"kama_gate_btc 適用後: n={len(tK)}/{len(t_full)}")

    mine = pd.Series(tK["R"].values, index=pd.DatetimeIndex(tK["time"]))
    gate1 = gg.gate1_check("btc_bo_kama", mine, cli_smoke)
    if gate1 is False:
        print("!!! btc_bo_kama 照合ゲート1 FAIL -- 以降の数字は信用しないこと。btc_bo_kamaをスキップする。")
        return

    entries, t2 = base.rebuild_entries(b4, args)
    gate2 = gg.gate2_check("btc_bo_kama", t2, t_full)
    if not gate2:
        print("!!! btc_bo_kama 照合ゲート2 FAIL -- entries復元を信用できない。btc_bo_kamaをスキップする。")
        return

    i_arr_full = base.match_entries_to_trades(entries, t_full, 0.0)  # pullback_frac=0(市場成行)
    print(f"[btc_bo_kama 照合ゲート3] entries<->trades 対応付け(マスク前): "
          f"{len(i_arr_full)}/{len(t_full)} 本すべて一意対応 => PASS")

    mask_arr = t_full["time"].isin(tK["time"]).values
    i_arr = i_arr_full[mask_arr]
    times = t_full["time"].values[mask_arr]
    R = tK["R"].values
    risk = tK["risk"].values
    span_years = max((pd.DatetimeIndex(times).max() - pd.DatetimeIndex(times).min()).days / 365.25, 0.1)
    print_baseline("btc_bo_kama baseline", R, span_years)

    atr_b4 = ta.atr(b4["high"], b4["low"], b4["close"], length=args.atr).values
    stop_atr = risk / atr_b4[i_arr]
    atr_pctile = eq.atr_percentile_at(atr_b4, i_arr, window=500)

    ks_arr, _ = reg.compute_kama_slope(b4, n=14, tf="1D")   # kama_gate_btc と同じ n=14, tf=1D
    kama_slope_up = ks_arr[i_arr]

    df = pd.DataFrame({"R": R, "t": times, "sa": stop_atr, "ap": atr_pctile, "ks": kama_slope_up})
    n_nan = df[["sa", "ap"]].isna().any(axis=1).sum()
    df2 = df.dropna(subset=["sa", "ap"]).reset_index(drop=True)
    print(f"[stop_atr/atr_pctile] window不足等でNaN除外: {n_nan}/{len(df)}  有効n={len(df2)}  "
          f"(n/年={len(df2)/span_years:.1f} -- 薄いので5分位は作らない)")

    combo2, _ = build_combo(df2["sa"].values, df2["ap"].values)
    df2["combo2"] = combo2
    spearman_report("combo2 (stop_atr+atr_pctile, btc_bo_kama)", df2["combo2"].values, df2["R"].values,
                     df2["t"].values)

    top_mask = (df2["combo2"] >= df2["combo2"].quantile(2.0 / 3.0)).values
    print(f"\n  === 判定本体: トップ1/3 (combo2>=p66.7, n={top_mask.sum()}) vs 残り2/3 (n={(~top_mask).sum()}) ===")
    topgap_table(df2["R"].values, top_mask)
    report_topgap_bootstrap(df2["t"].values, top_mask, df2["R"].values)
    era_topgap(df2["t"].values, top_mask, df2["R"].values)

    df3 = df.dropna(subset=["sa", "ap", "ks"]).reset_index(drop=True)
    _, combo3 = build_combo(df3["sa"].values, df3["ap"].values, extra=df3["ks"].values)
    spearman_report("combo3 (stop_atr+atr_pctile+kama_slope_up, 参考)", combo3, df3["R"].values,
                     df3["t"].values)

    return dict(leg="btc_bo_kama", n=len(df2), top_mask=top_mask, R=df2["R"].values, t=df2["t"].values,
                combo=df2["combo2"].values)


# ==================================================================== レッグ3: btc_pull

def run_btc_pull(cli_smoke):
    print(f"\n{'#'*78}\n# btc_pull -- 合成強度旗の移植検定 (n≈70, トップ1/3 vs 残り2/3, walk_ema)\n{'#'*78}")
    b4 = build_b4(cli_smoke)
    args = SimpleNamespace(**{**PB, "csv": "x", "tf": "4h"})
    with contextlib.redirect_stderr(io.StringIO()):
        t_full = run_pb(b4, "long", args, 0.0)
    if t_full is None:
        raise SystemExit("btc_pull: no entries (smoke過ぎ?)")
    print(f"btc_pull 再構築(マスク前): n={len(t_full)}  span={t_full['time'].iloc[0]} -> "
          f"{t_full['time'].iloc[-1]}  (smoke={cli_smoke})")

    tP = cycle_gate_pull(t_full)  # research/portfolio_kama.py と字句一致で再利用（全列を行マスク）
    print(f"cycle_gate_pull 適用後: n={len(tP)}/{len(t_full)}")

    mine = pd.Series(tP["R"].values, index=pd.DatetimeIndex(tP["time"]))
    gate1 = gg.gate1_check("btc_pull", mine, cli_smoke)
    if gate1 is False:
        print("!!! btc_pull 照合ゲート1 FAIL -- 以降の数字は信用しないこと。btc_pullをスキップする。")
        return

    # walk_ema は entries=(i,e,stop) の time=d.index[i] をそのまま使う（fillバーのズレが無い）ので、
    # entries直呼び再構築は不要 -- time から b4.index への逆引きで i を厳密に復元できる。自己点検で確認。
    i_arr_full = b4.index.get_indexer(pd.DatetimeIndex(t_full["time"]))
    ok_idx = (i_arr_full >= 0).all() and (b4.index[i_arr_full] == pd.DatetimeIndex(t_full["time"])).all()
    ok_unique = len(set(i_arr_full)) == len(i_arr_full)
    print(f"[btc_pull entries復元(walk_emaのtime直接逆引き)] 全件 b4.index に厳密一致 -> {ok_idx}  "
          f"重複無し -> {ok_unique}  => {'PASS' if (ok_idx and ok_unique) else 'FAIL'}")
    if not (ok_idx and ok_unique):
        print("!!! btc_pull time->i 復元 FAIL -- 以降の数字は信用しないこと。btc_pullをスキップする。")
        return

    mask_arr = t_full["time"].isin(tP["time"]).values
    i_arr = i_arr_full[mask_arr]
    times = t_full["time"].values[mask_arr]
    R = tP["R"].values
    risk = tP["risk"].values
    span_years = max((pd.DatetimeIndex(times).max() - pd.DatetimeIndex(times).min()).days / 365.25, 0.1)
    print_baseline("btc_pull baseline", R, span_years)

    atr_b4 = ta.atr(b4["high"], b4["low"], b4["close"], length=args.atr).values
    stop_atr = risk / atr_b4[i_arr]
    atr_pctile = eq.atr_percentile_at(atr_b4, i_arr, window=500)

    # 3軸参考: btc_pullのゲートはKAMAでなく「週足終値 <= 30週MA*1.10」(レベル)。同じ shift(1)+ffill
    # 規約で、そのゲートの元系列(週足30週SMA)自身の1本あたり変化率を「ゲートTF傾き」として代用する
    # (文字通りのkama_slopeではない -- 注記のとおり参考のみ)。
    w30 = b4["close"].resample("1W").last().rolling(30).mean().shift(1)   # cycle_gate_pullと同じ定義
    w30_slope = ((w30 - w30.shift(1)) / w30.shift(1)).reindex(b4.index, method="ffill").values
    gate_slope = w30_slope[i_arr]

    df = pd.DataFrame({"R": R, "t": times, "sa": stop_atr, "ap": atr_pctile, "gs": gate_slope})
    n_nan = df[["sa", "ap"]].isna().any(axis=1).sum()
    df2 = df.dropna(subset=["sa", "ap"]).reset_index(drop=True)
    print(f"[stop_atr/atr_pctile] window不足等でNaN除外: {n_nan}/{len(df)}  有効n={len(df2)}  "
          f"(n/年={len(df2)/span_years:.1f} -- 薄いので5分位は作らない)")

    combo2, _ = build_combo(df2["sa"].values, df2["ap"].values)
    df2["combo2"] = combo2
    spearman_report("combo2 (stop_atr+atr_pctile, btc_pull)", df2["combo2"].values, df2["R"].values,
                     df2["t"].values)

    top_mask = (df2["combo2"] >= df2["combo2"].quantile(2.0 / 3.0)).values
    print(f"\n  === 判定本体: トップ1/3 (combo2>=p66.7, n={top_mask.sum()}) vs 残り2/3 (n={(~top_mask).sum()}) ===")
    topgap_table(df2["R"].values, top_mask)
    report_topgap_bootstrap(df2["t"].values, top_mask, df2["R"].values)
    era_topgap(df2["t"].values, top_mask, df2["R"].values)

    df3 = df.dropna(subset=["sa", "ap", "gs"]).reset_index(drop=True)
    _, combo3 = build_combo(df3["sa"].values, df3["ap"].values, extra=df3["gs"].values)
    spearman_report("combo3 (stop_atr+atr_pctile+週足30週SMA傾き, 参考・注記あり)", combo3, df3["R"].values,
                     df3["t"].values)

    return dict(leg="btc_pull", n=len(df2), top_mask=top_mask, R=df2["R"].values, t=df2["t"].values,
                combo=df2["combo2"].values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    r1 = run_btc15mS(cli.smoke)
    r2 = run_btc_bo_kama(cli.smoke)
    r3 = run_btc_pull(cli.smoke)

    print(f"\n{'='*78}\nまとめ（詳細は上の各レッグ）\n{'='*78}")
    for r in (r1, r2, r3):
        if r is None:
            print("  (照合ゲートFAILでスキップされたレッグあり)")
            continue
        n_top = int(r["top_mask"].sum())
        print(f"  {r['leg']:<14} n={r['n']:>4}  top群n={n_top}")

    print(f"\n実行コマンド: .venv/bin/python scratchpad/strength_transplant_btc.py"
          f"{' --smoke' if cli.smoke else ''}")


if __name__ == "__main__":
    main()
