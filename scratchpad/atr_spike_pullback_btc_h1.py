"""仕様カード『BTC h1 ATR拡大足 — 入口を押し目指値に差し替えて固定RR出口で評価』の実装。

前段（凍結・再測定不要）: BTC h1 実体>ATR14[s-1]*k の拡大足は方向情報を持つ
(k=2.0 long: n=701, 76.7本/年, MFE中央値2.80R, 比1.545, 帰無%ile=100)。

今回のスコープ: 引き金は動かさず、入口だけ成行->押し目指値に差し替える。出口=固定RR。
実行は必ず src/engine/walk.py の walk()（ロング）/ src/engine/mirror.py の invert()（ショート）
を import して使う。自前の前方走査は書かない。

SCREEN = "atr_spike_btc_h1"

設計判断（仕様カードに明記が無かった点。ここに書く）:
  - walk() の entries タプル (i, e, stop, tgt, i_origin) の i は「busy_until のスロット判定用
    インデックス」であり、市場成行(pf=0)ではそのまま e_bar(建値バー)としても使われる。
    仕様は e=次足始値（=open[s+1]）を要求するので、i=s（引き金足自身）に設定した。こうすると
    market entryの forward scan は range(i+1, ...) = s+1 から始まり、s+1足自身のH/Lがstop/tgt
    判定に含まれる（open[s+1]で建てた直後からその足の残り値幅が有効に効く、というのが正しい
    約定モデル）。pullback-limit の fill-window scan も i+1=s+1 から始まるので「次足以降で指値待ち」
    という仕様の文言と一致する。i_origin=i=s とした（base_bars列は常に0になるが、このスクリプトの
    設計ではwaveの概念が無いため使わない列）。
  - A系・B系とも stop は「作業フレーム」(ロングは元データ、ショートは mirror.invert() した
    反転フレーム)上で常に「low[s]」(A系)または「e-stopk*atr_prev[s]」(B系)として計算する。
    ショート側は反転フレームに対して「ロングと同じ計算式」を適用するだけで正しい鏡像になる
    （mirror.py のコメント通り）。
  - maxDD は「トレード解像度」だが、仕様の縛り「0.01ロット固定・サイズ写像禁止」に従い、
    src.engine.stats.metrics() の risk%複利DD（サイズ写像を内包する）は使わず、
    累積R（cumsum、複利なし=固定ベット）のピークからの最大下落をR単位で報告する
    （maxDD_R）。PF/meanR/win%はいずれもR単位でサイズ非依存。
  - コストは args.cost=0 で walk() を1回だけ回し、trades の risk・e_px 列から
    R_cost = R_raw - cost_frac/risk*e_px を後付けでベクトル化計算する（walk()を3回回すより高速、
    walk() 内部のコスト適用式 `R -= args.cost/risk*e_px` と数式として同一なので数値は一致する）。
  - グリッド規模の都合による縮小（明記のうえ縮小）:
    - fwd=60（副）はRRを{2,3,4.5}・fill_win=200のみに絞る（fwd=20の主グリッドはRR全5値×fill_win全3値
      ×pf全6値×kストップ全部を回す）。理由: フルクロスは fwd20 だけで約3240セルあり、fwd60を
      同じ密度で足すと計算コストが倍増する一方、仕様は fwd=60 を「副」（優先度が低い比較対象）と
      明記しているため。
    - 年別PF/Nの内訳は全セルではなく「pf=0基準」と「各(k,system,stopk,direction)で最良のpf」の
      代表セルのみ印字する（3240セル全部の年別内訳は非現実的な量になるため）。フルグリッドの
      集計値（n, per_year, win, PF, meanR, totR, maxDD_R, 約定率, 間引き帰無%ile）はCSVに全セル出す。

Run:
  .venv/bin/python scratchpad/atr_spike_pullback_btc_h1.py --smoke 2>&1 | tee scratchpad/out_atr_spike_pullback_smoke.txt
  .venv/bin/python scratchpad/atr_spike_pullback_btc_h1.py 2>&1 | tee scratchpad/out_atr_spike_pullback_full.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time as _time
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pandas_ta as ta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_loader import load_mt5_csv          # noqa: E402
from src.engine.walk import walk                   # noqa: E402  -- THE execution walker (do not reinvent)
from src.engine.mirror import invert                # noqa: E402  -- short side = long machinery on inverted frame
from research.screen import run_screen              # noqa: E402

SCREEN = "atr_spike_btc_h1"

BTC_H1 = f"{ROOT}/data/vantage_btcusd_h1.csv"

KS = [1.5, 2.0, 2.5]
PFS = [0.0, 0.25, 0.382, 0.5, 0.618, 0.786]
FILL_WINS = [20, 50, 200]
RRS = [1.0, 1.5, 2.0, 3.0, 4.5]
FWD_MAIN = 20
FWD_SUB = 60
RRS_SUB = [2.0, 3.0, 4.5]
FILL_WIN_SUB = 200
COST_LADDER = [0.00025, 0.0005, 0.001]
N_NULL = 200
RNG = np.random.default_rng(20260721)


# ------------------------------------------------------------------ features

def compute_features(df):
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    atr_prev = atr.shift(1)          # s-1までで確定 (自己参照回避)
    body = (df["close"] - df["open"]).abs()
    return atr_prev.values, body.values


def raw_triggers(df, atr_prev, body, k):
    """引き金マスク（ストップ/リスクの妥当性は問わない、第1段と同じ定義）。"""
    warm = ~np.isnan(atr_prev)
    long_dir = df["close"].values > df["open"].values
    trig = warm & (body > atr_prev * k) & long_dir
    return np.where(trig)[0]


# ------------------------------------------------------------------ entries construction

def build_entries(df, atr_prev, s_idx, system, stopk, rr):
    """entries = (i, e, stop, tgt, i_origin) 。i=引き金足s自身、e=open[s+1]。
    A系: stop=low[s]。 B系: stop=e-stopk*atr_prev[s]。risk<=0 は除外。"""
    o = df["open"].values
    l = df["low"].values
    n = len(df)
    entries = []
    for s in s_idx:
        if s + 1 >= n:
            continue
        e = o[s + 1]
        if system == "A":
            stop = l[s]
        else:
            stop = e - stopk * atr_prev[s]
        risk = e - stop
        if risk <= 0:
            continue
        tgt = e + rr * risk
        if tgt <= e:
            continue
        entries.append((s, e, stop, tgt, s))
    return entries


def run_walk(df, entries, pf, fill_win, fwd, cost=0.0):
    if not entries:
        return None
    args = SimpleNamespace(pullback_frac=pf, fill_win=fill_win, fwd=fwd, cost=cost,
                            max_pos=1, swap_pct=0.0, exec_split=0, tp1_frac=0.0)
    t, rr_real = walk(df, entries, None, args)
    return t


# ------------------------------------------------------------------ metrics (fixed-bet, size-agnostic)

def fixed_bet_metrics(t):
    """0.01ロット固定運用の前提でサイズ写像を挟まない指標。maxDD はR単位の累積和(複利なし)。"""
    if t is None or len(t) == 0:
        return None
    R = t["R"].values
    n = len(R)
    win = float((R > 0).mean() * 100)
    pos = R[R > 0].sum()
    neg = abs(R[R <= 0].sum())
    pf = float(pos / neg) if neg > 0 else float("inf")
    meanR = float(R.mean())
    totR = float(R.sum())
    cum = np.cumsum(R)
    peak = np.maximum.accumulate(cum)
    maxdd_r = float((peak - cum).max())
    span_years = max((t["time"].iloc[-1] - t["time"].iloc[0]).days / 365.25, 1e-6)
    n_per_year = n / span_years
    yrs = sorted(t["y"].unique())
    half = yrs[len(yrs) // 2] if len(yrs) > 1 else None
    isr = t[t["y"] < half]["R"].mean() if half else t["R"].mean()
    oosr = t[t["y"] >= half]["R"].mean() if half else t["R"].mean()
    return dict(n=n, n_per_year=n_per_year, win=win, pf=pf, meanR=meanR, totR=totR,
                maxdd_r=maxdd_r, IS=float(isr), OOS=float(oosr))


def per_year_table(t):
    rows = []
    for y, g in t.groupby("y"):
        R = g["R"].values
        pos = R[R > 0].sum(); neg = abs(R[R <= 0].sum())
        pf = pos / neg if neg > 0 else float("inf")
        rows.append((int(y), len(g), float((R > 0).mean() * 100), pf, float(R.mean()), float(R.sum())))
    return rows


# ------------------------------------------------------------------ null: random-thinning of the market population

def null_percentile(market_R, n_fill, actual_meanR, reps=N_NULL, rng=RNG):
    Nm = len(market_R)
    if n_fill <= 0 or n_fill > Nm:
        return None, None, None
    means = np.empty(reps)
    for r in range(reps):
        idx = rng.choice(Nm, size=n_fill, replace=False)
        means[r] = market_R[idx].mean()
    pct = float((means < actual_meanR).mean() * 100)
    return pct, float(means.mean()), float(means.std())


# ------------------------------------------------------------------ block bootstrap (win% vs block length), A系最良セルのみ

def block_bootstrap_winrate(t, k_months_list, n_boot=1000, seed=20260721):
    s = t.set_index(pd.DatetimeIndex(t["time"]))["R"]
    months = sorted(s.index.to_period("M").unique())
    nm = len(months)
    by_month = {m: s[s.index.to_period("M") == m] for m in months}
    rng = np.random.default_rng(seed)
    out = {}
    for k_months in k_months_list:
        nblk = int(np.ceil(nm / k_months))
        wins = []
        for _ in range(n_boot):
            starts = rng.integers(0, nm, size=nblk)
            seq = np.concatenate([[(st + j) % nm for j in range(k_months)] for st in starts])
            samp = pd.concat([by_month[months[j]] for j in seq])
            if len(samp) < 5:
                continue
            wins.append(float((samp.values > 0).mean() * 100))
        wins = np.array(wins)
        out[k_months] = (float(np.median(wins)), float(np.percentile(wins, 2.5)),
                          float(np.percentile(wins, 97.5)), len(wins))
    return out


# ------------------------------------------------------------------ main grid driver

def cell_row(k, system, stopk, direction, pf, fill_win, rr, fwd, t, n_market, market_R):
    m = fixed_bet_metrics(t)
    row = dict(k=k, system=system, stopk=stopk if system == "B" else np.nan, direction=direction,
               pf=pf, fill_win=fill_win, rr=rr, fwd=fwd)
    if m is None:
        row.update(n=0, n_per_year=0.0, fill_rate=0.0, win=np.nan, pf_ratio=np.nan, meanR=np.nan,
                    totR=np.nan, maxdd_r=np.nan, IS=np.nan, OOS=np.nan, null_pct=np.nan,
                    null_mean=np.nan, null_std=np.nan)
        return row
    fill_rate = m["n"] / n_market if n_market > 0 else np.nan
    if pf > 0.0 and n_market > 0:
        pct, nmean, nstd = null_percentile(market_R, m["n"], m["meanR"])
    else:
        pct, nmean, nstd = np.nan, np.nan, np.nan
    row.update(n=m["n"], n_per_year=m["n_per_year"], fill_rate=fill_rate, win=m["win"],
               pf_ratio=m["pf"], meanR=m["meanR"], totR=m["totR"], maxdd_r=m["maxdd_r"],
               IS=m["IS"], OOS=m["OOS"], null_pct=pct, null_mean=nmean, null_std=nstd)
    return row


def run_grid(df_long, df_short, fwds, rrs, fill_wins, tag=""):
    rows = []
    trade_cache = {}   # (k, system, stopk, direction, pf, fill_win, rr, fwd) -> trades df
    t0 = _time.time()
    n_done = 0
    for k in KS:
        atrL, bodyL = compute_features(df_long)
        atrS, bodyS = compute_features(df_short)
        s_idx_L = raw_triggers(df_long, atrL, bodyL, k)
        s_idx_S = raw_triggers(df_short, atrS, bodyS, k)
        for system, stopks in (("A", [None]), ("B", [2.0, 2.5])):
            for stopk in stopks:
                for direction, work_df, atr_prev, s_idx in (
                        ("long", df_long, atrL, s_idx_L), ("short", df_short, atrS, s_idx_S)):
                    for rr in rrs:
                        entries = build_entries(work_df, atr_prev, s_idx, system, stopk, rr)
                        for fwd in fwds:
                            # market population (pf=0), computed once per (k,system,stopk,dir,rr,fwd)
                            t_mkt = run_walk(work_df, entries, 0.0, FILL_WINS[0], fwd, cost=0.0)
                            n_market = len(t_mkt) if t_mkt is not None else 0
                            market_R = t_mkt["R"].values if t_mkt is not None else np.array([])
                            row = cell_row(k, system, stopk, direction, 0.0, np.nan, rr, fwd,
                                           t_mkt, n_market, market_R)
                            rows.append(row)
                            trade_cache[(k, system, stopk, direction, 0.0, None, rr, fwd)] = t_mkt
                            n_done += 1
                            for pf in PFS:
                                if pf == 0.0:
                                    continue
                                for fill_win in fill_wins:
                                    t_pf = run_walk(work_df, entries, pf, fill_win, fwd, cost=0.0)
                                    row = cell_row(k, system, stopk, direction, pf, fill_win, rr,
                                                   fwd, t_pf, n_market, market_R)
                                    rows.append(row)
                                    trade_cache[(k, system, stopk, direction, pf, fill_win, rr, fwd)] = t_pf
                                    n_done += 1
    print(f"[grid{tag}] {n_done} cells in {_time.time()-t0:.1f}s", file=sys.stderr)
    return pd.DataFrame(rows), trade_cache


# ------------------------------------------------------------------ 検算 (数値assert)

def run_asserts(df_long):
    print("\n[検算1] 引き金の本数が第1段と一致すること (BTC h1 k=2.0 long)")
    atr_prev, body = compute_features(df_long)
    s_idx = raw_triggers(df_long, atr_prev, body, 2.0)
    span = (df_long.index[-1] - df_long.index[0]).days / 365.25
    n_per_year = len(s_idx) / span
    print(f"  n={len(s_idx)}  N/年={n_per_year:.1f}  (第1段報告: n=701, 76.7本/年)")
    assert len(s_idx) == 701, f"n={len(s_idx)} != 701 (第1段と不一致)"
    assert abs(n_per_year - 76.7) < 0.5, n_per_year
    print("  OK: n=701 と一致")

    print("\n[検算2] pf=0.0 (市場成行) の1本目のトレードRを手計算と突き合わせる"
          " (A系, k=2.0, long, RR=2.0, fwd=20, cost=0)")
    entries = build_entries(df_long, atr_prev, s_idx, "A", None, 2.0)
    t_mkt = run_walk(df_long, entries, 0.0, 200, 20, cost=0.0)
    i0, e0, stop0, tgt0, iorig0 = entries[0]
    h = df_long["high"].values; l = df_long["low"].values; c = df_long["close"].values
    risk0 = e0 - stop0
    R_manual = None
    exit_j = min(i0 + 20, len(c) - 1)
    for j in range(i0 + 1, min(i0 + 1 + 20, len(c))):
        if l[j] <= stop0:
            R_manual = -1.0; break
        if h[j] >= tgt0:
            R_manual = (tgt0 - e0) / risk0; break
    if R_manual is None:
        R_manual = (c[exit_j] - e0) / risk0
    R_engine = t_mkt["R"].iloc[0]
    print(f"  entry_time={t_mkt['time'].iloc[0]}  e={e0:.2f} stop={stop0:.2f} tgt={tgt0:.2f}"
          f"  R手計算={R_manual:+.4f}  R_walk()={R_engine:+.4f}")
    assert abs(R_manual - R_engine) < 1e-9, (R_manual, R_engine)
    print("  OK: 手計算と walk() の1本目Rが一致")

    print("\n[検算3] 同足の損切り優先: pfを深く(0.786)すると「約定足で損切り」の件数が"
          " pf=0.25 より増えること (A系, k=2.0, long, RR=2.0, fwd=20, fill_win=200)")

    def count_fillbar_stopped(pf):
        t = run_walk(df_long, entries, pf, 200, 20, cost=0.0)
        if t is None or len(t) == 0:
            return 0, 0
        idx_pos = df_long.index.get_indexer(pd.DatetimeIndex(t["time"]))
        # stop = e_px - risk (実測列から復元、内部フラグを読まない独立再計算)
        stop_rec = t["e_px"].values - t["risk"].values
        fb_stopped = (l[idx_pos] <= stop_rec)
        return int(fb_stopped.sum()), len(t)

    n25, tot25 = count_fillbar_stopped(0.25)
    n78, tot78 = count_fillbar_stopped(0.786)
    print(f"  pf=0.25: 同足損切り={n25}/{tot25}件 ({100*n25/tot25:.1f}%)"
          f"   pf=0.786: 同足損切り={n78}/{tot78}件 ({100*n78/tot78:.1f}%)")
    assert n78 > n25, (n78, n25)
    print("  OK: pf=0.786 の同足損切り件数が pf=0.25 を上回る（walk()の同足優先が効いている）")


# ------------------------------------------------------------------ cost ladder + $ conversion
#
# 🚨 mirror.invert() は C-p の反転（C=2*max(high)）で、価格の"差"（risk等）はそのまま実価格の
# 差に一致するが、価格の"値そのもの"（e_px）は実価格ではない（mirror.pyのコメントどおり
# 「比率ベースの特徴量は綺麗には鏡像化しない」）。round-trip cost の式 `cost_frac/risk*e_px` は
# e_px（価格の値そのもの）を使うので、ショート側でこの式に反転後のe_pxをそのまま渡すと
# 桁が合わない（Cは全期間の最大高値2倍=巨大な定数、初期の安い時代ほど実価格から乖離が大きい）。
# real_e_px = C - e_px_inv で実価格に戻してから cost を掛ける。C を渡さない場合（ロング）は
# e_px をそのまま実価格として使う。

def cost_ladder_report(work_df, entries, pf, fill_win, rr, fwd, direction, k, system, stopk, C=None):
    t0 = run_walk(work_df, entries, pf, fill_win, fwd, cost=0.0)
    if t0 is None or len(t0) == 0:
        return None
    risk = t0["risk"].values; e_px = t0["e_px"].values; R0 = t0["R"].values
    e_px_real = (C - e_px) if C is not None else e_px      # ショートは実価格に戻す
    rows = []
    for cost in COST_LADDER:
        Rc = R0 - cost / risk * e_px_real
        pos = Rc[Rc > 0].sum(); neg = abs(Rc[Rc <= 0].sum())
        pf_ratio = pos / neg if neg > 0 else float("inf")
        rows.append(dict(cost=cost, meanR=float(Rc.mean()), pf=float(pf_ratio),
                          totR=float(Rc.sum()), win=float((Rc > 0).mean() * 100)))
    return rows


def fixed_dollar_in_R(work_df, entries, pf, fill_win, rr, fwd, years=(2018, 2021, 2025), C=None):
    """固定$25コストが年代別の中央値トレードで何R相当かを換算(BTCは価格比例コストでないため)。
    risk（=risk_inv=risk_real、C-p変換で差は保存される）はそのまま使えるので C は不要
    （e_pxを使わないため）。引数として残すのはインタフェースの対称性のためだけ。"""
    t0 = run_walk(work_df, entries, pf, fill_win, fwd, cost=0.0)
    if t0 is None or len(t0) == 0:
        return {}
    out = {}
    for y in years:
        g = t0[t0["y"] == y]
        if len(g) == 0:
            out[y] = None
            continue
        risk_med = g["risk"].median()
        out[y] = dict(n=len(g), risk_med=float(risk_med), cost25_in_R=float(25.0 / risk_med))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    print("[load] BTC h1 full history")
    df_long = load_mt5_csv(BTC_H1)
    print(f"  n={len(df_long)}  span={df_long.index.min()} .. {df_long.index.max()}")

    if cli.smoke:
        df_long = df_long.loc[:"2019-12-31"]
        print(f"  --smoke: 2019年末までに切り詰め n={len(df_long)}")

    df_short = invert(df_long)

    run_asserts(df_long if not cli.smoke else load_mt5_csv(BTC_H1))  # 検算は常にフルデータで(n=701確認のため)

    # --- スクリーンの通行証 (BTC h1 k=2.0 long, フルデータ) ---
    print("\n[screen] フックの通行証を作成 (BTC h1 k=2.0 long)")
    df_full = df_long if not cli.smoke else load_mt5_csv(BTC_H1)
    atr_prev_full, body_full = compute_features(df_full)
    s_idx_full = raw_triggers(df_full, atr_prev_full, body_full, 2.0)
    o = df_full["open"].values
    screen_entries = []
    for s in s_idx_full:
        if s + 1 >= len(df_full):
            continue
        e = o[s + 1]
        sl = e - atr_prev_full[s]
        screen_entries.append((df_full.index[s + 1], 1, e, sl))
    run_screen(SCREEN, df_full, screen_entries, windows=[1200, 3600])  # h1: 20本=1200分, 60本=3600分

    # --- 主グリッド (fwd=20, RR全5値, fill_win全3値, pf全6値) ---
    print(f"\n[main grid] fwd={FWD_MAIN} 本, RR={RRS}, fill_win={FILL_WINS}, pf={PFS}")
    grid_main, cache_main = run_grid(df_long, df_short, [FWD_MAIN], RRS, FILL_WINS, tag="-main")

    out_csv_main = os.path.join(ROOT, "scratchpad",
                                 "out_atr_spike_pullback_grid_main_smoke.csv" if cli.smoke
                                 else "out_atr_spike_pullback_grid_main.csv")
    grid_main.to_csv(out_csv_main, index=False)
    print(f"[csv] {out_csv_main}  ({len(grid_main)} 行)")

    # --- 副グリッド (fwd=60, RR縮小{2,3,4.5}, fill_win=200のみ) ---
    print(f"\n[sub grid] fwd={FWD_SUB} 本 (副・縮小), RR={RRS_SUB}, fill_win=[{FILL_WIN_SUB}], pf={PFS}")
    grid_sub, cache_sub = run_grid(df_long, df_short, [FWD_SUB], RRS_SUB, [FILL_WIN_SUB], tag="-sub")
    out_csv_sub = os.path.join(ROOT, "scratchpad",
                                "out_atr_spike_pullback_grid_sub_smoke.csv" if cli.smoke
                                else "out_atr_spike_pullback_grid_sub.csv")
    grid_sub.to_csv(out_csv_sub, index=False)
    print(f"[csv] {out_csv_sub}  ({len(grid_sub)} 行)")

    # ================================================================
    # 圧縮レポート: fwd=20, fill_win=200 に固定して pf x RR x (k,system,stopk,direction) を表示
    # ================================================================
    print("\n" + "=" * 100)
    print("圧縮テーブル (fwd=20, fill_win=200): k / 系 / stopk / 方向 / RR ごとに pf を並べる")
    print("=" * 100)
    sub = grid_main[(grid_main.fwd == FWD_MAIN) &
                    ((grid_main.fill_win == 200) | (grid_main.fill_win.isna()))]
    for (k, system, stopk, direction) in sorted(set(zip(sub.k, sub.system,
                                                         sub.stopk.fillna(-1), sub.direction))):
        stopk_disp = None if stopk == -1 else stopk
        cell = sub[(sub.k == k) & (sub.system == system) &
                   (sub.stopk.fillna(-1) == stopk) & (sub.direction == direction)]
        print(f"\n--- k={k} 系={system} stopk={stopk_disp} 方向={direction} ---")
        header = f"  {'RR':>5}{'pf':>7}{'約定率':>8}{'N':>6}{'N/年':>7}{'勝率':>7}{'PF':>8}" \
                 f"{'meanR':>8}{'totR':>9}{'maxDD_R':>9}{'間引き%ile':>11}"
        print(header)
        for rr in sorted(cell.rr.unique()):
            crow = cell[cell.rr == rr].sort_values("pf")
            for _, r in crow.iterrows():
                pf_s = f"{r.pf_ratio:.2f}" if np.isfinite(r.pf_ratio) else "inf"
                nullp = f"{r.null_pct:.0f}" if pd.notna(r.null_pct) else "-"
                print(f"  {r.rr:>5.1f}{r.pf:>7.3f}{100*r.fill_rate:>7.1f}%{int(r.n):>6}"
                      f"{r.n_per_year:>7.1f}{r.win:>6.1f}%{pf_s:>8}{r.meanR:>+8.3f}"
                      f"{r.totR:>+9.1f}{r.maxdd_r:>9.2f}{nullp:>11}")

    # ================================================================
    # 帰無を明確に超えたセル (%ile>=95) の一覧
    # ================================================================
    print("\n" + "=" * 100)
    print("帰無(ランダム間引き)を明確に超えたセル (%ile>=95, fwd=20)")
    print("=" * 100)
    passed = grid_main[(grid_main.null_pct >= 95)].sort_values("null_pct", ascending=False)
    cols = ["k", "system", "stopk", "direction", "pf", "fill_win", "rr", "n", "n_per_year",
            "win", "pf_ratio", "meanR", "totR", "maxdd_r", "fill_rate", "null_pct"]
    if len(passed) == 0:
        print("  (該当なし)")
    else:
        with pd.option_context("display.max_rows", None, "display.width", 220,
                                "display.float_format", lambda x: f"{x:.3f}"):
            print(passed[cols].to_string(index=False))

    # ================================================================
    # 年別内訳: pf=0基準 と 各(k,system,stopk,direction)の最良pf の代表セルのみ
    # ================================================================
    print("\n" + "=" * 100)
    print("年別PF/N内訳 (代表セルのみ: RR=2.0, fwd=20, fill_win=200)")
    print("=" * 100)
    rep_rr = 2.0
    for (k, system, stopk, direction) in sorted(set(zip(sub.k, sub.system,
                                                         sub.stopk.fillna(-1), sub.direction))):
        stopk_key = None if stopk == -1 else stopk
        work_df = df_long if direction == "long" else df_short
        atr_prev, body = compute_features(work_df)
        s_idx = raw_triggers(work_df, atr_prev, body, k)
        entries = build_entries(work_df, atr_prev, s_idx, system, stopk_key, rep_rr)
        cellrows = sub[(sub.k == k) & (sub.system == system) &
                       (sub.stopk.fillna(-1) == stopk) & (sub.direction == direction) &
                       (sub.rr == rep_rr)]
        best = cellrows[cellrows.pf > 0].sort_values("meanR", ascending=False)
        best_pf = best.iloc[0].pf if len(best) else None
        print(f"\n--- k={k} 系={system} stopk={stopk_key} 方向={direction} RR={rep_rr} ---")
        t_mkt = cache_main.get((k, system, stopk_key, direction, 0.0, None, rep_rr, FWD_MAIN))
        if t_mkt is not None and len(t_mkt) > 0:
            print(f"  [pf=0.0 市場成行] 年別:")
            for y, n, win, pfv, meanR, totR in per_year_table(t_mkt):
                pf_s = f"{pfv:.2f}" if np.isfinite(pfv) else "inf"
                print(f"    {y}: n={n:>4} win={win:>5.1f}% PF={pf_s:>6} meanR={meanR:+.3f} totR={totR:+7.1f}")
        if best_pf is not None:
            t_best = cache_main.get((k, system, stopk_key, direction, best_pf, 200, rep_rr, FWD_MAIN))
            if t_best is not None and len(t_best) > 0:
                print(f"  [最良pf={best_pf}] 年別:")
                for y, n, win, pfv, meanR, totR in per_year_table(t_best):
                    pf_s = f"{pfv:.2f}" if np.isfinite(pfv) else "inf"
                    print(f"    {y}: n={n:>4} win={win:>5.1f}% PF={pf_s:>6} meanR={meanR:+.3f} totR={totR:+7.1f}")

    # ================================================================
    # コスト梯子 + $25換算 (A系最良セル代表)
    # ================================================================
    print("\n" + "=" * 100)
    print("コスト梯子 (A系, k=2.0, long, RR=2.0, fwd=20, fill_win=200)")
    print("=" * 100)
    atr_prev_L, body_L = compute_features(df_long)
    s_idx_L2 = raw_triggers(df_long, atr_prev_L, body_L, 2.0)
    entries_A2 = build_entries(df_long, atr_prev_L, s_idx_L2, "A", None, 2.0)
    for pf in [0.0, 0.5]:
        rows = cost_ladder_report(df_long, entries_A2, pf, 200, 2.0, FWD_MAIN, "long", 2.0, "A", None)
        print(f"\n  pf={pf}:")
        if rows:
            for r in rows:
                print(f"    cost={r['cost']:.5f}  meanR={r['meanR']:+.4f}  PF={r['pf']:.2f}"
                      f"  win={r['win']:.1f}%  totR={r['totR']:+.1f}")

    print("\n固定$25が年代別の中央値トレードで何R相当か (A系, k=2.0, long, pf=0.0):")
    conv = fixed_dollar_in_R(df_long, entries_A2, 0.0, 200, 2.0, FWD_MAIN)
    for y, v in conv.items():
        if v is None:
            print(f"  {y}: データ無し")
        else:
            print(f"  {y}: n={v['n']}  risk中央値=${v['risk_med']:.1f}  $25={v['cost25_in_R']:.4f}R相当")

    # ================================================================
    # 巡回ブロック・ブートストラップ (A系最良セル、勝率のブロック長依存)
    # ================================================================
    print("\n" + "=" * 100)
    print("巡回ブロック・ブートストラップ: 勝率のブロック長依存 (A系, k=2.0, long, RR=2.0, fwd=20)")
    print("=" * 100)
    a2_cells = sub[(sub.k == 2.0) & (sub.system == "A") & (sub.direction == "long") &
                   (sub.rr == 2.0) & (sub.pf > 0)].sort_values("meanR", ascending=False)
    if len(a2_cells) > 0:
        best_row = a2_cells.iloc[0]
        best_pf_a = best_row.pf
        t_best_a = cache_main.get((2.0, "A", None, "long", best_pf_a, 200, 2.0, FWD_MAIN))
        if t_best_a is not None and len(t_best_a) >= 20:
            print(f"  最良セル: pf={best_pf_a}  meanR={best_row.meanR:+.3f}  win={best_row.win:.1f}%"
                  f"  n={int(best_row.n)}")
            bb = block_bootstrap_winrate(t_best_a, [1, 3, 6, 12])
            for kmo, (med, lo, hi, nb) in bb.items():
                print(f"    ブロック{kmo:>2}か月: 勝率中央値={med:.1f}%  95%CI=[{lo:.1f},{hi:.1f}]  (有効draw={nb}/1000)")
        else:
            print("  最良セルのトレード数が不足（--smoke等）でブロックブートストラップをスキップ")
    else:
        print("  該当セル無し（pf>0の約定が無い）")

    print(f"\n実行コマンド: .venv/bin/python scratchpad/atr_spike_pullback_btc_h1.py"
          f"{' --smoke' if cli.smoke else ''}")


if __name__ == "__main__":
    main()
