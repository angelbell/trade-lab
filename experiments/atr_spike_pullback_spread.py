"""atr_spike_pullback_btc_h1.py の追補: 買い指値のASK基準感度（見積り）。

仕様カード: 「買い指値はASK基準（BID足では 指値-スプレッド でしか約定しない）。walk()はこれを
持たない=わずかに楽観的。A系の最良セルについてだけ、指値を1スプレッド分不利にした感度を出す」。
「買い指値」なので対象は LONG 側のみ（ショート側はミラー空間の"buy"=実世界のsell limitで、
ASK基準の議論はそのままでは適用できない）。

walk()を改造せず、フィル判定だけを再現する軽量な独立実装（見積り・spec許容の簡易版）:
  通常: l[j] <= lim で約定
  スプレッド込み: l[j] <= lim - spread で約定 (BID足でASKベースの指値が満たされる条件)
スプレッドは CLAUDE.md の BTC 実測レンジ($10-25)の中央 $15 を使う。

SCREEN = "atr_spike_btc_h1"

Run: .venv/bin/python experiments/atr_spike_pullback_spread.py 2>&1 | tee experiments/out_atr_spike_pullback_spread.txt
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from atr_spike_pullback_btc_h1 import (
    load_mt5_csv, BTC_H1, compute_features, raw_triggers, build_entries, FWD_MAIN, run_walk
)

SPREAD = 15.0  # $ (CLAUDE.md: BTC実コスト canon $15)

df_long = load_mt5_csv(BTC_H1)
atr_prev, body = compute_features(df_long)
s_idx = raw_triggers(df_long, atr_prev, body, 1.5)
entries = build_entries(df_long, atr_prev, s_idx, "A", None, 4.5)  # 長側flagship: k=1.5, RR=4.5

h, l, c = df_long["high"].values, df_long["low"].values, df_long["close"].values


def fill_scan(entries, pf, fill_win, spread):
    """walk()のmax_pos=1スロット排他(busy_until相当)を再現する。これを入れないと
    重複ポジションが二重計上され、真のflagship(walk()本体)の数字と大きく食い違う
    (最初の実装ミスで発覚: 見積りとはいえ engine の主要な排他ロジックは省略できない)。"""
    n_fill, n_total = 0, 0
    Rs = []
    open_x = []
    for (i, e, stop, tgt, i_origin) in entries:
        open_x = [x for x in open_x if x >= i]
        if len(open_x) >= 1:
            continue
        n_total += 1
        lim = e - pf * (e - stop)
        fj = None
        for j in range(i + 1, min(i + 1 + fill_win, len(c))):
            if h[j] >= tgt:
                break
            if l[j] <= lim - spread:
                fj = j
                break
        if fj is None:
            continue
        n_fill += 1
        e_px = lim  # 約定価格は指値レートそのもの（スプレッドはフィル判定にのみ影響）
        risk = e_px - stop
        if risk <= 0:
            continue
        reward = tgt - e_px
        R = None
        exit_j = min(fj + FWD_MAIN, len(c) - 1)
        fill_bar_stopped = l[fj] <= stop
        if fill_bar_stopped:
            R, exit_j = -1.0, fj
        else:
            for jj in range(fj + 1, min(fj + 1 + FWD_MAIN, len(c))):
                if l[jj] <= stop:
                    R, exit_j = -1.0, jj; break
                if h[jj] >= tgt:
                    R, exit_j = reward / risk, jj; break
            if R is None:
                R = (c[exit_j] - e_px) / risk
        Rs.append(R)
        open_x.append(exit_j)
    return n_fill, n_total, np.array(Rs)


for pf, fill_win in [(0.618, 200)]:
    print(f"=== long A系 k=1.5 RR=4.5 pf={pf} fill_win={fill_win} (flagship) ===")

    print("  [検算] spread=0 でこの簡易再実装が walk()本体と一致することを確認")
    nf0, nt0, Rs0 = fill_scan(entries, pf, fill_win, 0.0)
    t_true = run_walk(df_long, entries, pf, fill_win, FWD_MAIN, cost=0.0)
    print(f"    簡易再実装: n_fill={nf0}  meanR={Rs0.mean():+.4f}")
    print(f"    walk()本体: n_fill={len(t_true)}  meanR={t_true['R'].mean():+.4f}")
    assert nf0 == len(t_true), (nf0, len(t_true))
    assert abs(Rs0.mean() - t_true["R"].mean()) < 1e-6, (Rs0.mean(), t_true["R"].mean())
    print("    OK: 簡易再実装は walk() 本体と n・meanR ともビット一致 -> 以下のspread比較は信頼できる")

    for spread in [0.0, SPREAD]:
        nf, nt, Rs = fill_scan(entries, pf, fill_win, spread)
        meanR = Rs.mean() if len(Rs) else float("nan")
        pos = Rs[Rs > 0].sum(); neg = abs(Rs[Rs <= 0].sum())
        pfr = pos / neg if neg > 0 else float("inf")
        win = (Rs > 0).mean() * 100 if len(Rs) else float("nan")
        print(f"  spread=${spread:>5.1f}  約定={nf}/{nt} ({100*nf/nt:.1f}%)  n={len(Rs)}"
              f"  win={win:.1f}%  PF={pfr:.2f}  meanR={meanR:+.3f}")
