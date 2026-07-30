"""押し目指値の M1 解像度判定（M1フィードの定数オフセットを実測補正した版）。

背景: gold の h1/m15/m5 正典CSVは 2026-07-23 に XAUUSD+ から再取得済みだが、m1 は 2026-07-19 に
旧 XAUUSD から取得したままで、約 $0.05 の定数オフセットがある（CLAUDE.md 記載）。
前回の M1 判定は未補正のまま走らせたため、ブレイク検出（M1高値 >= buy_lvl）が不当に難しくなり
「同足で押し目に届いた」判定が保守側へ歪んでいた可能性がある。ここでは
  1. M1 を H1 に集約して高値・安値の差の中央値からオフセットを実測し、
  2. 差し引いた上で ±$0.01 / ±$0.05 の一致率を再点検し、
  3. その補正済み M1 で「ブレイクの瞬間より後の押しだけを約定に数える」判定を引き直す。

モード:
  same_ng          … 約定足での指値約定を一切認めない（最も保守的な下限）
  m1_after         … M1でブレイク足を特定し、その足以降の安値だけ約定に数える
  m1_after_strict  … ブレイク足自身も除き、翌M1足以降だけ数える（タダ乗りの完全排除）
出口はトレール（20本安値）でベースラインと同一。コストは apply_cost(0.3, 0.1)。
spread は買い指値がASK基準であることの反映（low <= limit - spread で初めて約定）。
"""
SCREEN = "donchian20_sar_gold_h1"

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
import pandas as pd

from src.data_loader import load_mt5_csv
from donchian20_sar_gold import (
    LENGTH, START, load_tf, apply_cost, years_span, max_drawdown, summarize,
)
from donchian20_long_gate_gold import (
    build_daily_gates, align_bool, pf_pct, year_breakdown_pct,
    daily_pnl_series, block_bootstrap_diff,
)
from donchian20_fixedtgt_gold import (
    simulate_long_exit, CENTRAL, GUARD, bootstrap_dd_median,
)

RNG_SEED = 20260730
M1_START = "2019-01-01"          # ブローカーのm1保持は2019年から
FRACS = [0.05, 0.10, 0.15, 0.20, 0.25]
EXPIRIES = [5, 20]
MODES = ["m1_after", "m1_after_strict", "same_ng"]
SPREAD_MAIN = 0.15
DD_TARGET = 15.0                 # 中央値DDをこの%に揃えて賭け率fを決める
N_BOOT = 400
BLOCK_MONTHS = [1, 3, 6, 12]


# ----------------------------------------------------------------------
# M1 オフセットの実測と補正
# ----------------------------------------------------------------------

def load_m1_corrected(h1):
    m1 = load_mt5_csv("data/vantage_xauusd_m1.csv")
    m1 = m1.loc[m1.index >= pd.Timestamp(M1_START, tz=m1.index.tz)]
    # H1 バーごとに M1 を集約して比較
    bnd = m1.index.searchsorted(h1.index, side="left")
    ends = np.append(bnd[1:], len(m1))
    h1h, h1l = h1["high"].values, h1["low"].values
    m1h, m1l = m1["high"].values, m1["low"].values
    agg_h = np.full(len(h1), np.nan)
    agg_l = np.full(len(h1), np.nan)
    for i in range(len(h1)):
        a, b = bnd[i], ends[i]
        if b > a:
            agg_h[i] = m1h[a:b].max()
            agg_l[i] = m1l[a:b].min()
    cov = np.isfinite(agg_h)
    print(f"[M1点検] H1被覆 = {cov.mean()*100:.1f}%  (被覆本数 {cov.sum()}/{len(h1)})")
    off_h = float(np.median((agg_h - h1h)[cov]))
    off_l = float(np.median((agg_l - h1l)[cov]))
    OFF = float(np.median([off_h, off_l]))
    print(f"[M1点検] 補正前オフセット中央値: 高値 {off_h:+.4f} / 安値 {off_l:+.4f} → 採用 OFF = {OFF:+.4f}")
    for tol in (0.01, 0.05):
        r0 = float((np.abs(agg_h - h1h)[cov] <= tol).mean() * 100)
        r1 = float((np.abs(agg_h - OFF - h1h)[cov] <= tol).mean() * 100)
        l0 = float((np.abs(agg_l - h1l)[cov] <= tol).mean() * 100)
        l1 = float((np.abs(agg_l - OFF - h1l)[cov] <= tol).mean() * 100)
        print(f"[M1点検] 一致率(±${tol:.2f}) 高値 補正前 {r0:5.1f}% → 補正後 {r1:5.1f}% / "
              f"安値 補正前 {l0:5.1f}% → 補正後 {l1:5.1f}%")
    m1c = m1.copy()
    for c in ("open", "high", "low", "close"):
        m1c[c] = m1c[c] - OFF
    return m1c, OFF


# ----------------------------------------------------------------------
# 押し目指値ウォーカー（出口はトレール、入口だけ指値化）
# ----------------------------------------------------------------------

def simulate_limit(df, frac, expiry, mode, gate_arr=None, spread=0.0,
                   m1_bnd=None, m1_h=None, m1_l=None):
    hh_lvl = df["high"].rolling(LENGTH).max().shift(1).values
    ll_lvl = df["low"].rolling(LENGTH).min().shift(1).values
    o, h, l = df["open"].values, df["high"].values, df["low"].values
    idx = df.index
    n = len(df)

    def sell_fill(i, lvl):
        return o[i] if o[i] <= lvl else lvl

    trades = []
    state = 0                    # 0 待機 / 1 指値発注中 / 2 建玉
    sig_i = limit = entry_i = entry_price = None
    n_sig = n_fill_sig = n_nobr = n_cancel = 0

    def try_close(i):
        nonlocal state, entry_i, entry_price
        cur_stop = ll_lvl[i]
        if not (l[i] <= cur_stop):
            return
        trades.append(dict(
            entry_time=idx[entry_i], exit_time=idx[i], direction=1,
            entry_raw=entry_price, exit_raw=sell_fill(i, cur_stop),
            bars_held=i - entry_i, entry_i=entry_i, exit_i=i, exit_reason="trail",
        ))
        state, entry_i, entry_price = 0, None, None

    def sigbar_fill_ok(i, buy_lvl, lim):
        """約定足そのもので指値に届いたか。ブレイクの瞬間より後の安値だけを数える。"""
        nonlocal n_nobr
        if mode == "same_ng":
            return False
        a, b = m1_bnd[i], m1_bnd[i + 1] if i + 1 < len(m1_bnd) else len(m1_h)
        if b <= a:
            n_nobr += 1
            return False
        br = np.where(m1_h[a:b] >= buy_lvl)[0]
        if len(br) == 0:
            n_nobr += 1
            return False
        st = a + br[0] + (1 if mode == "m1_after_strict" else 0)
        if st >= b:
            return False
        return bool((m1_l[st:b] <= lim - spread).any())

    for i in range(n):
        if np.isnan(hh_lvl[i]) or np.isnan(ll_lvl[i]):
            continue
        if state == 2:
            try_close(i)
            continue
        if state == 1:
            if (i - sig_i) > expiry:
                state, n_cancel = 0, n_cancel + 1
            else:
                trg = None
                if o[i] <= limit - spread:
                    trg = o[i]
                elif l[i] <= limit - spread:
                    trg = limit
                if trg is not None and ll_lvl[i] < trg:
                    state, entry_i, entry_price = 2, i, trg
                    try_close(i)
                continue
        if state == 0:
            buy_lvl = hh_lvl[i]
            if h[i] >= buy_lvl and (gate_arr is None or bool(gate_arr[i])):
                n_sig += 1
                lo_sig = ll_lvl[i]
                lim = buy_lvl - frac * (buy_lvl - lo_sig)
                if sigbar_fill_ok(i, buy_lvl, lim) and lo_sig < lim:
                    state, entry_i, entry_price = 2, i, lim
                    n_fill_sig += 1
                    try_close(i)
                else:
                    state, sig_i, limit = 1, i, lim

    diag = dict(n_sig=n_sig, n_fill_sig=n_fill_sig, n_nobr=n_nobr, n_cancel=n_cancel)
    return trades, diag


# ----------------------------------------------------------------------
# 評価
# ----------------------------------------------------------------------

def evaluate(trades_c, yrs, date_index, rng):
    s = summarize(trades_c, yrs)
    if s is None:
        return None
    daily = daily_pnl_series(trades_c, date_index)
    dds = bootstrap_dd_median(daily, 3, N_BOOT, rng)
    bdd = float(np.median(dds))
    f = DD_TARGET / bdd if bdd > 0 else np.nan
    pnl = np.array([t["pnl_pct"] for t in trades_c])
    mult = float(np.prod(1 + f * pnl / 100)) if np.isfinite(f) else np.nan
    comp = (mult ** (1 / yrs) - 1) * 100 if (np.isfinite(mult) and mult > 0) else np.nan
    return dict(n=s["n"], n_per_year=s["n_per_year"], pf=pf_pct(trades_c),
                win=s["win_pct"], ann=s["tot_pct_per_year"], dd=s["maxdd_pct"],
                ratio=s["tot_pct_per_year"] / s["maxdd_pct"] if s["maxdd_pct"] > 0 else np.nan,
                bdd=bdd, f=f, mult=mult, comp=comp, daily=daily, trades=trades_c)


def row(label, r, extra=""):
    if r is None:
        print(f"  {label:34s}  (トレード無し)")
        return
    print(f"  {label:34s} n={r['n']:4d} ({r['n_per_year']:5.1f}/年) PF%={r['pf']:5.2f} "
          f"勝率={r['win']:5.1f}% 年率={r['ann']:6.2f}% DD={r['dd']:6.2f}% 年率/DD={r['ratio']:5.2f} "
          f"bootDD={r['bdd']:5.2f} f={r['f']:5.2f} 倍率={r['mult']:7.2f} 複利={r['comp']:6.1f}% {extra}")


def main():
    rng = np.random.default_rng(RNG_SEED)
    h1_full = load_tf("h1")
    gates = build_daily_gates(h1_full)

    # --- 番人: frac=0（成行）でベースラインを再現できるか（全窓 2018-10-01→） ---
    tr, _, _ = simulate_long_exit(h1_full, "trail")
    trc = apply_cost(tr, *CENTRAL)
    yf = years_span(h1_full)
    sf = summarize(trc, yf)
    g = GUARD["h1"]
    ok = (sf["n"] == g["n"] and abs(sf["tot_pct_per_year"] - g["ann_pct"]) < 0.05
          and abs(sf["maxdd_pct"] - g["maxdd_pct"]) < 0.05)
    print(f"[番人] 全窓ベースライン n={sf['n']} 年率={sf['tot_pct_per_year']:.2f}% "
          f"DD={sf['maxdd_pct']:.2f}% → {'PASS' if ok else 'FAIL'} (期待 n={g['n']} "
          f"年率={g['ann_pct']}% DD={g['maxdd_pct']}%)")
    assert ok, "ベースライン不一致"

    # --- M1窓に切って全部を揃える ---
    h1 = h1_full.loc[h1_full.index >= pd.Timestamp(M1_START, tz=h1_full.index.tz)]
    yrs = years_span(h1)
    dates = pd.DatetimeIndex(sorted(set(h1.index.normalize())))
    print(f"\n窓 = {h1.index[0]} → {h1.index[-1]}  ({yrs:.2f}年, H1 {len(h1)}本)")

    m1, OFF = load_m1_corrected(h1)
    m1_bnd = np.append(m1.index.searchsorted(h1.index, side="left"), len(m1))
    m1_h, m1_l = m1["high"].values, m1["low"].values

    gate_arrs = {"ゲート無し": None, "SMA150": align_bool(gates["sma_assigned"], h1.index)}

    base = {}
    for gname, garr in gate_arrs.items():
        tr, _, _ = simulate_long_exit(h1, "trail", gate_arr=garr)
        base[gname] = evaluate(apply_cost(tr, *CENTRAL), yrs, dates, rng)

    for gname, garr in gate_arrs.items():
        print(f"\n{'='*118}\n### {gname}  (spread=${SPREAD_MAIN:.2f})\n{'='*118}")
        row("成行（基準）", base[gname])
        for mode in MODES:
            print(f"  --- {mode} ---")
            for expiry in EXPIRIES:
                for frac in FRACS:
                    trs, diag = simulate_limit(h1, frac, expiry, mode, gate_arr=garr,
                                               spread=SPREAD_MAIN, m1_bnd=m1_bnd,
                                               m1_h=m1_h, m1_l=m1_l)
                    r = evaluate(apply_cost(trs, *CENTRAL), yrs, dates, rng)
                    fillp = (r["n"] / diag["n_sig"] * 100) if (r and diag["n_sig"]) else 0.0
                    row(f"frac={frac:.2f} 期限{expiry:2d}本", r,
                        extra=f"約定率={fillp:4.1f}% 同足約定={diag['n_fill_sig']:3d} "
                              f"M1未検出={diag['n_nobr']:3d}")

    # --- 主要セルの年別＋月ブロック・ブートストラップ ---
    print(f"\n{'='*118}\n### 主要セルの年別と月ブロック（strict のみ）\n{'='*118}")
    for gname, garr in gate_arrs.items():
        for frac in (0.10, 0.15):
            trs, diag = simulate_limit(h1, frac, 5, "m1_after_strict", gate_arr=garr,
                                       spread=SPREAD_MAIN, m1_bnd=m1_bnd, m1_h=m1_h, m1_l=m1_l)
            r = evaluate(apply_cost(trs, *CENTRAL), yrs, dates, rng)
            if r is None:
                continue
            b = base[gname]
            print(f"\n[{gname} / frac={frac:.2f} / 期限5本 / m1_after_strict]")
            row("指値", r)
            row("成行", b)
            yl = {d["year"]: d for d in year_breakdown_pct(r["trades"])}
            yb = {d["year"]: d for d in year_breakdown_pct(b["trades"])}
            wins = 0
            tot = 0
            for y in sorted(set(yl) | set(yb)):
                a = yb.get(y, dict(n=0, tot_pct=0.0))
                c = yl.get(y, dict(n=0, tot_pct=0.0))
                tot += 1
                if c["tot_pct"] > a["tot_pct"]:
                    wins += 1
                print(f"    {y}: 成行 n={a['n']:3d} {a['tot_pct']:+7.2f}%  |  "
                      f"指値 n={c['n']:3d} {c['tot_pct']:+7.2f}%  "
                      f"{'指値勝ち' if c['tot_pct'] > a['tot_pct'] else ''}")
            print(f"    指値が勝った年: {wins}/{tot}")
            for bm in BLOCK_MONTHS:
                d = block_bootstrap_diff(b["daily"], r["daily"], bm, 400, rng)
                if len(d):
                    print(f"    月ブロック{bm:2d}: 指値が優る割合 = {(d > 0).mean()*100:5.1f}%  "
                          f"(年率/DD差の中央値 {np.median(d):+.3f})")

    # --- spread 感度（frac=0.15 / strict） ---
    print(f"\n{'='*118}\n### spread 感度（frac=0.15 / 期限5本 / m1_after_strict）\n{'='*118}")
    for gname, garr in gate_arrs.items():
        for sp in (0.0, 0.15, 0.35):
            trs, diag = simulate_limit(h1, 0.15, 5, "m1_after_strict", gate_arr=garr,
                                       spread=sp, m1_bnd=m1_bnd, m1_h=m1_h, m1_l=m1_l)
            r = evaluate(apply_cost(trs, *CENTRAL), yrs, dates, rng)
            row(f"{gname} spread=${sp:.2f}", r)

    print(f"\n[注記] M1オフセット補正量 OFF = {OFF:+.4f} を M1 の O/H/L/C から差し引いて判定した。")


if __name__ == "__main__":
    main()
