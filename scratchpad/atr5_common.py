"""共有ヘルパー ― 【仕様カード第5段】BTC h1 ATR拡大足の3実験（(a)出口対決 / (b)フィルタ検定 / (c)トレール正典化）で
共通に使う関数だけを置く置き場（トレード統計は出さないのでスクリーンゲート対象外）。

執行は必ず src.engine.walk.walk()（ロング）と src.engine.mirror.invert()（ショート）を使う。
自前の前方走査ループはここにもどこにも書かない（カード(c)でwalk()にtrail_atrを実装済み）。

損益は必ず「入口価格に対する%」で計算する（R では測らない、CLAUDE.md 反証 r-unit-pullback-inflation）。
walk() は cost=0.0 で呼び、コストは外側で実価格から引く（mirror-cost-overcharge 回避）。
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pandas_ta as ta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_loader import load_mt5_csv          # noqa: E402
from src.engine.walk import walk                  # noqa: E402
from src.engine.mirror import invert               # noqa: E402

BTC_H1 = f"{ROOT}/data/vantage_btcusd_h1.csv"
COST = 0.0005          # 往復コスト割合（凍結カードの規約値）


# ------------------------------------------------------------------ data / features

def load_frames():
    """ロングは素の df、ショートは mirror.invert() した反転フレーム。C は実価格復元用の定数。"""
    df = load_mt5_csv(BTC_H1)
    inv = invert(df)
    C = 2 * df["high"].max()
    return df, inv, C


def atr_prev_of(d, n=14):
    """ATR(n)[s-1]。ta.atr は engine の trail_atr と同じ実装（pandas_ta）＝定義を一致させる。"""
    return ta.atr(d["high"], d["low"], d["close"], length=n).shift(1).to_numpy()


def raw_triggers(d, atr_prev, k):
    """引き金 s の配列：実体 > ATR[s-1]*k かつ陽線（第1段・凍結カードと同じ定義）。"""
    o, c = d["open"].to_numpy(), d["close"].to_numpy()
    hit = (c - o > atr_prev * k) & (c > o) & np.isfinite(atr_prev)
    s = np.flatnonzero(hit)
    return s[s + 1 < len(d)]


# ------------------------------------------------------------------ entries (A系 / B系)

def build_entries(d, atr_prev, s_idx, system, rr, stopk=2.0):
    """entries = (i, e, stop, tgt, i_origin)。i=引き金足s、e=open[s+1]。
    A系: stop = 拡大足(足s)の安値。 B系: stop = e - stopk*ATR[s-1]（目標が届く形、既定stopk=2.0）。
    risk<=0 は除外。"""
    o, l = d["open"].to_numpy(), d["low"].to_numpy()
    ent = []
    for s in s_idx:
        e = o[s + 1]
        stop = l[s] if system == "A" else e - stopk * atr_prev[s]
        risk = e - stop
        if risk <= 0:
            continue
        ent.append((s, e, stop, e + rr * risk, s))
    return ent


# ------------------------------------------------------------------ engine call + % pnl

def run_cell(d, entries, pf, fill_win, fwd, trail_atr=0.0, trail_n=14, C=None, cost=COST,
             max_pos=1):
    """walk() を cost=0 で呼び、外側で価格%コストを引く。C を渡すとショート（反転フレーム）扱いで
    実価格に戻してからコストを掛ける（mirror-cost-overcharge 回避）。"""
    if not entries:
        return None
    args = SimpleNamespace(pullback_frac=pf, fill_win=fill_win, fwd=fwd, cost=0.0,
                            max_pos=max_pos, swap_pct=0.0, tp1_frac=0.0, exec_split=0,
                            trail_atr=trail_atr, trail_n=trail_n)
    t, _ = walk(d, entries, None, args)
    if t is None or len(t) == 0:
        return None
    e_real = (C - t["e_px"]) if C is not None else t["e_px"]
    pnl_px = t["R"] * t["risk"] - cost * e_real
    return t.assign(pnl_px=pnl_px, pnl_pct=pnl_px / e_real, y=t["time"].dt.year)


# ------------------------------------------------------------------ metrics (price-% based, fixed-bet)

def pf_of(p):
    w, l = p[p > 0].sum(), -p[p < 0].sum()
    return float(w / l) if l > 0 else float("nan")


def stats(t, span_years):
    p = t["pnl_pct"].to_numpy()
    n = len(p)
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max()) * 100 if n else np.nan
    return dict(N=n, N_yr=n / span_years, win=float((p > 0).mean() * 100), PF=pf_of(p),
                mean_pct=float(p.mean() * 100), tot_pct=float(p.sum() * 100), maxDD_pct=dd)


def per_year(t):
    rows = []
    for y, g in t.groupby("y"):
        p = g["pnl_pct"].to_numpy()
        rows.append(dict(year=int(y), N=len(p), win=float((p > 0).mean() * 100), PF=pf_of(p),
                          mean_pct=float(p.mean() * 100), tot_pct=float(p.sum() * 100)))
    return rows


def fmt_row(label, s, null_pf=None, null_mean=None):
    pf_s = f"{s['PF']:.2f}" if np.isfinite(s['PF']) else "inf"
    npf = f"{null_pf:.0f}" if null_pf is not None and np.isfinite(null_pf) else "-"
    nmn = f"{null_mean:.0f}" if null_mean is not None and np.isfinite(null_mean) else "-"
    return (f"{label:<46} N={s['N']:>5} N/年={s['N_yr']:>6.1f} 勝率={s['win']:>5.1f}% "
            f"PF={pf_s:>6} 平均%={s['mean_pct']:>+7.3f} 総%={s['tot_pct']:>+8.1f} "
            f"maxDD%={s['maxDD_pct']:>6.1f} | 帰無%ile(PF,平均)=({npf},{nmn})")


# ------------------------------------------------------------------ random-thinning null

def drop_null(pool_pct, n_fill, obs_mean, obs_pf, reps=400, seed=20260721):
    """成行(pf=0・フィルタ無し)母集団から同じ本数だけランダムに残す帰無。全て価格%単位。"""
    rng = np.random.default_rng(seed)
    Np = len(pool_pct)
    if n_fill <= 0 or n_fill > Np:
        return dict(pf_pct=np.nan, mean_pct=np.nan, null_pf_med=np.nan, null_mean_med=np.nan)
    means, pfs = np.empty(reps), np.empty(reps)
    for r in range(reps):
        s = rng.choice(pool_pct, size=n_fill, replace=False)
        means[r] = s.mean()
        pfs[r] = pf_of(s)
    return dict(pf_pct=float((pfs < obs_pf).mean() * 100),
                mean_pct=float((means < obs_mean).mean() * 100),
                null_pf_med=float(np.nanmedian(pfs)), null_mean_med=float(np.median(means) * 100))


# ------------------------------------------------------------------ cyclic block bootstrap (monthly blocks)

def block_bootstrap(t, k_months_list, metric="mean", n_boot=1000, seed=20260721):
    """巡回ブロック・ブートストラップ。metric='mean'(平均%) か 'pf'。月ブロックを丸ごと動かす
    （月内のトレード列は保つ）。返り値: {k: (median, lo2.5, hi97.5, n_eff)}"""
    s = t.set_index(pd.DatetimeIndex(t["time"]))["pnl_pct"]
    months = sorted(s.index.to_period("M").unique())
    nm = len(months)
    by_month = {m: s[s.index.to_period("M") == m] for m in months}
    rng = np.random.default_rng(seed)
    out = {}
    for km in k_months_list:
        nblk = int(np.ceil(nm / km))
        vals = []
        for _ in range(n_boot):
            starts = rng.integers(0, nm, size=nblk)
            seq = np.concatenate([[(st + j) % nm for j in range(km)] for st in starts])
            samp = pd.concat([by_month[months[j]] for j in seq])
            if len(samp) < 5:
                continue
            v = samp.mean() * 100 if metric == "mean" else pf_of(samp.to_numpy())
            vals.append(v)
        vals = np.array([v for v in vals if np.isfinite(v)])
        if len(vals) == 0:
            out[km] = (np.nan, np.nan, np.nan, 0)
            continue
        out[km] = (float(np.median(vals)), float(np.percentile(vals, 2.5)),
                    float(np.percentile(vals, 97.5)), len(vals))
    return out


# ------------------------------------------------------------------ direction settings shared by (a) and (b)
# (k, side, system, rr_anchor, pf) ―― pf はショート側の凍結済み最良指値、ロングは成行(pf=0)
DIRECTIONS = [
    dict(name="long k2.0 RR3",  k=2.0, side="long",  rr=3.0, pf=0.0),
    dict(name="long k1.5 RR4.5", k=1.5, side="long",  rr=4.5, pf=0.0),
    dict(name="short k2.0 RR3 pf0.5",   k=2.0, side="short", rr=3.0, pf=0.5),
    dict(name="short k1.5 RR4.5 pf0.382", k=1.5, side="short", rr=4.5, pf=0.382),
]


def span_years(d):
    return (d.index[-1] - d.index[0]).days / 365.25


# ------------------------------------------------------------------ (b) session / pdh_dist filters

def session_hour(d, s_idx):
    """引き金足sのブローカー時刻（0-23）。フィルタは 0<=hour<8 をアジアとする。"""
    return d.index[s_idx].hour.to_numpy()


def build_pdh_dist_series(d, atr_prev):
    """pdh_dist[s] = (close[s] - 前日高値) / ATR[s-1]。前日高値は暦日(ブローカー時刻)の
    resample("1D").max() を1日shiftしたもの＝当日中は既知の過去値のみ使う（先読み無し）。
    ショート（mirror.invert()した反転フレーム）にも同じ式をそのまま適用する
    （ロングの「高値」がミラーでは実物の「安値」の鏡像になり、正しく前日安値相当になる。
    mirror.py の規約どおり「反転フレームにロングと同じ式を適用するだけで正しい鏡像になる」）。
    """
    daily_high = d["high"].resample("1D").max()
    prev_daily_high = daily_high.shift(1).ffill()
    day_key = d.index.normalize()
    pdh = prev_daily_high.reindex(day_key)
    pdh.index = d.index
    close = d["close"]
    return ((close - pdh) / atr_prev).to_numpy()


def check_no_lookahead_pdh(d, atr_prev, s_idx, cut_frac=0.5):
    """先読み検査: 末尾を切り落としても、切った点より前の pdh_dist が変わらないこと。"""
    full = build_pdh_dist_series(d, atr_prev)
    cut = int(len(d) * cut_frac)
    d_trim = d.iloc[:cut]
    atr_trim = atr_prev[:cut]
    trimmed = build_pdh_dist_series(d_trim, atr_trim)
    n = min(len(trimmed), cut)
    a = full[:n]
    b = trimmed[:n]
    ok_mask = np.isfinite(a) & np.isfinite(b)
    return bool(np.allclose(a[ok_mask], b[ok_mask], equal_nan=True)), int(ok_mask.sum())
