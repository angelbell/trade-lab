"""共有ヘルパー ―【仕様カード第7段】USDJPY h1 ATR拡大足の掃引で共通に使う関数だけを置く置き場
（トレード統計は出さないのでスクリーンゲート対象外）。

執行は必ず src.engine.walk.walk()（ロング）と src.engine.mirror.invert()（ショート）を使う。
自前の前方走査ループはここにもどこにも書かない。

損益は必ず「入口価格に対する%」で計算する（Rでは測らない）。walk() は cost=0.0 で呼び、
コストは外側で実価格から引く（mirror-cost-overcharge 回避、ショートは e_real=C-e_pxを使う）。

レジーム（メイン判定で採用・ロング側のみ使用）:
  daily_up  = 前日終値 > 前日確定の日足SMA200
  weekly_up = 前週終値 > 前週確定の30週SMA（週足はさらに1週ラグしてh1にffill）
両方 shift(1) 済みの確定値のみを使う（先読み無し、検算つき）。
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pandas_ta as ta

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT) if os.path.basename(ROOT) == "experiments" else ROOT
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_loader import load_mt5_csv          # noqa: E402
from src.engine.walk import walk                  # noqa: E402
from src.engine.mirror import invert               # noqa: E402

USDJPY_H1 = f"{ROOT}/data/vantage_usdjpy_h1.csv"
USDJPY_D1 = f"{ROOT}/data/vantage_usdjpy_d1.csv"
USDJPY_W1 = f"{ROOT}/data/vantage_usdjpy_w1.csv"
COST = 0.009           # 往復コスト＝絶対値(円) = 0.9pip（価格に対する割合ではない、FXはpip建て定額）


# ------------------------------------------------------------------ data / features

def load_frames(start="2000-01-01"):
    """ロングは素の df、ショートは mirror.invert() した反転フレーム。C は実価格復元用の定数。"""
    df = load_mt5_csv(USDJPY_H1).loc[start:]
    inv = invert(df)
    C = 2 * df["high"].max()
    return df, inv, C


def atr_prev_of(d, n=14):
    """Wilder ATR(n)[s-1]。ta.atr は engine の trail_atr と同じ実装＝定義を一致させる。"""
    return ta.atr(d["high"], d["low"], d["close"], length=n).shift(1).to_numpy()


def raw_triggers(d, atr_prev, k):
    """引き金 s の配列：実体 > ATR[s-1]*k かつ陽線（凍結カードと同じ定義）。"""
    o, c = d["open"].to_numpy(), d["close"].to_numpy()
    hit = (c - o > atr_prev * k) & (c > o) & np.isfinite(atr_prev)
    s = np.flatnonzero(hit)
    return s[s + 1 < len(d)]


# ------------------------------------------------------------------ regime (long-side only)

def daily_up_regime(d):
    """前日終値 > 前日確定の日足SMA200。h1インデックスへreindex+ffill。先読み無し(shift済み)。"""
    d1 = load_mt5_csv(USDJPY_D1)
    sma200 = d1["close"].rolling(200).mean().shift(1)
    up = pd.Series(np.where(d1["close"].shift(1) > sma200, 1, -1), index=d1.index)
    up_h1 = up.reindex(d.index.floor("D")).ffill()
    up_h1.index = d.index
    return up_h1.to_numpy() > 0


def weekly_up_regime(d):
    """前週終値 > 前週確定の30週SMA。さらに1週ラグしてh1へffill。先読み無し。"""
    w1 = load_mt5_csv(USDJPY_W1)
    sma30 = w1["close"].rolling(30).mean().shift(1)
    up = pd.Series(np.where(w1["close"].shift(1) > sma30, 1, -1), index=w1.index)
    up_shift = up.shift(1)
    up_h1 = up_shift.reindex(d.index, method="ffill")
    return up_h1.to_numpy() > 0


# ------------------------------------------------------------------ entries (A系/B系 x 固定RR/トレール)

def build_entries(d, atr_prev, s_idx, system, rr, stopk=2.0, trail=False):
    """entries = (i, e, stop, tgt, i_origin)。i=引き金足s、e=open[s+1]。
    A系: stop=拡大足(足s)の安値。 B系: stop=e-stopk*ATR[s-1]。
    trail=True の時は tgt を e+1000*risk にして事実上無効化（walk()側の trail_atr が出口を決める）。
    risk<=0 は除外。"""
    o, l = d["open"].to_numpy(), d["low"].to_numpy()
    ent = []
    for s in s_idx:
        e = o[s + 1]
        stop = l[s] if system == "A" else e - stopk * atr_prev[s]
        risk = e - stop
        if risk <= 0:
            continue
        tgt_rr = 1000.0 if trail else rr
        ent.append((s, e, stop, e + tgt_rr * risk, s))
    return ent


# ------------------------------------------------------------------ engine call + % pnl

def run_cell(d, entries, fill_win, fwd, trail_atr=0.0, trail_n=14, C=None, cost=COST,
             max_pos=1, pf=0.0):
    """walk() を cost=0 で呼び、外側で価格%コストを引く。C を渡すとショート(反転フレーム)扱いで
    実価格に戻してからコストを引く。cost は絶対値(円、往復0.9pip=0.009)で、レートに掛けない
    （FXのpipコストは定額。%換算する時だけ e_real で割る）。"""
    if not entries:
        return None
    args = SimpleNamespace(pullback_frac=pf, fill_win=fill_win, fwd=fwd, cost=0.0,
                            max_pos=max_pos, swap_pct=0.0, tp1_frac=0.0, exec_split=0,
                            trail_atr=trail_atr, trail_n=trail_n)
    t, _ = walk(d, entries, None, args)
    if t is None or len(t) == 0:
        return None
    e_real = (C - t["e_px"]) if C is not None else t["e_px"]
    pnl_px = t["R"] * t["risk"] - cost
    return t.assign(e_real=e_real, pnl_px=pnl_px, pnl_pct=pnl_px / e_real, y=t["time"].dt.year)


# ------------------------------------------------------------------ metrics

def pf_of(p):
    w, l = p[p > 0].sum(), -p[p < 0].sum()
    return float(w / l) if l > 0 else float("nan")


def stats(t, span_years):
    p = t["pnl_pct"].to_numpy()
    n = len(p)
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max()) * 100 if n else np.nan
    yr = t.groupby("y")["pnl_pct"].sum()
    pos_years = int((yr > 0).sum())
    return dict(N=n, N_yr=n / span_years, win=float((p > 0).mean() * 100), PF=pf_of(p),
                mean_pct=float(p.mean() * 100), tot_pct=float(p.sum() * 100), maxDD_pct=dd,
                pos_years=pos_years, n_years=len(yr))


def fmt_row(label, s, null_pf=None, null_mean=None):
    pf_s = f"{s['PF']:.2f}" if np.isfinite(s['PF']) else "inf"
    npf = f"{null_pf:.0f}" if null_pf is not None and np.isfinite(null_pf) else "-"
    nmn = f"{null_mean:.0f}" if null_mean is not None and np.isfinite(null_mean) else "-"
    return (f"{label:<48} N={s['N']:>5} N/年={s['N_yr']:>6.1f} 勝率={s['win']:>5.1f}% "
            f"PF={pf_s:>6} 平均%={s['mean_pct']:>+7.4f} 総%={s['tot_pct']:>+8.1f} "
            f"maxDD%={s['maxDD_pct']:>6.1f} 黒字年={s['pos_years']:>2}/{s['n_years']:<2} "
            f"| 帰無%ile(PF,平均)=({npf},{nmn})")


# ------------------------------------------------------------------ random-thinning null

def drop_null(pool_pct, n_fill, obs_mean, obs_pf, reps=400, seed=20260722):
    """成行・フィルタ無し(ゲートは維持)母集団から同じ本数だけランダムに残す帰無。価格%単位。"""
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

def block_bootstrap(t, k_months_list, metric="mean", n_boot=1000, seed=20260722):
    """巡回ブロック・ブートストラップ。metric='mean'(平均%) か 'pf'。月ブロックを丸ごと動かす。
    返り値: {k: (median, lo2.5, hi97.5, n_eff)}"""
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


def span_years(d):
    return (d.index[-1] - d.index[0]).days / 365.25


# ------------------------------------------------------------------ session / pdh filters

def session_hour(d, s_idx):
    """引き金足sのブローカー時刻(0-23)。"""
    return d.index[s_idx].hour.to_numpy()


SESSIONS = {
    "アジア(0-7時)": (0, 8),
    "ロンドン(8-15時)": (8, 16),
    "NY(16-23時)": (16, 24),
}
OPEN_WINDOWS = {
    "ロンドンOPEN前後(7-9時)": (7, 10),
    "NY OPEN前後(15-17時)": (15, 18),
}


def build_pdh_dist_series(d, atr_prev):
    """pdh_dist[s] = (close[s]-前日高値)/ATR[s-1]。前日高値=暦日resample("1D").max()を1日shift。
    ショート(反転フレーム)にも同じ式をそのまま適用する(mirror規約どおり正しい鏡像になる)。"""
    daily_high = d["high"].resample("1D").max()
    prev_daily_high = daily_high.shift(1).ffill()
    day_key = d.index.normalize()
    pdh = prev_daily_high.reindex(day_key)
    pdh.index = d.index
    close = d["close"]
    return ((close - pdh) / atr_prev).to_numpy()


def check_no_lookahead_pdh(d, atr_prev, s_idx, cut_frac=0.5):
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
