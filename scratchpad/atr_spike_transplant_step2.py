"""【仕様カード第6段】ATR拡大足機構の全銘柄横展開 ― STEP2: トレード水準。

STEP1 を通った 銘柄×方向 だけに、BTC h1 で凍結した仕様をそのまま当てる（一切調整しない）:
  引き金: 実体 > ATR14[s-1]*k, k in {1.5,2.0,2.5}
  入口: ロング=次足始値・成行 / ショート=戻り売り指値pf=0.5(fill_win=200)
  損切り: 拡大足の反対端（A系）
  出口: ATR×3トレール(trail_atr=3.0, trail_n=14)、保有上限fwd=20
  ゲート: 無し
  フィルタ: ロングのみ pdh_dist>0（前日高値超え）。ショートには付けない。

損益は「入口価格に対する%」（ロット0.01固定）。walk() は cost=0 で回し、コストは外側で
実価格に対する割合として引く（mirror-cost-overcharge 回避、x_conventions.md 参照）。
執行は src/engine/walk.py の walk() と src/engine/mirror.py の invert() のみを使う。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pandas_ta as ta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.data_loader import load_mt5_csv   # noqa: E402
from src.engine.walk import walk           # noqa: E402
from src.engine.mirror import invert       # noqa: E402

FWD = 20
FILL_WIN = 200
TRAIL_ATR = 3.0
TRAIL_N = 14


# ------------------------------------------------------------------ features / entries

def atr_prev_of(d, n=14):
    return ta.atr(d["high"], d["low"], d["close"], length=n).shift(1).to_numpy()


def raw_triggers(d, atr_prev, k):
    o, c = d["open"].to_numpy(), d["close"].to_numpy()
    hit = (c - o > atr_prev * k) & (c > o) & np.isfinite(atr_prev)
    s = np.flatnonzero(hit)
    return s[s + 1 < len(d)]


def build_pdh_dist(d, atr_prev):
    daily_high = d["high"].resample("1D").max()
    prev_daily_high = daily_high.shift(1).ffill()
    day_key = d.index.normalize()
    pdh = prev_daily_high.reindex(day_key)
    pdh.index = d.index
    return ((d["close"] - pdh) / atr_prev).to_numpy()


def build_entries(d, atr_prev, s_idx, rr=1000.0):
    """A系: stop=拡大足の安値。rr=1000 は実質「トレールのみ」（目標が生きたまま非現実的遠方に
    置かれ、事実上トレールだけが決める。atr_spike_btc_h1-horizon 節の EXITS 定義と同じ規約）。"""
    o, l = d["open"].to_numpy(), d["low"].to_numpy()
    ent = []
    for s in s_idx:
        e = o[s + 1]
        stop = l[s]
        risk = e - stop
        if risk <= 0:
            continue
        ent.append((s, e, stop, e + rr * risk, s))
    return ent


def run_cell(d, entries, pf, cost_frac, C=None):
    if not entries:
        return None
    args = SimpleNamespace(pullback_frac=pf, fill_win=FILL_WIN, fwd=FWD, cost=0.0,
                            max_pos=1, swap_pct=0.0, tp1_frac=0.0, exec_split=0,
                            trail_atr=TRAIL_ATR, trail_n=TRAIL_N)
    t, _ = walk(d, entries, None, args)
    if t is None or len(t) == 0:
        return None
    e_real = (C - t["e_px"]) if C is not None else t["e_px"]
    pnl_px = t["R"] * t["risk"] - cost_frac * e_real
    return t.assign(pnl_px=pnl_px, pnl_pct=pnl_px / e_real, y=t["time"].dt.year)


# ------------------------------------------------------------------ metrics

def pf_of(p):
    w, l = p[p > 0].sum(), -p[p < 0].sum()
    return float(w / l) if l > 0 else float("inf")


def max_streak(p):
    run_l = best = 0
    for x in p:
        run_l = run_l + 1 if x <= 0 else 0
        best = max(best, run_l)
    return best


def stats(t, span_years):
    p = t["pnl_pct"].to_numpy()
    n = len(p)
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max()) * 100 if n else np.nan
    yr = t.groupby("y")["pnl_pct"].sum()
    return dict(N=n, N_yr=n / span_years, win=float((p > 0).mean() * 100), PF=pf_of(p),
                mean_pct=float(p.mean() * 100), tot_pct=float(p.sum() * 100), maxDD_pct=dd,
                ratio_dd=float(p.sum() * 100 / dd) if dd else float("nan"),
                streak=max_streak(p), pos_years=int((yr > 0).sum()), n_years=int(len(yr)))


def per_year_rows(t):
    rows = []
    for y, g in t.groupby("y"):
        p = g["pnl_pct"].to_numpy()
        rows.append(dict(year=int(y), N=len(p), win=float((p > 0).mean() * 100), PF=pf_of(p),
                          mean_pct=float(p.mean() * 100), tot_pct=float(p.sum() * 100)))
    return rows


def fmt(s):
    pf_s = f"{s['PF']:.2f}" if np.isfinite(s['PF']) else "inf"
    return (f"N={s['N']:>5} N/年={s['N_yr']:>6.1f} 勝率={s['win']:>5.1f}% PF={pf_s:>6} "
            f"平均%={s['mean_pct']:>+7.3f} 総%={s['tot_pct']:>+8.1f} maxDD%={s['maxDD_pct']:>6.1f} "
            f"総/DD={s['ratio_dd']:>6.2f} 連敗={s['streak']:>3} 黒字年={s['pos_years']}/{s['n_years']}")


# ------------------------------------------------------------------ drop-null (random thinning)

def drop_null(pool_pct, n_fill, obs_mean, obs_pf, reps=400, seed=20260721):
    rng = np.random.default_rng(seed)
    Np = len(pool_pct)
    if n_fill <= 0 or n_fill > Np:
        return dict(pf_pctile=float("nan"), mean_pctile=float("nan"))
    means, pfs = np.empty(reps), np.empty(reps)
    for r in range(reps):
        s = rng.choice(pool_pct, size=n_fill, replace=False)
        means[r] = s.mean()
        pfs[r] = pf_of(s)
    return dict(pf_pctile=float((pfs < obs_pf).mean() * 100),
                mean_pctile=float((means < obs_mean).mean() * 100))
