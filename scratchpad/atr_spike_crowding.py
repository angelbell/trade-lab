"""クラウディング診断: 拡大足の「フォロースルー窓」は年とともに縮んでいるか。

問い: 直近2年の劣化はノイズか、それとも機構そのものの摩耗（先回りされて続きが出なくなった）か。
トレード数（年10〜30本）では区別できないので、【足の水準】で測る。標本は年 数百〜数千本になる。

測るもの（拡大足 i ごと）:
  rev  = 引き金足の【反対端（安値）】を割るまでの本数（上限200本で打ち切り）
  mfe  = 反対端を割る前に close[i] から伸びた最大幅（ATR単位）
  ft   = mfe >= 1.0ATR に到達したか（＝続きが出たか）

帰無（同じ年の中で作る。年ごとのボラ・レジームを吸収するため）:
  (a) 素の帰無  : その年の全バーから無作為
  (b) 距離一致帰無: その年の【非】拡大足のうち (close-low)/ATR が拡大足と ±10% 以内のもの。
      「反対端までの距離が同じだが拡大足ではないバー」＝機構だけを抜いた対照。こちらが主。

判定: 比（拡大足 / 帰無）の年別系列に対する Spearman(年, 比)。
      単調に縮んでいれば摩耗、ゼロ付近なら直近2年はノイズ。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv          # noqa: E402

CAP = 200          # 打ち切り本数
FT_ATR = 1.0       # 「続きが出た」と見なす幅
NDRAW = 5          # 帰無1件あたりの抽出数
RNG = np.random.default_rng(20260722)


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def excursion(h, l, c, ap, idx):
    """idx の各バーについて (反対端を割るまでの本数, その間の最大幅ATR単位, 続き有無)。"""
    rev = np.empty(len(idx), dtype=float)
    mfe = np.empty(len(idx), dtype=float)
    n = len(c)
    for q, i in enumerate(idx):
        end = min(i + CAP, n - 1)
        lo0, c0, a0 = l[i], c[i], ap[i]
        j = i + 1
        top = -np.inf
        while j <= end:
            if h[j] > top:
                top = h[j]
            if l[j] <= lo0:
                break
            j += 1
        rev[q] = j - i
        mfe[q] = (top - c0) / a0 if np.isfinite(top) else np.nan
    return rev, mfe, mfe >= FT_ATR


def matched_null(years, ap, c, l, spike_mask, yr_of, d_all, sp_idx):
    """同じ年・(close-low)/ATR が ±10% の【非】拡大足を NDRAW 件ずつ。"""
    out = []
    for y in np.unique(yr_of[sp_idx]):
        pool = np.flatnonzero((yr_of == y) & (~spike_mask) & np.isfinite(d_all))
        if len(pool) < 20:
            continue
        dp = d_all[pool]
        order = np.argsort(dp)
        pool_s, dp_s = pool[order], dp[order]
        for i in sp_idx[yr_of[sp_idx] == y]:
            t = d_all[i]
            for tol in (0.10, 0.25, 0.50):
                lo_i = np.searchsorted(dp_s, t * (1 - tol), "left")
                hi_i = np.searchsorted(dp_s, t * (1 + tol), "right")
                if hi_i - lo_i >= NDRAW:
                    out.append(RNG.choice(pool_s[lo_i:hi_i], NDRAW, replace=False))
                    break
    return np.concatenate(out) if out else np.array([], dtype=int)


def analyse(name, path, k, lo_year, tz_utc=False, start=None):
    d = load_mt5_csv(path)
    if tz_utc:
        idx = d.index.tz_convert("Europe/Riga").tz_localize(None).tz_localize("UTC")
        d = d.set_index(idx)
    if start:
        d = d.loc[start:]
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    yr = d.index.year.to_numpy()

    spike = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    d_all = np.where(np.isfinite(ap) & (ap > 0), (c - l) / ap, np.nan)
    ok = np.isfinite(ap) & (ap > 0) & (yr >= lo_year) & (np.arange(len(c)) < len(c) - 1)
    sp = np.flatnonzero(spike & ok)
    if len(sp) < 50:
        print(f"  {name}: 拡大足が {len(sp)} 本しかない — 省略")
        return None

    nl_raw = RNG.choice(np.flatnonzero(ok & ~spike), min(len(sp) * NDRAW,
                                                          int(ok.sum() * 0.8)), replace=False)
    nl_mat = matched_null(None, ap, c, l, spike, yr, d_all, sp)

    rows = []
    for lab, idx in (("spike", sp), ("null_raw", nl_raw), ("null_mat", nl_mat)):
        if len(idx) == 0:
            continue
        rev, mfe, ft = excursion(h, l, c, ap, idx)
        rows.append(pd.DataFrame({"lab": lab, "y": yr[idx], "rev": rev,
                                  "mfe": mfe, "ft": ft.astype(float)}))
    R = pd.concat(rows, ignore_index=True)

    print(f"\n===== {name}  (k={k}, {lo_year}-)  拡大足 N={len(sp)}  "
          f"距離一致帰無 N={len(nl_mat)}")
    print(f"{'年':>6} {'N':>5} | {'反対端まで(本) 拡大/一致帰無':>28} | "
          f"{'続き率(>=1ATR) 拡大/一致帰無':>30} | {'最大幅ATR 拡大/一致帰無':>26}")
    ser = {}
    for y in sorted(R.loc[R.lab == "spike", "y"].unique()):
        a = R[(R.lab == "spike") & (R.y == y)]
        b = R[(R.lab == "null_mat") & (R.y == y)]
        if len(a) < 15 or len(b) < 15:
            continue
        rv_a, rv_b = a["rev"].median(), b["rev"].median()
        ft_a, ft_b = a["ft"].mean(), b["ft"].mean()
        mf_a, mf_b = a["mfe"].median(), b["mfe"].median()
        ser[y] = (rv_a / rv_b, ft_a - ft_b, mf_a - mf_b, len(a))
        print(f"{y:>6} {len(a):>5} | {rv_a:8.1f} /{rv_b:7.1f}  比{rv_a/rv_b:5.2f} | "
              f"{ft_a*100:8.1f}% /{ft_b*100:6.1f}%  差{(ft_a-ft_b)*100:+6.1f}pt | "
              f"{mf_a:7.2f} /{mf_b:6.2f}  差{mf_a-mf_b:+5.2f}")

    ys = np.array(sorted(ser))
    if len(ys) >= 5:
        print(f"  {'':>4}Spearman(年, 指標)  "
              f"反対端までの比 rho={stats.spearmanr(ys, [ser[y][0] for y in ys])[0]:+.2f} "
              f"(P={stats.spearmanr(ys, [ser[y][0] for y in ys])[1]:.3f}) · "
              f"続き率の差 rho={stats.spearmanr(ys, [ser[y][1] for y in ys])[0]:+.2f} "
              f"(P={stats.spearmanr(ys, [ser[y][1] for y in ys])[1]:.3f}) · "
              f"最大幅の差 rho={stats.spearmanr(ys, [ser[y][2] for y in ys])[0]:+.2f} "
              f"(P={stats.spearmanr(ys, [ser[y][2] for y in ys])[1]:.3f})")
    # 前半 vs 後半
    mid = ys[len(ys) // 2]
    e = R[(R.lab == "spike") & (R.y < mid)]
    lt = R[(R.lab == "spike") & (R.y >= mid)]
    en = R[(R.lab == "null_mat") & (R.y < mid)]
    ln_ = R[(R.lab == "null_mat") & (R.y >= mid)]
    print(f"  {'':>4}前半(-{mid-1}) 続き率 {e['ft'].mean()*100:.1f}% (帰無 {en['ft'].mean()*100:.1f}%) "
          f"→ 後半({mid}-) {lt['ft'].mean()*100:.1f}% (帰無 {ln_['ft'].mean()*100:.1f}%)   "
          f"リフト {(e['ft'].mean()-en['ft'].mean())*100:+.1f}pt → "
          f"{(lt['ft'].mean()-ln_['ft'].mean())*100:+.1f}pt")
    return R


if __name__ == "__main__":
    out = {}
    for nm, path, tz, st, ly in (
            ("Binance BTCUSDT 1H", "data/binance_btcusdt_h1.csv", True, "2018-01-01", 2018),
            ("Binance ETHUSDT 1H", "data/binance_ethusdt_h1.csv", True, "2018-01-01", 2018),
            ("Vantage USDJPY 1H", "data/vantage_usdjpy_h1.csv", False, "2000-01-01", 2000)):
        for k in (1.0, 2.0):
            r = analyse(nm, path, k, ly, tz_utc=tz, start=st)
            if r is not None:
                out[(nm, k)] = r

    b = out[("Binance BTCUSDT 1H", 2.0)]
    sp = b[b.lab == "spike"]
    nm_ = b[b.lab == "null_mat"]
    assert 400 <= len(sp) <= 1400, len(sp)
    assert sp["ft"].mean() > nm_["ft"].mean(), (sp["ft"].mean(), nm_["ft"].mean())
    print(f"\nOK: BTC k2.0 拡大足 N={len(sp)} 続き率 {sp['ft'].mean()*100:.1f}% "
          f"> 距離一致帰無 {nm_['ft'].mean()*100:.1f}%（機構が存在する側の確認）")
