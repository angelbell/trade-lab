"""FVGの物語2の測定:「価格がFVGに戻ってきたとき、そこで反発するのか」。

事象: 生成後に価格が帯の近位端へ初めて戻った瞬間（=指値約定の代理）から前進レース。
勝ち = e+K*ATR(K=0.5/1/2, ATRは戻りバー前の確定値) に先に到達 / 負け = 帯遠位端-0.1ATR を先に割る。
同足は負け優先・上限500本。bearish は鏡像。

対照:
  C1 = 距離マッチングのランダム帯（fvg_fill_survey と同じ）→「場所」の意味を消す
  C2 = ギャップ無しの同型displacement（真ん中の足のレンジ>=1ATR・方向一致・帯は開かず）、
       近位端 = candle1 の高値(bull)/安値(bear)、幅は実FVGの中央値幅を使用
       →「FVGという帯」対「ただの押し目」を分離

Run: .venv/bin/python scratchpad/fvg_bounce_survey.py [--smoke] 2>/dev/null | tee scratchpad/out_fvg_bounce_survey.txt
"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_mt5_csv
from breakout_wave import resample
from fvg_fill_survey import detect, first_touch, SYMS

TFS = ["15min", "60min", "4h"]
MIN_ATR = 0.15
KS = [0.5, 1.0, 2.0]
BUF = 0.1
FWD = 500
N_CTRL = 3
RNG = np.random.default_rng(20260717)
NBOOT = 400


def detect_c2(h, l, c, a, min_range, is_bull):
    """C2: 真ん中の足のレンジ>=1ATR・方向一致・ギャップ無しの3本組。
    近位端 = candle1 の高値(bull)/安値(bear)。"""
    n = len(h)
    av = a[2:]
    ok = np.isfinite(av) & (av > 0)
    mid_rng = (h[1:-1] - l[1:-1]) / av
    kb = np.arange(2, n)
    if is_bull:
        mid_up = c[1:-1] > (h[1:-1] + l[1:-1]) / 2
        no_gap = l[2:] <= h[:-2]
        m = ok & (mid_rng >= min_range) & mid_up & no_gap & (c[2:] > c[:-2])
        return [(k, h[k - 2]) for k in kb[m]]
    else:
        mid_dn = c[1:-1] < (h[1:-1] + l[1:-1]) / 2
        no_gap = h[2:] >= l[:-2]
        m = ok & (mid_rng >= min_range) & mid_dn & no_gap & (c[2:] < c[:-2])
        return [(k, l[k - 2]) for k in kb[m]]


def race(h, l, a, starts, edges, widths, is_bull, ts=None, kpos=None, max_lag_days=90):
    """各ユニット: 戻り(近位端タッチ、生成から max_lag_days 以内)を待ち、戻ったバーからレース。
    返り値 = (retested, rb, out[K])。"""
    n = len(h)
    m = len(starts)
    # 1) 戻り: bull は low<=edge
    touch = first_touch(l if is_bull else h, starts, [[e, e, e] for e in edges], is_bull)[:, 0]
    retested = touch >= 0
    if ts is not None and kpos is not None:
        lag_ok = np.zeros(m, dtype=bool)
        hit = touch >= 0
        lag_ok[hit] = ((ts[touch[hit]] - ts[kpos[hit]]) / np.timedelta64(1, "D")) <= max_lag_days
        retested = retested & lag_ok
    rb = np.where(retested, touch, 0).astype(np.int64)
    atr_rb = a[np.maximum(rb - 1, 0)]
    ok = retested & np.isfinite(atr_rb) & (atr_rb > 0) & (rb < n - 1)
    # 2) レース: 勝ち水準3つ(K)と負け水準1つ。勝ちは反対側トリガ。
    idx = np.where(ok)[0]
    win_lv = [[(edges[u] + (1 if is_bull else -1) * K * atr_rb[u]) for K in KS] for u in idx]
    fail_lv = [[(edges[u] - (1 if is_bull else -1) * (widths[u] + BUF * atr_rb[u]))] * 3 for u in idx]
    st = rb[idx]
    wres = first_touch(h if is_bull else l, st, win_lv, not is_bull)
    fres = first_touch(l if is_bull else h, st, fail_lv, is_bull)[:, 0]
    out = np.full((m, len(KS)), -9, dtype=np.int64)   # -9=未戻り, 0=負け, 1=勝ち, 2=未決着
    for j, u in enumerate(idx):
        fj = fres[j] if fres[j] >= 0 else n + FWD
        cap = st[j] + FWD
        for e in range(len(KS)):
            wj = wres[j][e] if wres[j][e] >= 0 else n + FWD
            if wj > cap and fj > cap:
                out[u, e] = 2
            elif fj <= wj:                     # 同足は負け優先
                out[u, e] = 0
            else:
                out[u, e] = 1
    return retested, rb, out


def brate(out, e):
    """決着した勝負の勝率と、決着率。"""
    dec = (out[:, e] == 0) | (out[:, e] == 1)
    if dec.sum() == 0:
        return np.nan, 0
    return 100.0 * (out[dec, e] == 1).mean(), int(dec.sum())


def boot_diff(out_r, out_c, mon_r, mon_c, e):
    umon = mon_r.unique()
    gr = {mm: np.where(mon_r == mm)[0] for mm in umon}
    gc = {mm: np.where(mon_c == mm)[0] for mm in umon}
    diffs = []
    for _ in range(NBOOT):
        pick = RNG.choice(len(umon), size=len(umon), replace=True)
        ir = np.concatenate([gr[umon[i]] for i in pick])
        ic = np.concatenate([gc.get(umon[i], np.empty(0, dtype=int)) for i in pick])
        a1, n1 = brate(out_r[ir], e)
        a2, n2 = brate(out_c[ic], e)
        if n1 > 10 and n2 > 10:
            diffs.append(a1 - a2)
    return (np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan))


def cell(sym, tf, d):
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    a = ta.atr(d["high"], d["low"], d["close"], length=14).values
    ts = d.index.values.astype("datetime64[ns]")
    n = len(c)
    valid = np.where(np.isfinite(a) & (a > 0))[0]
    valid = valid[(valid >= 2) & (valid < n - 1)]
    bull, bear = detect(h, l, a, MIN_ATR)
    for is_bull, fvgs, tag in ((True, bull, "bull"), (False, bear, "bear")):
        if len(fvgs) < 30:
            print(f"  {sym:<7}{tf:<6}{tag:<5} n少なすぎskip"); continue
        w_med = float(np.median([abs(t - b) for (_, t, b, _) in fvgs]))
        # 実FVG
        st_r = np.array([k + 1 for (k, t, b, s) in fvgs])
        ed_r = np.array([t for (k, t, b, s) in fvgs])
        wd_r = np.array([abs(t - b) for (k, t, b, s) in fvgs])
        kp_r = np.array([k for (k, t, b, s) in fvgs])
        # C1: ランダム帯（同ATR距離・同幅）
        st_c1, ed_c1, wd_c1, kp_c1 = [], [], [], []
        for (k, t, b, s) in fvgs:
            d_top = (c[k] - t) / a[k] if is_bull else (t - c[k]) / a[k]
            width = abs(t - b) / a[k]
            for r in RNG.choice(valid, size=N_CTRL, replace=True):
                e2 = c[r] - d_top * a[r] if is_bull else c[r] + d_top * a[r]
                st_c1.append(r + 1); ed_c1.append(e2); wd_c1.append(width * a[r]); kp_c1.append(r)
        st_c1, ed_c1, wd_c1, kp_c1 = map(np.array, (st_c1, ed_c1, wd_c1, kp_c1))
        # C2: ギャップ無しdisplacement（幅=実の中央値）。実FVGと同じく「離れてから戻る」力学に
        # 揃えるため、生成終値→ゾーンの距離が実の中央値以上のものだけ残す（無いと即時タッチ混入）。
        d_med = float(np.median([((c[k] - t) if is_bull else (t - c[k])) / a[k]
                                 for (k, t, b, s) in fvgs]))
        c2 = detect_c2(h, l, c, a, 1.0, is_bull)
        c2 = [(k, e) for (k, e) in c2
              if (((c[k] - e) if is_bull else (e - c[k])) / a[k]) >= d_med]
        st_c2 = np.array([k + 1 for (k, e) in c2])
        ed_c2 = np.array([e for (k, e) in c2])
        wd_c2 = np.full(len(c2), w_med)
        kp_c2 = np.array([k for (k, e) in c2])

        rt_r, rb_r, out_r = race(h, l, a, st_r, ed_r, wd_r, is_bull, ts, kp_r)
        rt_c1, _, out_c1 = race(h, l, a, st_c1, ed_c1, wd_c1, is_bull, ts, kp_c1)
        rt_c2, _, out_c2 = race(h, l, a, st_c2, ed_c2, wd_c2, is_bull, ts, kp_c2)

        mon_r = pd.PeriodIndex(pd.DatetimeIndex(ts[kp_r]), freq="M")
        mon_c1 = pd.PeriodIndex(pd.DatetimeIndex(ts[kp_c1]), freq="M")
        mon_c2 = pd.PeriodIndex(pd.DatetimeIndex(ts[kp_c2]), freq="M") if len(c2) else None

        lag = (ts[rb_r[rt_r]] - ts[kp_r[rt_r]]) / np.timedelta64(1, "D")
        line = (f"  {sym:<7}{tf:<6}{tag:<5} n={len(fvgs):>6} 戻り率={100*rt_r.mean():>5.1f}% "
                f"lag中央={np.median(lag):.2f}日 |")
        for e, K in enumerate(KS):
            wr, nd = brate(out_r, e)
            w1, _ = brate(out_c1, e)
            w2, n2d = brate(out_c2, e)
            if K == 1.0:
                lo1, hi1 = boot_diff(out_r, out_c1, mon_r, mon_c1, e)
                lo2, hi2 = boot_diff(out_r, out_c2, mon_r, mon_c2, e) if mon_c2 is not None else (np.nan, np.nan)
                line += (f" K1: 実{wr:.1f}%(決着{nd}) C1 {w1:.1f}% Δ{wr-w1:+.1f}[{lo1:+.1f},{hi1:+.1f}]"
                         f" C2 {w2:.1f}%(n{n2d}) Δ{wr-w2:+.1f}[{lo2:+.1f},{hi2:+.1f}] |")
            else:
                line += f" K{K:g}: 実{wr:.1f}/C1 {w1:.1f}/C2 {w2:.1f} |"
        print(line)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    syms = {"eurusd": SYMS["eurusd"]} if args.smoke else SYMS
    tfs = ["15min"] if args.smoke else TFS
    print("反発レース（帯の近位端に戻った瞬間から: 勝=+K*ATR先着 / 負=帯遠位端-0.1ATR先着 / 同足負け優先 / 500本上限）")
    print("C1=ランダム同距離帯（場所の意味を消す） C2=ギャップ無し同型displacement（帯 vs ただの押し目）")
    for sym, path in syms.items():
        df = load_mt5_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path))
        if sym == "usdjpy":
            df = df.loc["2000-01-01":]
        if args.smoke:
            df = df.loc["2024-07-01":]
        for tf in tfs:
            cell(sym, tf, resample(df, tf))


if __name__ == "__main__":
    main()
