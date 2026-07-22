"""dow_ema_4step.py -- faithful mechanization of the video method (ハル: ダウ理論+2EMA 4STEP):

  STEP1 (1h): EMA20 golden-crosses EMA80, THEN Dow confirms (last two confirmed ZigZag
         swing highs ascending AND last two swing lows ascending) = trend ON.
  STEP2 (1h): pullback -- price approaches the 80EMA (low <= EMA80 + 0.25*ATR1h) while
         trend ON -> opens a 5m watch window.
  STEP3 (5m): during the window, the decline's 戻り高値 = last 5m ZigZag swing high
         CONFIRMED after the window opened; enter long when a 5m bar CLOSES above it.
         Weekend rule: no entries Friday >= 20:00 broker time.
  STEP4: stop = min(lowest 5m low since window open, 1h EMA80) - 0.05*ATR1h
         (below the 5m low AND below the 80EMA, per the video). 3 equal lots, TPs at the
         nearest overhead confirmed 1h swing highs (fallback: equal spacing if fewer than
         3 exist); after TP1 fills, remaining stops -> break-even.

Cells: exits {3分割+BE (faithful) / 3分割 no-BE / RR2 all-out / RR3 all-out}
     x signal {GC後の初動のみ (first entry per GC, as-written) / 全押し目}.
Instruments: GOLD 5m (full 7.7yr = the faithful testbed), USDJPY 5m (97 days only!),
USDJPY 1h approximation over 16yr (STEP3 replaced by 1h close back above EMA20).
Costs: gold $0.3 RT, USDJPY 1.2 pips RT. Conservative same-bar rule: stop before TP.

Pre-registered read: if the all-pullback RR2 base is net meanR<=0, the skeleton is dead
and no exit engineering can save it. The BE-move column prices the "負けなし" comfort.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag, resample

TOUCH_K = 0.25         # STEP2: low within this*ATR1h above EMA80 opens the window
BUF_K = 0.05           # stop buffer in ATR1h
WIN_MAX = 576          # 5m bars: window expires after 2 days without a trigger
FWD5, FWD1H = 2016, 500


def prep_1h(d1h):
    c = d1h["close"]
    e20 = c.ewm(span=20, adjust=False).mean()
    e80 = c.ewm(span=80, adjust=False).mean()
    atr = ta.atr(d1h["high"], d1h["low"], d1h["close"], 14)
    a = np.where(np.isnan(atr.values), np.nanmean(atr.values), atr.values)
    sw = swings_zigzag(d1h["high"].values, d1h["low"].values, a, 2.0)
    n = len(c)
    dow = np.zeros(n, bool)
    lastH = prevH = lastL = prevL = None
    si = 0
    for i in range(n):
        while si < len(sw) and sw[si][0] <= i:
            cc, ii, pp, kk = sw[si]
            if kk == 1:
                prevH, lastH = lastH, pp
            else:
                prevL, lastL = lastL, pp
            si += 1
        dow[i] = (lastH is not None and prevH is not None and lastH > prevH and
                  lastL is not None and prevL is not None and lastL > prevL)
    gc = (e20 > e80) & (e20.shift(1) <= e80.shift(1))
    sh = [(cc, pp) for cc, ii, pp, kk in sw if kk == 1]      # confirmed swing highs
    return dict(e20=e20.values, e80=e80.values, atr=atr.values, dow=dow,
                up=(e20 > e80).values, gcid=gc.cumsum().values, sh=sh, idx=d1h.index)


def tps_overhead(sh, i1h_confirmed, e):
    """nearest 3 confirmed 1h swing highs above entry; equal-spacing fallback."""
    ov = sorted(p for cc, p in sh if cc <= i1h_confirmed and p > e)[:3]
    if not ov:
        return None
    while len(ov) < 3:
        ov.append(ov[-1] + (ov[-1] - (ov[-2] if len(ov) > 1 else e)))
    return ov


def walk(h, l, c, j0, e, stop, tps, be, fwd):
    """three equal lots to tps[0..2]; stop-first same-bar; BE after TP1 if be.
    tps may be a 1-list => all-out single target. Returns (R_gross, exit_j)."""
    risk = e - stop
    units = len(tps)
    alive = list(range(units))
    cur_stop, R, hit1 = stop, 0.0, False
    end = min(j0 + fwd, len(c) - 1)
    for j in range(j0 + 1, end + 1):
        if l[j] <= cur_stop:
            R += len(alive) * (cur_stop - e) / risk / units
            return R, j
        while alive and h[j] >= tps[alive[0]]:
            k = alive.pop(0)
            R += (tps[k] - e) / risk / units
            if not hit1:
                hit1 = True
                if be:
                    cur_stop = e
        if not alive:
            return R, j
    R += len(alive) * (c[end] - e) / risk / units
    return R, end


def signals_5m(d5, H):
    """candidate entries: (j, e, stop, tps, gcid). No busy filter here."""
    h5, l5, c5 = d5["high"].values, d5["low"].values, d5["close"].values
    atr5 = ta.atr(d5["high"], d5["low"], d5["close"], 14).values
    a5 = np.where(np.isnan(atr5), np.nanmean(atr5), atr5)
    sw5 = swings_zigzag(h5, l5, a5, 2.0)
    sh5 = [(cc, pp) for cc, ii, pp, kk in sw5 if kk == 1]
    i1 = np.searchsorted(H["idx"].values, d5.index.values, side="right") - 1
    conf = np.maximum(i1 - 1, 0)                      # prior COMPLETED 1h bar
    trend = (H["up"] & H["dow"])[conf] & (i1 >= 1)
    e80 = H["e80"][conf]
    a1 = H["atr"][conf]
    gcid = H["gcid"][conf]
    out = []
    wopen, w0, minlow = False, -1, np.inf
    sp = 0
    rh = None                                          # 戻り高値 confirmed in-window
    for j in range(len(c5)):
        while sp < len(sw5) and sw5[sp][0] <= j:
            if sw5[sp][3] == 1 and wopen and sw5[sp][0] >= w0:
                rh = sw5[sp][2]
            sp += 1
        if not trend[j] or np.isnan(a1[j]):
            wopen = False
            continue
        if not wopen:
            if l5[j] <= e80[j] + TOUCH_K * a1[j]:
                wopen, w0, minlow, rh = True, j, l5[j], None
            continue
        minlow = min(minlow, l5[j])
        if j - w0 > WIN_MAX:
            wopen = False
            continue
        if rh is not None and c5[j] > rh:
            t = d5.index[j]
            if t.weekday() == 4 and t.hour >= 20:      # weekend-carry skip
                wopen = False
                continue
            e = c5[j]
            stop = min(minlow, e80[j]) - BUF_K * a1[j]
            if e - stop > 0:
                tps = tps_overhead(H["sh"], conf[j], e)
                if tps is not None:
                    out.append((j, e, stop, tps, gcid[j]))
            wopen = False
    return out


def signals_1h(d1h, H):
    """USDJPY approximation: STEP3 = 1h close back above EMA20 after the 80EMA touch."""
    h1, l1, c1 = d1h["high"].values, d1h["low"].values, d1h["close"].values
    out = []
    wopen, w0, minlow = False, -1, np.inf
    for i in range(1, len(c1)):
        tr = H["up"][i - 1] and H["dow"][i - 1]        # prior completed bar
        a1 = H["atr"][i - 1]
        if not tr or np.isnan(a1):
            wopen = False
            continue
        e80 = H["e80"][i - 1]
        if not wopen:
            if l1[i] <= e80 + TOUCH_K * a1:
                wopen, w0, minlow = True, i, l1[i]
            continue
        minlow = min(minlow, l1[i])
        if i - w0 > 72:
            wopen = False
            continue
        if c1[i] > H["e20"][i - 1] and c1[i] > c1[i - 1]:
            t = d1h.index[i]
            if t.weekday() == 4 and t.hour >= 20:
                wopen = False
                continue
            e = c1[i]
            stop = min(minlow, e80) - BUF_K * a1
            if e - stop > 0:
                tps = tps_overhead(H["sh"], i - 1, e)
                if tps is not None:
                    out.append((i, e, stop, tps, H["gcid"][i - 1]))
            wopen = False
    return out


def evaluate(name, d, sigs, rt, fwd, span):
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    print(f"\n===== {name} ({span:.1f}yr)  候補シグナル n={len(sigs)} =====")
    modes = [("3分割+建値移動 (動画忠実)", "tp3", True), ("3分割 建値移動なし", "tp3", False),
             ("RR2 全量", 2.0, False), ("RR3 全量", 3.0, False)]
    for fresh in (False, True):
        tag_f = "初動のみ(GC後1回)" if fresh else "全押し目"
        for mtag, mode, be in modes:
            last_x, used, rows = -1, set(), []
            for (j, e, stop, tps, gid) in sigs:
                if j <= last_x:
                    continue
                if fresh:
                    if gid in used:
                        continue
                    used.add(gid)
                risk = e - stop
                t_ = tps if mode == "tp3" else [e + mode * risk]
                R, xj = walk(h, l, c, j, e, stop, t_, be, fwd)
                rows.append((d.index[j], R - rt / risk))
                last_x = xj
            if len(rows) < 8:
                print(f"  [{tag_f}] {mtag:<22} n={len(rows)} few")
                continue
            ts = pd.DatetimeIndex([r[0] for r in rows])
            Rn = np.array([r[1] for r in rows])
            yr = ts.year.values
            half = np.median(yr)
            pf = Rn[Rn > 0].sum() / max(1e-9, abs(Rn[Rn <= 0].sum()))
            eq = np.cumsum(Rn)
            dd = (np.maximum.accumulate(eq) - eq).max()
            g = sum(Rn[yr == y].sum() > 0 for y in np.unique(yr))
            print(f"  [{tag_f}] {mtag:<22} N/yr={len(Rn)/span:5.1f} win={(Rn>0).mean()*100:4.1f}% "
                  f"PF={pf:4.2f} meanR={Rn.mean():+.3f} IS/OOS={Rn[yr<half].mean():+.2f}/"
                  f"{Rn[yr>=half].mean():+.2f} totR/yr={Rn.sum()/span:+5.1f} DD={dd:5.1f}R "
                  f"grn={g}/{len(np.unique(yr))}")


def main():
    g5 = load_mt5_csv("data/vantage_xauusd_m5.csv").loc["2018-09-14":]
    H = prep_1h(resample(g5, "60min"))
    span = (g5.index[-1] - g5.index[0]).days / 365.25
    evaluate("GOLD 1h+5m (忠実)", g5, signals_5m(g5, H), 0.3, FWD5, span)

    j1 = load_mt5_csv("data/vantage_usdjpy_h1.csv")
    Hj = prep_1h(j1)
    span = (j1.index[-1] - j1.index[0]).days / 365.25
    evaluate("USDJPY 1h近似 (16yr)", j1, signals_1h(j1, Hj), 0.012, FWD1H, span)

    jm = load_mt5_csv("data/vantage_usdjpy_m1.csv")
    j5 = resample(jm, "5min")
    Hj5 = prep_1h(resample(j5, "60min"))
    span = max((j5.index[-1] - j5.index[0]).days / 365.25, 1e-6)
    evaluate("USDJPY 1h+5m (忠実・97日のみ!)", j5, signals_5m(j5, Hj5), 0.012, FWD5, span)


if __name__ == "__main__":
    main()
