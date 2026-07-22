"""dow4step_anatomy.py -- WHERE does the 1h 20/80EMA 4STEP method die? Stage-wise autopsy.

The full method scored win%~RR-breakeven on gold AND USDJPY. That aggregate hides which
stage is diseased. Decompose on gold (5m LTF, the faithful corpse) and BTC 15m (the one
instrument where pullback edge exists -> separates instrument-disease from method-disease):

STAGE1  regime: does [GC(20>80) AND Dow HH/HL] on 1h predict anything? +-1ATR(1h) barrier
        race win% and fwd-24h net drift, regime-ON vs ALL (beta). If ON==ALL the regime
        label is decorative.
STAGE2  location: first touch of the 1h 80EMA(+0.25ATR) *within* regime-ON. Race win% vs
        hour-matched beta of regime-ON bars. If touch==regime beta, the 80EMA is not a
        location (consistent w/ the line-family nulls).
STAGE3  trigger (LTF window walk, RR2 fixed, cost-netted, no-overlap):
        [T] LTF close breaks the decline's 戻り高値 (as implemented, dow ON)  = the method
        [A] enter AT the touch bar close, no confirmation (dow ON)           = tax ablation
        [D] variant T with the Dow HH/HL requirement dropped (GC only)       = dow ablation
        + 確信の税 = median (entryT - entryA)/ATR1h on windows where T fired, fire-rate.
Stops: min(pullback minlow, 80EMA)-0.05ATR1h (method faithful). Target RR2. fwd 2016 LTF bars.
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
from dow_ema_4step import prep_1h

K_RACE = 96
TOUCH_K = 0.25
WIN_MAX_D = 2.0          # window expiry in days
FWD = 2016               # LTF bars (5m: 7d / 15m: 21d)


def race(h, l, c, atr, K=K_RACE):
    n = len(c)
    up_l, dn_l = c + atr, c - atr
    t_up = np.full(n, K + 1, np.int32)
    t_dn = np.full(n, K + 1, np.int32)
    for k in range(1, K + 1):
        hs, ls = np.empty(n), np.empty(n)
        hs[:n - k], ls[:n - k] = h[k:], l[k:]
        hs[n - k:], ls[n - k:] = -np.inf, np.inf
        t_up = np.where((t_up > K) & (hs >= up_l), k, t_up)
        t_dn = np.where((t_dn > K) & (ls <= dn_l), k, t_dn)
    win = (np.minimum(t_up, t_dn) <= K) & (t_up < t_dn)
    valid = ~np.isnan(atr) & (atr > 0) & (np.arange(n) < n - K)
    return win, valid


def stage12(name, d1h, H):
    h, l, c = d1h["high"].values, d1h["low"].values, d1h["close"].values
    atr = ta.atr(d1h["high"], d1h["low"], d1h["close"], 14).shift(1).values
    win, valid = race(h, l, c, atr)
    on = H["up"] & H["dow"]
    # shift 1: regime known at prior completed bar
    onp = np.zeros(len(c), bool); onp[1:] = on[:-1]
    # fwd 24h drift (ATR-normalized, no barriers)
    fwd = np.full(len(c), np.nan)
    fwd[:-24] = (c[24:] - c[:-24]) / np.where(atr[:-24] > 0, atr[:-24], np.nan)
    m_all, m_on = valid, valid & onp
    print(f"\n[STAGE1 {name}] regime = GC(20>80) ∧ Dow(HH∧HL) on 1h   ON率={onp[valid].mean()*100:.0f}%")
    print(f"  race win: ON {win[m_on].mean()*100:4.1f}%  vs ALL {win[m_all].mean()*100:4.1f}%  "
          f"diff {(win[m_on].mean()-win[m_all].mean())*100:+.1f}pt   (n_on={m_on.sum()})")
    print(f"  fwd24h/ATR: ON {np.nanmean(fwd[m_on]):+.3f}  vs ALL {np.nanmean(fwd[m_all]):+.3f}"
          f"   GCのみ(dow無視): win {win[valid & np.concatenate(([False], H['up'][:-1]))].mean()*100:4.1f}%")
    # STAGE2: first 80EMA touch inside regime
    e80p = np.concatenate(([np.nan], H["e80"][:-1]))
    a1p = np.concatenate(([np.nan], H["atr"][:-1]))
    touch = onp & (l <= e80p + TOUCH_K * a1p)
    first = touch & ~np.concatenate(([False], touch[:-1]))
    m_t = valid & first
    hours = d1h.index.hour.values
    bmap = {hh: win[m_on & (hours == hh)].mean() for hh in range(24)}
    beta = np.mean([bmap[hh] for hh in hours[m_t] if not np.isnan(bmap.get(hh, np.nan))])
    print(f"[STAGE2 {name}] 80EMA初タッチ(レジーム内) n={m_t.sum()}  race win {win[m_t].mean()*100:4.1f}%"
          f"  vs レジーム内beta {beta*100:4.1f}%  diff {(win[m_t].mean()-beta)*100:+.1f}pt")


def windows(dL, H, use_dow):
    """LTF walk -> per-window records: (j0, jT|None, minlow@T, e80, atr1h at open/trigger)."""
    hL, lL, cL = dL["high"].values, dL["low"].values, dL["close"].values
    atrL = ta.atr(dL["high"], dL["low"], dL["close"], 14).values
    aL = np.where(np.isnan(atrL), np.nanmean(atrL), atrL)
    sw = swings_zigzag(hL, lL, aL, 2.0)
    i1 = np.searchsorted(H["idx"].values, dL.index.values, side="right") - 1
    conf = np.maximum(i1 - 1, 0)
    tr = (H["up"] & H["dow"])[conf] if use_dow else H["up"][conf]
    tr = tr & (i1 >= 1)
    e80 = H["e80"][conf]; a1 = H["atr"][conf]
    bar_d = (dL.index[1] - dL.index[0]).total_seconds() / 86400.0
    win_max = int(WIN_MAX_D / bar_d)
    recs = []
    wopen, w0, minlow, rh = False, -1, np.inf, None
    sp = 0
    for j in range(len(cL)):
        while sp < len(sw) and sw[sp][0] <= j:
            if sw[sp][3] == 1 and wopen and sw[sp][0] >= w0:
                rh = sw[sp][2]
            sp += 1
        if not tr[j] or np.isnan(a1[j]):
            wopen = False
            continue
        if not wopen:
            if lL[j] <= e80[j] + TOUCH_K * a1[j]:
                wopen, w0, minlow, rh = True, j, lL[j], None
            continue
        minlow = min(minlow, lL[j])
        if j - w0 > win_max:
            recs.append((w0, None, minlow, e80[w0], a1[w0], e80[w0], a1[w0]))
            wopen = False
            continue
        if rh is not None and cL[j] > rh:
            t = dL.index[j]
            if t.weekday() == 4 and t.hour >= 20:
                wopen = False
                continue
            recs.append((w0, j, minlow, e80[w0], a1[w0], e80[j], a1[j]))
            wopen = False
    return recs, (hL, lL, cL)


def walk_rr2(entries, arrs, rt, span, tag):
    h, l, c = arrs
    rows, last_x = [], -1
    for (j, e, stop) in entries:
        if j <= last_x or e - stop <= 0:
            continue
        risk = e - stop
        tgt = e + 2.0 * risk
        R, xj = None, min(j + FWD, len(c) - 1)
        for k in range(j + 1, min(j + 1 + FWD, len(c))):
            if l[k] <= stop: R, xj = -1.0, k; break
            if h[k] >= tgt:  R, xj = 2.0, k; break
        if R is None:
            R = (c[xj] - e) / risk
        rows.append(R - rt / risk)
        last_x = xj
    Rn = np.array(rows)
    if len(Rn) < 15:
        print(f"  {tag:<34} n={len(Rn)} few"); return
    pf = Rn[Rn > 0].sum() / max(1e-9, abs(Rn[Rn <= 0].sum()))
    print(f"  {tag:<34} N/yr={len(Rn)/span:5.1f} win={(Rn>0).mean()*100:4.1f}% "
          f"PF={pf:4.2f} meanR={Rn.mean():+.3f} totR/yr={Rn.sum()/span:+5.1f}")


def stage3(name, dL, H, rt, span):
    print(f"[STAGE3 {name}] LTFトリガー分解 (RR2固定, net)")
    for use_dow, tag in [(True, "dowあり"), (False, "dowなし(GCのみ)")]:
        recs, arrs = windows(dL, H, use_dow)
        h, l, c = arrs
        fired = [r for r in recs if r[1] is not None]
        # variant T: trigger entry
        eT = [(jT, c[jT], min(ml, e80t) - 0.05 * a1t) for (j0, jT, ml, e80o, a1o, e80t, a1t) in fired]
        # variant A: enter at touch (window-open bar close), stop w/ that bar's low
        eA = [(j0, c[j0], min(l[j0], e80o) - 0.05 * a1o) for (j0, jT, ml, e80o, a1o, e80t, a1t) in recs]
        if use_dow:
            tax = [(c[jT] - c[j0]) / a1o for (j0, jT, ml, e80o, a1o, e80t, a1t) in fired if a1o > 0]
            print(f"  window数={len(recs)} 発火率={len(fired)/max(1,len(recs))*100:.0f}%  "
                  f"確信の税 med={np.median(tax):+.2f}ATR sd={np.std(tax):.2f}")
        walk_rr2(eT, arrs, rt, span, f"[T] 戻り高値ブレイク確認 ({tag})")
        if use_dow:
            walk_rr2(eA, arrs, rt, span, f"[A] タッチ即エントリー ({tag})")


def main():
    g5 = load_mt5_csv("data/vantage_xauusd_m5.csv").loc["2018-09-14":]
    g1 = resample(g5, "60min")
    Hg = prep_1h(g1)
    span = (g5.index[-1] - g5.index[0]).days / 365.25
    print(f"===== GOLD (1h context + 5m trigger, {span:.1f}yr, net $0.3) =====")
    stage12("GOLD", g1, Hg)
    stage3("GOLD", g5, Hg, 0.3, span)

    b = load_mt5_csv("data/vantage_btcusd_m15.csv")
    cnt = b.groupby(b.index.date).size()
    okd = cnt[cnt.rolling(30).median() >= 80]
    b15 = resample(b[b.index.date >= okd.index[0]], "15min")
    b1 = resample(b15, "60min")
    Hb = prep_1h(b1)
    span = (b15.index[-1] - b15.index[0]).days / 365.25
    print(f"\n===== BTC (1h context + 15m trigger, {span:.1f}yr, net $15) =====")
    stage12("BTC", b1, Hb)
    stage3("BTC", b15, Hb, 15.0, span)


if __name__ == "__main__":
    main()
