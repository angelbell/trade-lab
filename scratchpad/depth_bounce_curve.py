"""depth_bounce_curve.py -- does BOUNCE RATE vary with pullback DEPTH (% of the last
impulse), and does a structure LINE at that depth add anything?

Ledger state: on USDJPY the depth curve is flat (fib 0.382-0.618 + deep 0.70-0.85 all
= beta) and lines are info-zero in every form. But gold/BTC never got an explicit
depth-stratified bounce curve, and one hint exists (VWAP work: SHALLOW reclaims were
actively bad). Direct test:

Event (causal): confirmed ZigZag up-impulse L0->H1 (at H1's confirm bar). As the pullback
falls, the FIRST bar whose low enters each depth bucket d=(H1-low)/(H1-L0) fires one event
per bucket per impulse. Void once low < L0 (impulse origin broken). Split each bucket by
LINE presence = any PRIOR confirmed swing low within 0.3*ATR of the bucket's touch price.
Measure: +-1 ATR barrier race win% from the touch bar's close vs SAME-HOUR beta.
Read: bounce-verification order step 1 (rate only); no targets/stops swept.
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

K_RACE = 96
BUCKETS = [(0.25, 0.40), (0.40, 0.55), (0.55, 0.70), (0.70, 0.85), (0.85, 1.00)]
TOL = 0.3          # "line here" = prior swing low within TOL*ATR of the touch price


def screen(name, d):
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    atr = ta.atr(d["high"], d["low"], d["close"], 14).shift(1).values
    n = len(c)
    span = (d.index[-1] - d.index[0]).days / 365.25

    # +-1 ATR barrier race, vectorized (same construction as fib_confluence_bounce)
    up_l, dn_l = c + atr, c - atr
    t_up = np.full(n, K_RACE + 1, np.int32)
    t_dn = np.full(n, K_RACE + 1, np.int32)
    for k in range(1, K_RACE + 1):
        hs, ls = np.empty(n), np.empty(n)
        hs[:n - k], ls[:n - k] = h[k:], l[k:]
        hs[n - k:], ls[n - k:] = -np.inf, np.inf
        t_up = np.where((t_up > K_RACE) & (hs >= up_l), k, t_up)
        t_dn = np.where((t_dn > K_RACE) & (ls <= dn_l), k, t_dn)
    win_long = (np.minimum(t_up, t_dn) <= K_RACE) & (t_up < t_dn)
    valid = ~np.isnan(atr) & (np.arange(n) < n - K_RACE)
    hours = d.index.hour.values
    bL = {hh: win_long[valid & (hours == hh)].mean() for hh in range(24)}

    sw = swings_zigzag(h, l, np.where(np.isnan(atr), np.nanmean(atr), atr), 2.0)
    low_pool = [(cc, pp) for cc, ii, pp, kk in sw if kk == -1]   # (confirm_bar, price)

    evs = []          # (bar, depth_bucket_idx, has_line)
    for t in range(1, len(sw)):
        cH, iH, pH, kH = sw[t]
        cL, iL, pL, kL = sw[t - 1]
        if not (kH == 1 and kL == -1 and pH > pL):
            continue
        if cH >= n - K_RACE or np.isnan(atr[cH]) or atr[cH] <= 0:
            continue
        rng_ = pH - pL
        prior = [p for cc, p in low_pool if cc < cH][-30:]
        end = sw[t + 1][0] if t + 1 < len(sw) else min(cH + 400, n - K_RACE)
        fired = set()
        for j in range(cH + 1, min(end, n - K_RACE)):
            if l[j] < pL:                    # origin broken -> impulse void
                break
            dep = (pH - l[j]) / rng_
            for bi, (lo_, hi_) in enumerate(BUCKETS):
                if bi in fired or dep < lo_:
                    continue
                lvl = pH - lo_ * rng_        # the bucket's entry price level
                line = any(abs(p - lvl) <= TOL * atr[j] for p in prior)
                evs.append((j, bi, line))
                fired.add(bi)

    print(f"\n===== {name} ({span:.1f}yr)  impulse events, race +-1ATR {K_RACE} bars =====")
    idx = np.array([e[0] for e in evs])
    bkt = np.array([e[1] for e in evs])
    lin = np.array([e[2] for e in evs])
    print(f"  {'深さ帯':>12} {'n':>6} {'win%':>6} {'beta%':>6} {'diff':>6} | "
          f"{'線あり n/win/diff':>20} | {'線なし n/win/diff':>20}")
    for bi, (lo_, hi_) in enumerate(BUCKETS):
        m = bkt == bi
        if m.sum() < 30:
            print(f"  {lo_:.2f}-{hi_:.2f} n={m.sum()} few"); continue
        w = win_long[idx[m]].mean()
        b = np.mean([bL[hh] for hh in hours[idx[m]]])
        cells = []
        for lm, tag in [(m & lin, "線あり"), (m & ~lin, "線なし")]:
            if lm.sum() < 30:
                cells.append(f"{lm.sum():>5} few        ")
                continue
            wl = win_long[idx[lm]].mean()
            bl_ = np.mean([bL[hh] for hh in hours[idx[lm]]])
            cells.append(f"{lm.sum():>5} {wl*100:4.1f} {(wl-bl_)*100:+5.1f}pt")
        print(f"  {lo_:>4.2f}-{hi_:.2f} {m.sum():>6} {w*100:>6.1f} {b*100:>6.1f} "
              f"{(w-b)*100:>+5.1f}pt | {cells[0]:>20} | {cells[1]:>20}")


def main():
    b = load_mt5_csv("data/vantage_btcusd_m15.csv")
    cnt = b.groupby(b.index.date).size()
    okd = cnt[cnt.rolling(30).median() >= 80]
    screen("BTC 15m", resample(b[b.index.date >= okd.index[0]], "15min"))
    g = load_mt5_csv("data/vantage_xauusd_m5.csv").loc["2018-09-14":]
    screen("GOLD 15m", resample(g, "15min"))


if __name__ == "__main__":
    main()
