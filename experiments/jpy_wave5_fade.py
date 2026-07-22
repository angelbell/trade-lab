"""jpy_wave5_fade.py -- B8 STEP1-2: USDJPY Elliott wave-5 exhaustion FADE screen.

Event (causal): ZigZag(zz-k2) confirmed-pivot sequence L0,H1,L2,H3,L4,H5 with
H5>H3>H1 and L4>L2>L0 (bullish 5-wave impulse). Trigger = the CONFIRMATION bar of H5
(i.e., price has reversed 2xATR off the wave-5 top -- the ZigZag confirmation IS the
reversal evidence). Fade short from that bar's close. Mirror for bearish 5-wave -> long.
STEP1: +-1 ATR barrier race (favorable side first; tie = loss) vs SAME-HOUR beta of the
same directional race. TF ladder 15m (m15 file, 8.4yr) / 1h / 4h (h1 file, 16yr).
STEP2 strata (15m): impulse size (H5-L0)/ATR, wave-5 stretch (H5-H3)/(H3-L2),
running day-range vs daily ATR (intervention proxy). Per-year for 15m.
PASS bar (B8): diff >= +5pt or MFE/MAE >= 1.3 downstream. Predicted kill: reversal
already priced (BB-class MR rediscovery) or n too thin.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}


def screen(name, d, k_race):
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    atr = ta.atr(d["high"], d["low"], d["close"], 14).shift(1).values
    n = len(c)
    span = (d.index[-1] - d.index[0]).days / 365.25
    up_l, dn_l = c + atr, c - atr
    t_up = np.full(n, k_race + 1, np.int32)
    t_dn = np.full(n, k_race + 1, np.int32)
    for k in range(1, k_race + 1):
        hs, ls = np.empty(n), np.empty(n)
        hs[:n - k], ls[:n - k] = h[k:], l[k:]
        hs[n - k:], ls[n - k:] = -np.inf, np.inf
        t_up = np.where((t_up > k_race) & (hs >= up_l), k, t_up)
        t_dn = np.where((t_dn > k_race) & (ls <= dn_l), k, t_dn)
    win_long = (np.minimum(t_up, t_dn) <= k_race) & (t_up < t_dn)
    win_short = (np.minimum(t_up, t_dn) <= k_race) & (t_dn < t_up)
    valid = ~np.isnan(atr) & (np.arange(n) < n - k_race)
    hours = d.index.hour.values
    bL = {hh: win_long[valid & (hours == hh)].mean() for hh in range(24)}
    bS = {hh: win_short[valid & (hours == hh)].mean() for hh in range(24)}

    sw = swings_zigzag(h, l, np.where(np.isnan(atr), np.nanmean(atr), atr), 2.0)
    ev_s, ev_l = [], []          # (bar, ext_atr, w5_ratio)
    for t in range(5, len(sw)):
        cN, iN, pN, kN = sw[t]
        if cN >= n - k_race or np.isnan(atr[cN]) or atr[cN] <= 0:
            continue
        seq = sw[t - 5:t + 1]
        if kN == 1:              # potential H5: [L0,H1,L2,H3,L4,H5]
            L0, H1, L2, H3, L4, H5 = (s[2] for s in seq)
            kinds = [s[3] for s in seq]
            if kinds == [-1, 1, -1, 1, -1, 1] and H5 > H3 > H1 and L4 > L2 > L0:
                ext = (H5 - L0) / atr[cN]
                w5 = (H5 - H3) / max(H3 - L2, 1e-9)
                ev_s.append((cN, ext, w5))
        else:                    # potential L5 (bearish impulse): [H0,L1,H2,L3,H4,L5]
            H0, L1, H2, L3, H4, L5 = (s[2] for s in seq)
            kinds = [s[3] for s in seq]
            if kinds == [1, -1, 1, -1, 1, -1] and L5 < L3 < L1 and H4 < H2 < H0:
                ext = (H0 - L5) / atr[cN]
                w5 = (L3 - L5) / max(L1 - L3, 1e-9)
                ev_l.append((cN, ext, w5))

    def rep(tag, ev, win, bmap):
        if len(ev) < 25:
            print(f"  {tag:<26} n={len(ev)} too few")
            return None
        idx = np.array([e[0] for e in ev])
        w = win[idx]
        b = np.mean([bmap[hh] for hh in hours[idx]])
        print(f"  {tag:<26} n={len(idx):5d} N/yr={len(idx)/span:5.0f}  win={w.mean()*100:4.1f}%"
              f"  beta={b*100:4.1f}%  diff={(w.mean()-b)*100:+4.1f}pt")
        return idx, w

    print(f"\n===== {name} ({span:.1f}yr, 5-wave confirm fade, race {k_race} bars) =====")
    rs = rep("bull 5-wave -> SHORT", ev_s, win_short, bS)
    rl = rep("bear 5-wave -> LONG", ev_l, win_long, bL)
    return d, ev_s, ev_l, win_short, win_long, bS, bL, hours, span


def main():
    m15 = load_mt5_csv("data/vantage_usdjpy_m15.csv")
    cnt = m15.groupby(m15.index.date).size()
    ok = cnt[cnt.rolling(30).median() >= 80]
    m15 = m15[m15.index.date >= ok.index[0]]
    h1 = load_mt5_csv("data/vantage_usdjpy_h1.csv")

    d, ev_s, ev_l, wS, wL, bS, bL, hours, span = screen("USDJPY 15m", m15, 96)
    screen("USDJPY 1h", h1, 48)
    screen("USDJPY 4h", pd.DataFrame({k: getattr(h1[k].resample("4h"), v)() for k, v in
           [("open","first"),("high","max"),("low","min"),("close","last")]}).dropna(), 30)

    # ---------- STEP2 strata on 15m (pooled short+long, direction-favorable win) ----------
    print("\n===== STEP2 strata (15m, short+long pooled) =====")
    idx = np.array([e[0] for e in ev_s] + [e[0] for e in ev_l])
    ext = np.array([e[1] for e in ev_s] + [e[1] for e in ev_l])
    w5r = np.array([e[2] for e in ev_s] + [e[2] for e in ev_l])
    w = np.concatenate([wS[[e[0] for e in ev_s]], wL[[e[0] for e in ev_l]]])
    bmaps = [bS] * len(ev_s) + [bL] * len(ev_l)
    b_ind = np.array([bm[hh] for bm, hh in zip(bmaps, d.index.hour.values[idx])])
    # intervention proxy: running day range up to the event bar vs daily ATR14
    day = pd.Series(d.index.date, index=d.index)
    run_hi = d["high"].groupby(day).cummax().values
    run_lo = d["low"].groupby(day).cummin().values
    datr = ta.atr(d["high"].resample("1D").max(), d["low"].resample("1D").min(),
                  d["close"].resample("1D").last().dropna(), 14).shift(1)
    datr_b = datr.reindex(d.index, method="ffill").values
    dr = (run_hi - run_lo) / datr_b
    for tag, m in [("ext>=median", ext >= np.median(ext)), ("ext<median", ext < np.median(ext)),
                   ("wave5伸び>=0.6", w5r >= 0.6), ("wave5伸び<0.6", w5r < 0.6),
                   ("当日レンジ<1.0xATRd", dr[idx] < 1.0), ("当日レンジ>=1.5 (介入級)", dr[idx] >= 1.5)]:
        if m.sum() < 25:
            print(f"  {tag:<22} n={m.sum()} too few"); continue
        print(f"  {tag:<22} n={m.sum():4d}  win={w[m].mean()*100:4.1f}%  beta={b_ind[m].mean()*100:4.1f}%"
              f"  diff={(w[m].mean()-b_ind[m].mean())*100:+4.1f}pt")
    yr = d.index.year.values[idx]
    line = []
    for y in np.unique(yr):
        m = yr == y
        if m.sum() >= 15:
            line.append(f"{y}:{(w[m].mean()-b_ind[m].mean())*100:+.0f}(n={m.sum()})")
    print("  per-year diff: " + "  ".join(line))


if __name__ == "__main__":
    main()
