"""Bar-CONTEXT strata on the CONFIRMED fib-bounce entry (the user's actual read):
event = touch of fib zone then a bar CLOSES back above (reaction confirmation);
entry at that confirmation close. Features: (a) confirmation-bar close position in its
range (clpos), (b) touch-bar lower-wick ratio (hammer-ness), (c) bars touch->confirm (k),
(d) approach speed into the touch (bars from z+1ATR to touch; fast = V-plunge).
Race +-0.5/+-1.0 ATR vs same-hour beta. USDJPY 15m + GOLD 15m."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag

WATCH, K = 96, 96
def screen(name, d):
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    atr = ta.atr(d["high"], d["low"], d["close"], 14).shift(1).values
    n = len(c); span = (d.index[-1]-d.index[0]).days/365.25
    races = {}
    for B in (0.5, 1.0):
        t_up = np.full(n, K+1, np.int32); t_dn = np.full(n, K+1, np.int32)
        for k in range(1, K+1):
            hs, ls = np.empty(n), np.empty(n)
            hs[:n-k], ls[:n-k] = h[k:], l[k:]
            hs[n-k:], ls[n-k:] = -np.inf, np.inf
            t_up = np.where((t_up > K) & (hs >= c + B*atr), k, t_up)
            t_dn = np.where((t_dn > K) & (ls <= c - B*atr), k, t_dn)
        races[B] = (np.minimum(t_up, t_dn) <= K) & (t_up < t_dn)
    valid = ~np.isnan(atr) & (np.arange(n) < n-K)
    hours = d.index.hour.values
    beta = {B: {hh: races[B][valid & (hours == hh)].mean() for hh in range(24)} for B in races}
    sw = swings_zigzag(h, l, np.where(np.isnan(atr), np.nanmean(atr), atr), 2.0)

    evs = []   # (j_confirm, clpos, wick, kbars, approach)
    for t in range(1, len(sw)):
        cH, iH, pH, kH = sw[t]; cL, iL, pL, kL = sw[t-1]
        if not (kH == 1 and kL == -1 and pH > pL): continue
        if cH >= n-K or np.isnan(atr[cH]) or atr[cH] <= 0: continue
        a0 = atr[cH]
        for f in (0.382, 0.5, 0.618):
            z = pH - f*(pH - pL)
            jt = None
            for j in range(cH+1, min(cH+WATCH, n-K)):
                if l[j] <= z: jt = j; break
                if l[j] < pL: break
            if jt is None: continue
            jc = None
            for j in range(jt, min(jt+8, n-K)):
                if c[j] < z - 0.75*a0: break            # plunged through -> void
                if c[j] > z: jc = j; break
            if jc is None: continue
            rngc = max(h[jc]-l[jc], 1e-9)
            clpos = (c[jc]-l[jc])/rngc
            rngt = max(h[jt]-l[jt], 1e-9)
            wick = (min(o[jt], c[jt]) - l[jt])/rngt
            back = 0
            for b in range(1, 25):
                if jt-b < 0 or h[jt-b] >= z + 1.0*a0: back = b; break
            evs.append((jc, clpos, wick, jc-jt, back))
    idx = np.array([e[0] for e in evs])
    clp = np.array([e[1] for e in evs]); wck = np.array([e[2] for e in evs])
    kb = np.array([e[3] for e in evs]); ap = np.array([e[4] for e in evs])
    print(f"\n===== {name} ({span:.1f}yr) CONFIRMED fib-bounce entry, bar-context strata =====")
    def row(tag, m):
        if m.sum() < 40: print(f"  {tag:<28} n={m.sum()} too few"); return
        line = f"  {tag:<28} n={m.sum():5d}"
        for B in (0.5, 1.0):
            w = races[B][idx[m]].mean()*100
            b = np.mean([beta[B][hh] for hh in hours[idx[m]]])*100
            line += f"  ±{B}: {w:4.1f} (b{b:4.1f}, {w-b:+4.1f})"
        print(line)
    row("all confirmed", np.ones(len(idx), bool))
    row("確認足強い (clpos>=0.7)", clp >= 0.7)
    row("確認足弱い (clpos<0.5)", clp < 0.5)
    row("タッチ足ハンマー (下ヒゲ>=0.5)", wck >= 0.5)
    row("同足確認 (k=0, ヒゲで拒否)", kb == 0)
    row("k=1-3 (数本もみ後に確認)", (kb >= 1) & (kb <= 3))
    row("V字突入 (approach<=6本)", (ap > 0) & (ap <= 6))
    row("緩やか접근 (>12本 or 無)", (ap == 0) | (ap > 12))
    row("裁量束: clpos>=0.7 & ハンマー", (clp >= 0.7) & (wck >= 0.5))

jp = load_mt5_csv("data/vantage_usdjpy_m15.csv")
cnt = jp.groupby(jp.index.date).size()
ok = cnt[cnt.rolling(30).median() >= 80]
screen("USDJPY 15m", jp[jp.index.date >= ok.index[0]])
gd = load_mt5_csv("data/vantage_xauusd_m5.csv").loc["2018-09-14":]
screen("GOLD 15m", gd.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna())
