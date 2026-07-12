"""User's refined method, causal version: a fib level that ALREADY produced one observed
bounce ('respected' -> the re-drawn line) -- does its SECOND touch react better?
touch1: first l<=z after H1 confirm. bounce = high reaches z+0.5ATR within 12 bars before
close < z-0.25ATR. If bounced: wait for price to leave (>=0.5ATR above z), then touch2 =
next l<=z. Event at touch2 close; race +-0.5/+-1.0 ATR vs same-hour beta.
Controls: FAKE random-depth lines through the SAME two-stage logic (kills the
'draw the line where it bounced' survivorship). Strata at touch2: RSI14<=35, 4h trend up.
USDJPY 15m + GOLD 15m."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag
from radar_gate_race import comps_tf

WATCH, K = 192, 96
def screen(name, d, seed=7):
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    atr = ta.atr(d["high"], d["low"], d["close"], 14).shift(1).values
    rsi = ta.rsi(d["close"], 14).values
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
    C4 = comps_tf(d, "240min"); tr_up = C4["stack"] > 0
    sw = swings_zigzag(h, l, np.where(np.isnan(atr), np.nanmean(atr), atr), 2.0)
    rng = np.random.default_rng(seed)

    def second_touches(levels_fn):
        evs = []
        for t in range(1, len(sw)):
            cH, iH, pH, kH = sw[t]; cL, iL, pL, kL = sw[t-1]
            if not (kH == 1 and kL == -1 and pH > pL): continue
            if cH >= n-K-WATCH or np.isnan(atr[cH]) or atr[cH] <= 0: continue
            a0 = atr[cH]
            for z in levels_fn(pH, pL):
                j1 = None
                for j in range(cH+1, min(cH+WATCH, n-K)):
                    if l[j] <= z: j1 = j; break
                    if l[j] < pL: break
                if j1 is None: continue
                bounced = False; jb = None
                for j in range(j1, min(j1+12, n-K)):
                    if c[j] < z - 0.25*a0: break
                    if h[j] >= z + 0.5*a0: bounced = True; jb = j; break
                if not bounced: continue
                left = False; j2 = None
                for j in range(jb+1, min(cH+WATCH, n-K)):
                    if not left and h[j] >= z + 0.5*a0: left = True
                    if left and l[j] <= z: j2 = j; break
                if j2 is None: continue
                evs.append(j2)
        return np.array(sorted(set(evs)))

    real = second_touches(lambda pH, pL: [pH - f*(pH-pL) for f in (0.382, 0.5, 0.618)])
    fake = second_touches(lambda pH, pL: [pH - u*(pH-pL) for u in rng.uniform(0.30, 0.90, 3)])
    print(f"\n===== {name} ({span:.1f}yr) SECOND touch of a RESPECTED level (long) =====")
    def row(tag, idx):
        if len(idx) < 40: print(f"  {tag:<26} n={len(idx)} too few"); return
        line = f"  {tag:<26} n={len(idx):5d} N/yr={len(idx)/span:4.0f}"
        for B in (0.5, 1.0):
            w = races[B][idx].mean()*100
            b = np.mean([beta[B][hh] for hh in hours[idx]])*100
            line += f"  ±{B}: {w:4.1f} (b{b:4.1f}, {w-b:+4.1f})"
        print(line)
    row("real fib 2nd touch", real)
    row("  + RSI14<=35", real[rsi[real] <= 35])
    row("  + 4hトレンド上", real[tr_up[real]])
    row("  + RSI<=35 & 4h上", real[(rsi[real] <= 35) & tr_up[real]])
    row("FAKE 2nd touch (乱数線)", fake)
    row("  FAKE + RSI<=35 & 4h上", fake[(rsi[fake] <= 35) & tr_up[fake]])

jp = load_mt5_csv("data/vantage_usdjpy_m15.csv")
cnt = jp.groupby(jp.index.date).size()
ok = cnt[cnt.rolling(30).median() >= 80]
screen("USDJPY 15m", jp[jp.index.date >= ok.index[0]])
gd = load_mt5_csv("data/vantage_xauusd_m5.csv").loc["2018-09-14":]
screen("GOLD 15m", gd.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna())
