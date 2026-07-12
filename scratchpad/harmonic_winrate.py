"""Per-signal WIN RATE of the Multi-ZigZag Harmonic patterns, mechanized CAUSALLY
(faithful port of the Pine zigzag + ratio templates; act only on CONFIRMED pivots,
= waitForConfirmation). Each pattern's D = reversal entry (long if D is the low, short
if high), stop just beyond D, RR1 barrier. KEY baseline = 'reverse at EVERY confirmed
pivot' (no pattern filter): if harmonics don't beat it, the Fib templates add nothing."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

ERR = 10; emin = (100 - ERR) / 100; emax = (100 + ERR) / 100

def causal_pivots(high, low, length):
    """Pine pivots(): current bar's high is the highest of the last `length` bars
    (backward window only = causal). dir carries until a clean high/low flips it."""
    n = len(high); dirv = np.zeros(n, int)
    rmax = pd.Series(high).rolling(length).max().values
    rmin = pd.Series(low).rolling(length).min().values
    phigh = np.where(high >= rmax, high, np.nan)
    plow = np.where(low <= rmin, low, np.nan)
    d = 0
    for s in range(n):
        ph = not np.isnan(phigh[s]); pl = not np.isnan(plow[s])
        if ph and not pl: d = 1
        elif pl and not ph: d = -1
        dirv[s] = d
    return dirv, phigh, plow

def build_zigzag(high, low, length):
    """faithful port of zigzag(): returns confirmed pivot arrays (price,bar,dir),
    newest-first, evolving; we snapshot at each bar for causal pattern detection."""
    dirv, phigh, plow = causal_pivots(high, low, length)
    piv, pb, pd_ = [], [], []
    snaps = {}                                   # bar -> copy of pivot list (newest-first)
    prevdir = 0
    for s in range(len(high)):
        ph = not np.isnan(phigh[s]); pl = not np.isnan(plow[s])
        dir = dirv[s]
        dirchanged = dir != prevdir; prevdir = dir
        if ph or pl:
            value = phigh[s] if dir == 1 else plow[s]
            bar = s; newDir = dir
            if (not dirchanged) and len(piv) >= 1:
                pivot = piv.pop(0); pivotbar = pb.pop(0); pivotdir = pd_.pop(0)
                useNew = value * pivotdir < pivot * pivotdir
                if useNew: value = pivot; bar = pivotbar
            if len(piv) >= 2:
                LastPoint = piv[1]
                newDir = dir * 2 if dir * value > dir * LastPoint else dir
            piv.insert(0, value); pb.insert(0, bar); pd_.insert(0, newDir)
            if len(piv) > 100:
                piv.pop(); pb.pop(); pd_.pop()
        snaps[s] = (list(piv), list(pb), list(pd_))
    return snaps

def match(P, B, D):
    """P=prices newest-first from index `start`; returns set of pattern names matching."""
    d, c, b, a, x, y = P[0], P[1], P[2], P[3], P[4], P[5]
    hi = max(x, a, b, c, d); lo = min(x, a, b, c, d)
    xab = abs(b - a) / max(abs(x - a), 1e-9)
    abc = abs(c - b) / max(abs(a - b), 1e-9)
    bcd = abs(d - c) / max(abs(b - c), 1e-9)
    xad = abs(d - a) / max(abs(x - a), 1e-9)
    yxa = abs(a - x) / max(abs(y - x), 1e-9)
    out = set()
    inb = lo < b < hi
    if inb and xab >= .618*emin and xab <= .618*emax and abc >= .382*emin and abc <= .886*emax and (bcd >= 1.272*emin and bcd <= 1.618*emax or xad >= .786*emin and xad <= .786*emax): out.add("Gartley")
    if inb and xab >= .382*emin and xab <= .618*emax and abc >= .382*emin and abc <= .886*emax and (bcd >= 2.24*emin and bcd <= 3.618*emax or xad >= 1.618*emin and xad <= 1.618*emax): out.add("Crab")
    if inb and xab >= .886*emin and xab <= .886*emax and abc >= .382*emin and abc <= .886*emax and (bcd >= 2.0*emin and bcd <= 3.618*emax or xad >= 1.618*emin and xad <= 1.618*emax): out.add("DeepCrab")
    if inb and xab >= .382*emin and xab <= .50*emax and abc >= .382*emin and abc <= .886*emax and (bcd >= 1.618*emin and bcd <= 2.618*emax or xad >= .886*emin and xad <= .886*emax): out.add("Bat")
    if inb and xab >= .786*emin and xab <= .786*emax and abc >= .382*emin and abc <= .886*emax and (bcd >= 1.618*emin and bcd <= 2.618*emax or xad >= 1.272*emin and xad <= 1.618*emax): out.add("Butterfly")
    if inb and abc >= 1.13*emin and abc <= 1.618*emax and bcd >= 1.618*emin and bcd <= 2.24*emax and xad >= .886*emin and xad <= 1.13*emax: out.add("Shark")
    if inb and xab >= .382*emin and xab <= .618*emax and abc >= 1.13*emin and abc <= 1.414*emax and (bcd >= 1.272*emin and bcd <= 2.0*emax or xad >= .786*emin and xad <= .786*emax): out.add("Cypher")
    if yxa >= .618*emin and yxa <= .618*emax and xab >= 1.27*emin and xab <= 1.618*emax and abc >= .618*emin and abc <= .618*emax and bcd >= 1.27*emin and bcd <= 1.618*emax: out.add("3Drive")
    if xab >= 1.13*emin and xab <= 1.618*emax and abc >= 1.618*emin and abc <= 2.24*emax and bcd >= .5*emin and bcd <= .5*emax: out.add("5-0")
    abcdDir = 1 if (a < b and a < c and c < b and c < d and a < d and b < d) else (-1 if (a > b and a > c and c > b and c > d and a > d and b > d) else 0)
    if abc >= .618*emin and abc <= .786*emax and bcd >= 1.272*emin and bcd <= 1.618*emax and abcdDir != 0: out.add("ABCD")
    tr = abs(B[2]-B[3]); cr = abs(B[0]-B[1]); pr = abs(c-d)/max(abs(a-b),1e-9); trat = (abs(B[0]-B[1]))/max(abs(B[2]-B[3]),1)
    if trat >= emin and trat <= emax and pr >= emin and pr <= emax and abcdDir != 0: out.add("AB=CD")
    if pr >= 1.272*emin and pr <= 1.618*emax and abc >= .618*emin and abc <= .786*emax and abcdDir != 0: out.add("ABCDext")
    return out

def run(name, d, length, cost, RR=1.0):
    o, h, l, c = (d[k].values for k in ("open","high","low","close"))
    at = ta.atr(d["high"], d["low"], d["close"], 14).values
    snaps = build_zigzag(h, l, length)
    start = 1
    res = {}; base = []                          # base = reversal at every new confirmed pivot
    last_dbar = -1
    for s in range(length+2, len(c)-1):
        piv, pb, pd_ = snaps[s]
        if len(piv) < 6 + start: continue
        dbar = pb[start]
        if dbar == last_dbar: continue
        last_dbar = dbar
        P = piv[start:start+6]; B = pb[start:start+6]
        dd = P[0]; cc = P[1]
        side = 1 if cc > dd else -1              # D is the low -> long(1); high -> short(-1)
        e = o[s+1]; sd = abs(e - dd) + 0.1*at[s]
        if sd <= 0: continue
        def barrier():
            if side == 1:
                stop = e - sd; tgt = e + RR*sd
                for j in range(s+1, min(s+1+200, len(c))):
                    if l[j] <= stop: return 0
                    if h[j] >= tgt: return 1
                return 0
            else:
                stop = e + sd; tgt = e - RR*sd
                for j in range(s+1, min(s+1+200, len(c))):
                    if h[j] >= stop: return 0
                    if l[j] <= tgt: return 1
                return 0
        w = barrier(); base.append(w)
        for pat in match(P, B, dd):
            res.setdefault(pat, []).append(w)
    print(f"\n== {name} H1  zigzag={length}  RR{RR} (breakeven win={100/(1+RR):.0f}%) ==")
    print(f"  {'BASELINE any-pivot':<14} n={len(base):>4} win={np.mean(base)*100:>4.1f}%")
    for pat in ["Gartley","Crab","DeepCrab","Bat","Butterfly","Shark","Cypher","3Drive","5-0","ABCD","AB=CD","ABCDext"]:
        if pat in res and len(res[pat]) >= 10:
            v = res[pat]; print(f"  {pat:<14} n={len(v):>4} win={np.mean(v)*100:>4.1f}%")

gold = load_mt5_csv("data/vantage_xauusd_h1.csv")
btc = load_mt5_csv("data/vantage_btcusd_h1.csv")
for length in (5, 10):
    run("GOLD", gold, length, 0.40)
    run("BTC", btc, length, 15.0)
