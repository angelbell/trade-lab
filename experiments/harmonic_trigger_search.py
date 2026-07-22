"""SEARCH for a pattern that beats the ~50% baseline by adding the real ENTRY TRIGGER
(the rule), not just the shape. Key compares, causal:
  base      = reverse at D directly (shown ~50%, coin flip)
  Wbottom   = two ~equal lows + neckline; enter on CONFIRMED close above neckline (long)
  Mtop      = mirror (short)
  conf-harm = harmonic D, but wait for close beyond C in reversal dir before entry
  swingBO   = plain: close beyond the last confirmed swing high/low (= gold_bo essence) ANCHOR
If the winners ≈ swingBO, the 'pattern' adds nothing over a breakout (lab law)."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
exec(open("/home/angelbell/dev/auto-trade/experiments/harmonic_winrate.py").read().split("def run(")[0])

TOL = 0.03      # two lows/highs "equal" within 3%
TRIGW = 40      # bars allowed for the trigger to fire

def outcome(o, h, l, c, at, ei, side, e, sd, RR, cost):
    if sd <= 0: return None
    if side == 1:
        stop = e - sd; tgt = e + RR * sd
        for j in range(ei, min(ei + 200, len(c))):
            if l[j] <= stop: return -1.0 - cost / sd
            if h[j] >= tgt: return RR - cost / sd
        return (c[min(ei + 200, len(c) - 1)] - e) / sd - cost / sd
    else:
        stop = e + sd; tgt = e - RR * sd
        for j in range(ei, min(ei + 200, len(c))):
            if h[j] >= stop: return -1.0 - cost / sd
            if l[j] <= tgt: return RR - cost / sd
        return (e - c[min(ei + 200, len(c) - 1)]) / sd - cost / sd

def run(name, d, length, cost, RR=1.0):
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    at = ta.atr(d["high"], d["low"], d["close"], 14).values
    snaps = build_zigzag(h, l, length)
    res = {k: [] for k in ("base", "Wbottom", "Mtop", "conf-harm", "swingBO", "anyLong", "longBO")}
    rng = np.random.default_rng(0)
    last = -1
    for s in range(length + 2, len(c) - 1):
        piv, pb, pdr = snaps[s]
        if len(piv) < 7: continue
        if pb[1] == last: continue
        last = pb[1]
        d0, c0, b0 = piv[1], piv[2], piv[3]; dd, dc, db = pdr[1], pdr[2], pdr[3]
        atr = at[s]
        # base: reverse at D
        side = 1 if c0 > d0 else -1
        e = o[s + 1]; sd = abs(e - d0) + 0.1 * atr
        r = outcome(o, h, l, c, at, s + 1, side, e, sd, RR, cost)
        if r is not None: res["base"].append(1 if r > 0 else 0)
        # BETA NULL: just go LONG at this confirmed pivot (no pattern), same RR1 geometry
        e2 = o[s + 1]; sd2 = atr + 0.1 * atr
        rL = outcome(o, h, l, c, at, s + 1, 1, e2, sd2, RR, cost)
        if rL is not None: res["anyLong"].append(1 if rL > 0 else 0)
        # swingBO anchor: close beyond last confirmed swing (D's prior opposite pivot = c0)
        if c0 > d0:   # D is a low, C is a high above -> breakout = close above C -> long
            for j in range(s + 1, min(s + 1 + TRIGW, len(c))):
                if c[j] > c0:
                    e = o[j + 1] if j + 1 < len(c) else c[j]; sd = abs(e - d0) + 0.1 * atr
                    r = outcome(o, h, l, c, at, j + 1, 1, e, sd, RR, cost)
                    if r is not None:
                        res["swingBO"].append(1 if r > 0 else 0); res["longBO"].append(1 if r > 0 else 0)
                    break
        else:
            for j in range(s + 1, min(s + 1 + TRIGW, len(c))):
                if c[j] < c0:
                    e = o[j + 1] if j + 1 < len(c) else c[j]; sd = abs(e - d0) + 0.1 * atr
                    r = outcome(o, h, l, c, at, j + 1, -1, e, sd, RR, cost)
                    if r is not None: res["swingBO"].append(1 if r > 0 else 0)
                    break
        # conf-harm: at D wait for close beyond C in reversal dir
        if c0 > d0:
            for j in range(s + 1, min(s + 1 + TRIGW, len(c))):
                if c[j] > c0:
                    e = o[j + 1] if j + 1 < len(c) else c[j]; sd = abs(e - d0) + 0.1 * atr
                    r = outcome(o, h, l, c, at, j + 1, 1, e, sd, RR, cost)
                    if r is not None: res["conf-harm"].append(1 if r > 0 else 0)
                    break
        # Wbottom: d & b two ~equal lows, c the neckline high between -> confirm close>c
        if dd < 0 and dc > 0 and db < 0 and abs(d0 - b0) / max(abs(d0), 1e-9) < TOL:
            neck = c0
            for j in range(s + 1, min(s + 1 + TRIGW, len(c))):
                if c[j] > neck:
                    e = o[j + 1] if j + 1 < len(c) else c[j]; sd = abs(e - min(d0, b0)) + 0.1 * atr
                    r = outcome(o, h, l, c, at, j + 1, 1, e, sd, RR, cost)
                    if r is not None: res["Wbottom"].append(1 if r > 0 else 0)
                    break
        # Mtop: d & b two ~equal highs, c the neckline low -> confirm close<c
        if dd > 0 and dc < 0 and db > 0 and abs(d0 - b0) / max(abs(d0), 1e-9) < TOL:
            neck = c0
            for j in range(s + 1, min(s + 1 + TRIGW, len(c))):
                if c[j] < neck:
                    e = o[j + 1] if j + 1 < len(c) else c[j]; sd = abs(e - max(d0, b0)) + 0.1 * atr
                    r = outcome(o, h, l, c, at, j + 1, -1, e, sd, RR, cost)
                    if r is not None: res["Mtop"].append(1 if r > 0 else 0)
                    break
    print(f"\n== {name} H1 zz{length} RR{RR} (breakeven win=50%) ==")
    for k in ("base", "anyLong", "longBO", "swingBO", "conf-harm", "Wbottom", "Mtop"):
        v = res[k]
        if len(v) >= 10: print(f"  {k:<10} n={len(v):>4} win={np.mean(v)*100:>4.1f}%")

gold = load_mt5_csv("data/vantage_xauusd_h1.csv")
btc = load_mt5_csv("data/vantage_btcusd_h1.csv")
for length in (5, 10):
    run("GOLD", gold, length, 0.40)
    run("BTC", btc, length, 15.0)
