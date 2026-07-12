"""Perfect Order as a SIGNAL (not a gate): when the EMA stack ESTABLISHES into
perfect order (fast>mid>slow, rising for long / fast<mid<slow, falling for short),
which way does price go? = the "don't fight the trend" continuation edge.
Per verification order: forward MFE/MAE (which way) first, then meanR at RR.
EMA(20,50,100). Entry = next bar open on the establishment bar. Both sides.
gold/BTC (trend-followers) vs USDJPY (mean-reverter) as the control. H1 & 4H."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}

def run(name, d, cost, EM=(20, 50, 100), FWD=50):
    cl = d["close"]
    ef, em, es = (cl.ewm(span=n, adjust=False).mean().values for n in EM)
    at = ta.atr(d["high"], d["low"], d["close"], 14).values
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    n = len(c)
    po_long = (ef > em) & (em > es) & (ef > np.concatenate([[ef[0]], ef[:-1]]))
    po_short = (ef < em) & (em < es) & (ef < np.concatenate([[ef[0]], ef[:-1]]))
    out = {}
    for side, est in (("long", po_long), ("short", po_short)):
        rows = []
        for s in range(max(EM) + 2, n - 1):
            if not (est[s] and not est[s - 1]): continue          # NEW establishment (transition)
            if np.isnan(at[s]) or at[s] <= 0: continue
            e = o[s + 1]; sd = at[s]
            # forward MFE/MAE in ATR over FWD bars
            seg = slice(s + 1, min(s + 1 + FWD, n))
            if side == "long":
                mfe = (h[seg].max() - e) / sd; mae = (e - l[seg].min()) / sd
            else:
                mfe = (e - l[seg].min()) / sd; mae = (h[seg].max() - e) / sd
            # meanR at RR1/RR2 with 1ATR stop (intrabar, stop-first)
            def rr(RR):
                if side == "long":
                    stop = e - sd; tgt = e + RR * sd
                    for j in range(s + 1, min(s + 1 + 300, n)):
                        if l[j] <= stop: return -1.0 - cost / sd
                        if h[j] >= tgt: return RR - cost / sd
                    return (c[min(s + 300, n - 1)] - e) / sd - cost / sd
                else:
                    stop = e + sd; tgt = e - RR * sd
                    for j in range(s + 1, min(s + 1 + 300, n)):
                        if h[j] >= stop: return -1.0 - cost / sd
                        if l[j] <= tgt: return RR - cost / sd
                    return (e - c[min(s + 300, n - 1)]) / sd - cost / sd
            rows.append((mfe, mae, rr(1.0), rr(2.0)))
        if len(rows) < 12:
            print(f"  {name:<14} {side:<5} n={len(rows)} too few"); continue
        A = np.array(rows); mfe, mae, r1, r2 = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
        print(f"  {name:<14} {side:<5} n={len(A):>4} | MFE/MAE med={np.median(mfe):.2f}/{np.median(mae):.2f} "
              f"ratio={np.median(mfe)/np.median(mae):.2f} | RR1 meanR={r1.mean():+.3f} win={(r1>0).mean()*100:.0f}% "
              f"| RR2 meanR={r2.mean():+.3f} win={(r2>0).mean()*100:.0f}%")

gold = load_mt5_csv("data/vantage_xauusd_h1.csv")
btc = load_mt5_csv("data/vantage_btcusd_h1.csv")
jpy = load_mt5_csv("data/vantage_usdjpy_h1.csv")
INST = [("GOLD", gold, 0.40), ("BTC", btc, 15.0), ("USDJPY", jpy, 0.015)]
print("MFE/MAE ratio >1.2 = continuation edge; ~1.0 = no edge. (EMA 20/50/100)\n")
for tf in ("1H", "4H"):
    print(f"===== {tf} =====")
    for label, d, cost in INST:
        dd = d if tf == "1H" else d.resample("4H").agg(AGG).dropna()
        run(label, dd, cost)
    print()
