"""Test the user's 経験則: 'most up-legs retrace ~30-50% before continuing.'
If true, a LIMIT at the pullback (vs market 飛び乗り) gives better price + more range
+ avoids spread = changes the low-TF cost economics. Causal ATR-ZigZag; per L->H up-leg,
measure retrace to the next L, and whether a later H exceeds it (continuation).
Report: among CONTINUING legs, median retrace% and %>=30/50; and the limit-at-50% economics."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}


def zigzag_atr(h, l, c, atr, k=3.0):
    piv = []; trend = 1; ext = h[0]; exti = 0
    for i in range(1, len(c)):
        thr = k * atr[i]
        if np.isnan(thr) or thr <= 0: continue
        if trend > 0:
            if h[i] > ext: ext = h[i]; exti = i
            elif ext - l[i] > thr: piv.append((exti, ext, 1)); trend = -1; ext = l[i]; exti = i
        else:
            if l[i] < ext: ext = l[i]; exti = i
            elif h[i] - ext > thr: piv.append((exti, ext, -1)); trend = 1; ext = h[i]; exti = i
    return piv


def measure(name, df, k=3.0):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    atr = ta.atr(df["high"], df["low"], df["close"], 14).values
    piv = zigzag_atr(h, l, c, atr, k)
    fracs, cont = [], []
    for j in range(len(piv) - 3):
        (ia, pa, ta_), (ih, ph, th), (ib, pb, tb), (ih2, ph2, th2) = piv[j], piv[j+1], piv[j+2], piv[j+3]
        if not (ta_ == -1 and th == 1 and tb == -1 and th2 == 1): continue   # L, H, L, H
        up = ph - pa
        if up <= 0: continue
        frac = (ph - pb) / up               # retrace depth as % of the up-leg
        fracs.append(frac); cont.append(ph2 > ph)   # did the next swing exceed the prior high?
    fracs = np.array(fracs); cont = np.array(cont)
    contfr = fracs[cont]                     # retrace of legs that CONTINUED
    # limit @ 50% retrace economics: of all legs, P(reach 50% AND continue) vs P(reach 50% AND fail)
    reach50 = fracs >= 0.5
    win50 = (reach50 & cont).sum(); lose50 = (reach50 & ~cont).sum()
    print(f"  {name:<12} n_legs={len(fracs):>4} cont-rate={cont.mean()*100:>3.0f}% | "
          f"retrace(of continuing): med={np.median(contfr)*100:>3.0f}% >=30%:{(contfr>=.3).mean()*100:>3.0f}% "
          f">=50%:{(contfr>=.5).mean()*100:>3.0f}% | limit@50%: fill {reach50.mean()*100:>3.0f}% of legs, "
          f"of those win {win50/(win50+lose50)*100 if win50+lose50 else 0:>3.0f}%")


gold = load_mt5_csv("data/vantage_xauusd_m5.csv")
btc = load_mt5_csv("data/vantage_btcusd_h1.csv")
print("pullback frequency (ZigZag k=3 ATR):  does 'most legs retrace 30-50%' hold?\n")
for tf, fr in [("1h", "60min"), ("4h", "240min"), ("1d", "1440min")]:
    print(f"== {tf} ==")
    measure("GOLD", gold.resample(fr).agg(AGG).dropna())
    measure("BTC", btc if fr == "60min" else btc.resample(fr).agg(AGG).dropna())
