"""nwe_vs_bb.py -- head-to-head: Nadaraya-Watson Envelope (LuxAlgo) vs Bollinger Bands, as a FADE.

Mechanizes the HONEST (non-repaint / endpoint) NWE = a causal Gaussian-weighted MA +- mult*MAE:
  out[t]  = sum_i gauss(i,h)*src[t-i] / sum_i gauss(i,h)   (causal FIR, fixed weights -> np.convolve)
  band    = out +- mult * SMA(|src-out|, L)
BB        = SMA(len) +- mult * std(len).
Same fade rule for both (apples-to-apples): LONG when close crosses BELOW the lower band, SHORT when
it crosses ABOVE the upper band; next-bar-open fill; exit = revert to the CENTER line or 1.5*ATR stop;
one position at a time; net of round-trip spread. Judge net@ real spreads + IS/OOS.

(The default repaint=true NWE is a TWO-SIDED kernel = lookahead; its pretty historical signals use
future bars and don't exist live. We test only the causal version -- the fair comparison to BB.)

  .venv/bin/python research/nwe_vs_bb.py --csv data/vantage_usdjpy_m5.csv --pip 0.01 --start 2018-06-01
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv

SL_ATR, MAXH, L = 1.5, 400, 499


def nwe_bands(src, h, mult):
    k = np.exp(-(np.arange(L) ** 2) / (2 * h * h)); k /= k.sum()
    out = np.convolve(src, k)[:len(src)]                      # causal: out[t]=sum_i k[i]*src[t-i]
    out[:L] = np.nan
    mae = pd.Series(np.abs(src - out)).rolling(L).mean().values * mult
    return out, out + mae, out - mae


def bb_bands(src, length, mult):
    s = pd.Series(src)
    basis = s.rolling(length).mean().values
    dev = mult * s.rolling(length).std(ddof=0).values
    return basis, basis + dev, basis - dev


def fade(d, center, upper, lower, pip):
    cv = d["close"].values; op = d["open"].values; hi = d["high"].values; lo = d["low"].values
    atr = ta.atr(d["high"], d["low"], d["close"], length=14).values
    yr = d.index.year.values; n = len(cv)
    xdn = (cv < lower) & np.r_[False, cv[:-1] >= lower[:-1]]   # cross below lower -> fade long
    xup = (cv > upper) & np.r_[False, cv[:-1] <= upper[:-1]]   # cross above upper -> fade short
    rows = []; busy = -1
    for i in range(n - 1):
        if not (xdn[i] or xup[i]) or i + 1 <= busy or np.isnan(atr[i + 1]) or np.isnan(center[i]):
            continue
        dr = 1 if xdn[i] else -1
        ei = i + 1; e = op[ei]; stop = e - dr * SL_ATR * atr[ei]; ex = None
        for j in range(ei + 1, min(ei + 1 + MAXH, n)):
            if dr > 0 and lo[j] <= stop: ex = stop; break
            if dr < 0 and hi[j] >= stop: ex = stop; break
            if dr > 0 and hi[j] >= center[j]: ex = center[j]; break    # revert to center = TP
            if dr < 0 and lo[j] <= center[j]: ex = center[j]; break
            busy = j
        if ex is None:
            ex = cv[min(ei + MAXH, n - 1)]; busy = min(ei + MAXH, n - 1)
        else:
            busy = j
        rows.append((yr[ei], (ex - e) / pip * dr))
    return pd.DataFrame(rows, columns=["y", "g"])


def pf(g, sp=0.0):
    gn = g - sp; w = gn[gn > 0].sum(); l = gn[gn < 0].sum()
    return w / abs(l) if l else float("inf")


def report(tag, t):
    if len(t) < 30:
        print(f"  {tag:<18} n={len(t)} (too few)"); return
    isr, oos = t[t.y < 2022].g, t[t.y >= 2022].g
    print(f"  {tag:<18} n={len(t):>5} win={(t.g>0).mean()*100:>3.0f}% gross={pf(t.g):.2f}  "
          f"net@0.5={pf(t.g,0.5):.2f} net@1.0={pf(t.g,1.0):.2f} net@1.9={pf(t.g,1.9):.2f}  "
          f"| IS@1.0={pf(isr,1.0):.2f} OOS@1.0={pf(oos,1.0):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/vantage_usdjpy_m5.csv")
    ap.add_argument("--pip", type=float, default=0.01)
    ap.add_argument("--start", default="2018-06-01")
    a = ap.parse_args()
    d = load_mt5_csv(a.csv).loc[a.start:]
    src = d["close"].values
    print(f"\n=== NWE(causal) vs BB fade  {os.path.basename(a.csv)}  {d.index[0].date()}->{d.index[-1].date()} "
          f" (spreads in pips; pip={a.pip}) ===")
    print("  -- Nadaraya-Watson Envelope (endpoint/causal), default h=8 mult=3 --")
    for h, m in [(8, 3.0), (8, 2.0), (5, 3.0)]:
        c, u, lw = nwe_bands(src, h, m)
        report(f"NWE h{h}/m{m}", fade(d, c, u, lw, a.pip))
    print("  -- Bollinger Bands (matched-ish), len20/34 mult2/2.5 --")
    for ln, m in [(20, 2.0), (34, 2.0), (20, 2.5)]:
        c, u, lw = bb_bands(src, ln, m)
        report(f"BB {ln}/m{m}", fade(d, c, u, lw, a.pip))


if __name__ == "__main__":
    main()
