"""wick_select.py -- can SELECTION turn the sub-cost wick-fade into a cost-surviving edge?

User's idea: the wick-fade dies on cost because it's thin; combining signals to SELECT a higher-conviction
subset might clear cost. Test honestly: for each candidate selector (volume percentile / stretch from
SMA50 / wick length / extreme strength), keep the top quantile and ask -- does it beat a RANDOM equal-size
subset of all wick trades (the random-drop null)? If not, the 'selection' is just n-trimming (luck-sorter),
not real conviction. At REALISTIC cost (0.05%).

Falsifier (up front): a selector PASSES only if its CAGR/DD beats the random-drop null >=90%ile AND meanR
is meaningfully positive at 0.05% cost AND it plateaus across the keep-quantile. Volume failing but stretch
passing => selection helps but volume isn't the lever. All failing => selection can't rescue a sub-cost edge.
In-sample; live-forward arbitrates.
  .venv/bin/python research/wick_select.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from research.volume_reversal_screen import resample, atr
from research.portfolio_kama import cagr_dd

SPLIT = 2018


def wick_trades(d, L=10, wick_k=1.0, rr=2.0, buf=0.2, fwd=24, cost=0.0005):
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    a = atr(d).values
    volpr = d["volume"].rolling(100).rank(pct=True).values
    sma = d["close"].rolling(50).mean().values
    hiL = pd.Series(h).rolling(L).max().values
    loL = pd.Series(l).rolling(L).min().values
    uwick = h - np.maximum(o, c); lwick = np.minimum(o, c) - l
    rng = (h - l)
    rows = []; last = -1
    for i in range(max(L, 50), len(d) - 1):
        if i <= last or not np.isfinite(a[i]) or a[i] <= 0:
            continue
        top = (h[i] >= hiL[i]) and (uwick[i] >= wick_k * a[i]) and (c[i] < h[i] - 0.25 * rng[i])
        bot = (l[i] <= loL[i]) and (lwick[i] >= wick_k * a[i]) and (c[i] > l[i] + 0.25 * rng[i])
        if not (top or bot):
            continue
        isS = top
        e = o[i + 1]
        stop = (h[i] + buf * a[i]) if isS else (l[i] - buf * a[i])
        risk = (stop - e) if isS else (e - stop)
        if risk <= 0:
            continue
        tgt = (e - rr * risk) if isS else (e + rr * risk)
        R = None; end = min(i + 1 + fwd, len(d))
        for j in range(i + 1, end):
            if (h[j] >= stop) if isS else (l[j] <= stop):
                R = -1; break
            if (l[j] <= tgt) if isS else (h[j] >= tgt):
                R = rr; break
        if R is None:
            R = ((e - c[end - 1]) if isS else (c[end - 1] - e)) / risk
            j = end - 1
        R -= cost * e / risk
        wlen = (uwick[i] if isS else lwick[i]) / a[i]
        stretch = ((c[i] - sma[i]) if isS else (sma[i] - c[i])) / a[i]   # how stretched into the move
        rows.append((d.index[i], R, volpr[i], wlen, stretch)); last = j
    return pd.DataFrame(rows, columns=["time", "R", "volpr", "wlen", "stretch"])


def null_pctile(t, keep_n, real_cdd, iters=3000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.array([cagr_dd(t.iloc[np.sort(rng.choice(len(t), keep_n, replace=False))][["time", "R"]])[2]
                    for _ in range(iters)])
    return (arr < real_cdd).mean() * 100


def main():
    d = resample(load_mt5_csv("data/vantage_usdjpy_h1.csv"), "4h")
    t = wick_trades(d)
    cg, dd, cdd0, _ = cagr_dd(t[["time", "R"]])
    print(f"== wick-fade base (USDJPY 4h, cost0.05%): n={len(t)} meanR={t.R.mean():+.2f} CAGR/DD={cdd0:.2f} ==")
    print("   selection test: keep TOP quantile by each signal; beat random-drop null (>=90%ile)?\n")
    print(f"  {'selector':<10}{'keep':>6}{'n':>5}{'meanR':>7}{'CAGR/DD':>9}{'vs random-drop':>16}")
    for sel in ("volpr", "wlen", "stretch"):
        for q in (0.50, 0.33, 0.25):
            thr = t[sel].quantile(1 - q)
            kept = t[t[sel] >= thr]
            if len(kept) < 12:
                continue
            c, dd, cdd, _ = cagr_dd(kept[["time", "R"]])
            pct = null_pctile(t, len(kept), cdd)
            flag = "PASS" if (pct >= 90 and kept.R.mean() > 0.10) else ("weak" if pct >= 75 else "n-trim")
            print(f"  {sel:<10}top{int(q*100):>3}%{len(kept):>5}{kept.R.mean():>+7.2f}{cdd:>9.2f}{pct:>13.0f}%ile {flag}")
    print("\n  also: 2-way intersection wick-length AND stretch (the two price-action signals) top33% each:")
    k = t[(t.wlen >= t.wlen.quantile(0.5)) & (t.stretch >= t.stretch.quantile(0.5))]
    if len(k) >= 12:
        c, dd, cdd, _ = cagr_dd(k[["time", "R"]])
        print(f"     wlen&stretch (top50%x50%): n={len(k)} meanR={k.R.mean():+.2f} CAGR/DD={cdd:.2f} "
              f"vs random {null_pctile(t, len(k), cdd):.0f}%ile")
    print("\n  verdict: a selector earns its place only if CAGR/DD beats random-drop >=90%ile AND meanR>~0.1 at cost.")


if __name__ == "__main__":
    main()
