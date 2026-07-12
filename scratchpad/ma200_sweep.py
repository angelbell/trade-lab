import os, sys
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import numpy as np, pandas as pd
from research.ma200_bounce import resample, find_signals, simulate, beta_null, stats
from src.data_loader import load_mt5_csv

FWD, COST = 200, 0.001

def probe(name, csv, tf, side):
    raw = load_mt5_csv(csv)
    d = resample(raw, tf)
    print(f"\n=== {name} {tf} {side}  (plateau check: slopeK x tol x RR) ===")
    print(f"{'slopeK':>7}{'tol':>6}{'RR':>5}{'n':>6}{'win%':>6}{'meanR':>8}{'IS':>7}{'OOS':>7}{'grn':>6}{'beta%':>7}")
    for slopeK in (10, 20, 30):
        for tol in (0.10, 0.25, 0.50):
            for rr in (1.5, 2.0, 3.0):
                sigs, ema, a = find_signals(d, side, slopeK=slopeK, tol=tol)
                t = simulate(d, sigs, side, rr, FWD, COST)
                st = stats(t)
                if st is None or st["n"] < 8:
                    continue
                bp = beta_null(d, side, sigs, rr, FWD, COST, ema, a, n_iter=300)
                star = " <<" if st["meanR"] > 0.15 and (bp or 0) >= 90 and st["OOS"] > 0 else ""
                print(f"{slopeK:>7}{tol:>6.2f}{rr:>5.1f}{st['n']:>6}{st['win']:>6.0f}"
                      f"{st['meanR']:>8.2f}{st['IS']:>7.2f}{st['OOS']:>7.2f}"
                      f"{st['green']:>3}/{st['nyr']:<2}{(bp if bp==bp else 0):>7.0f}{star}")

probe("GOLD", "data/vantage_xauusd_h1.csv", "8h", "long")
probe("BTC", "data/vantage_btcusd_h1.csv", "1d", "long")

# per-year detail for the gold 8h default config
print("\n=== GOLD 8h long per-year (slopeK20 tol0.25 RR2) ===")
d = resample(load_mt5_csv("data/vantage_xauusd_h1.csv"), "8h")
sigs, ema, a = find_signals(d, "long", slopeK=20, tol=0.25)
t = simulate(d, sigs, "long", 2.0, FWD, COST); t["y"] = t["time"].dt.year
print(" ".join(f"{y}:{g.R.sum():+.0f}(n{len(g)})" for y, g in t.groupby("y")))
