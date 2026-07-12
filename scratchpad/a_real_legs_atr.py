"""A: apply the ATR_regime>1.2 trend-presence gate to the REAL book legs.
Core head-to-head on the UNGATED btc 4h breakout: none / KAMA / ATRreg / both.
Plus gold_bo (add on top of daily-SMA gate) and btc_pull (does high-vol HURT the pullback?)."""
import os, sys
from types import SimpleNamespace
import numpy as np, pandas as pd, pandas_ta as ta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from breakout_wave import run as run_bo, resample
from research.regime_gate_lab import CFG, at
from research.portfolio_kama import get_legs, kama_gate_btc, cagr_dd, PB
from ema_pullback import run as run_pb
from research.overfit_audit import cdd_R
import research.edge_harness as EH

def atr_cond(df, thr=1.2):
    a = ta.atr(df["high"], df["low"], df["close"], 14)
    return (a / a.rolling(100).median()) >= thr
def gate_atr(t, df, thr=1.2):
    return t[at(atr_cond(df, thr).shift(1), t.time)]

def isoos(t):
    yr = t.time.dt.year.values; med = np.median(np.unique(yr))
    IS = t.R[yr < med].mean(); OOS = t.R[yr >= med].mean(); return IS, OOS
def green(t):
    yr = t.time.dt.year.values; uy = np.unique(yr)
    return sum(1 for y in uy if t.R[yr == y].sum() > 0), len(uy)
def row(nm, t):
    c, dd, cdd, ret = cagr_dd(t); IS, OOS = isoos(t); g, ny = green(t)
    print(f"  {nm:<20}n={len(t):>4}  meanR={t.R.mean():+.3f}  CAGR/DD={cdd:>+5.2f}  IS/OOS={IS:+.2f}/{OOS:+.2f}  green={g}/{ny}")

dbtc = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h")
dgold = resample(load_mt5_csv("data/vantage_xauusd_h1.csv"), "1h")
btc = run_bo(dbtc, SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0, "fwd": 300}))[["time", "R"]]
legs = get_legs()
gold, pb = legs["gold_bo"], legs["btc_pull"]

print("=== BTC 4h breakout — GATE HEAD-TO-HEAD (ungated base) ===")
row("ungated", btc)
row("KAMA_rising", kama_gate_btc(btc))
row("ATRreg>1.2", gate_atr(btc, dbtc))
row("KAMA & ATRreg", gate_atr(kama_gate_btc(btc), dbtc))

print("\n=== gold_bo (already daily-SMA gated) + ATRreg on top ===")
row("gold_bo", gold); row("gold_bo + ATRreg", gate_atr(gold, dgold))

print("\n=== btc_pull (cycle-gated) + ATRreg — does high-vol HURT the pullback? ===")
row("btc_pull", pb); row("btc_pull + ATRreg", gate_atr(pb, dbtc))

# overfit audit: btc breakout gate family
def stream(t): return list(zip(t.time.tolist(), t.R.tolist()))
configs = {"ungated": stream(btc), "KAMA": stream(kama_gate_btc(btc))}
for thr in (1.0, 1.1, 1.2, 1.3, 1.4): configs[f"ATRreg>{thr}"] = stream(gate_atr(btc, dbtc, thr))
configs["KAMA&ATR"] = stream(gate_atr(kama_gate_btc(btc), dbtc))
print("\n### AUDIT flagship=ATRreg>1.2 (real btc_bo base) ###")
EH.audit(configs, flagship="ATRreg>1.2")
print("### AUDIT flagship=KAMA (incumbent) ###")
EH.audit(configs, flagship="KAMA")
