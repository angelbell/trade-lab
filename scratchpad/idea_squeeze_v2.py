"""Salvage the squeeze breakout — attack the 病巣 from v1:
  (1) couple the trigger to the squeeze: break the CONTRACTION BOX itself
      (box = prior-K high/low) — not a decoupled 20-Donchian.
  (2) don't transplant RR3 — SWEEP RR and compare to the PLAIN breakout on CAGR/DD
      (ret/DD), HTF-focused (1d hinted at 89%ile in v1).
Honest bar: proper-squeeze must BEAT plain breakout's ret/DD at matched RR/TF, with a
plateau. If not, it's dead. (verification order: this is the RR step done from data.)"""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src.data_loader import load_mt5_csv
from research.edge_harness import evaluate, check_causal, LADDERS, AGG


def plain_bo(df):
    c = df["close"].values; sma = df["close"].rolling(100).mean().values
    dch = pd.Series(df["high"].values).rolling(20).max().shift(1).values
    dcl = pd.Series(df["low"].values).rolling(20).min().shift(1).values
    sig = np.zeros(len(c)); sig[(c > dch) & (c > sma)] = 1; sig[(c < dcl) & (c < sma)] = -1
    sig[:120] = 0; return sig


def make_squeeze(K=10, thr=2.0):
    """break the CONTRACTION BOX: box = prior-K high/low; squeeze = box spans < thr*ATR.
    long when close breaks box-high out of a tight box (trend-aligned). Causal."""
    def sig_fn(df):
        c = df["close"].values; h = df["high"].values; l = df["low"].values
        sma = df["close"].rolling(100).mean().values
        atr = (df["high"] - df["low"]).rolling(14).mean().values  # simple range-ATR
        bh = pd.Series(h).rolling(K).max().shift(1).values        # box high (prior K, confirmed)
        bl = pd.Series(l).rolling(K).min().shift(1).values        # box low
        tight = (bh - bl) < thr * atr                              # contraction
        sig = np.zeros(len(c))
        sig[tight & (c > bh) & (c > sma)] = 1
        sig[tight & (c < bl) & (c < sma)] = -1
        sig[:120] = 0; return sig
    sig_fn.__name__ = f"squeeze_K{K}_thr{thr}"
    return sig_fn


def rd(name, sig, inst, tf, rr, skip):
    r = evaluate(inst, sig, rr=rr, only=[tf], stop_slip=0.5, skip_hours=skip,
                 beta_trials=0, quiet=True, _return=True)
    if tf in r:
        _, s = r[tf]; return s
    return None


check_causal(make_squeeze(), load_mt5_csv(LADDERS["GOLD"][0]).resample("240min").agg(AGG).dropna())
sq = make_squeeze(K=10, thr=2.0)
for inst, skip in [("GOLD", (12, 13, 14)), ("BTC", None)]:
    print(f"\n############ {inst}  — proper-squeeze vs plain, ret/DD (N) ############")
    print(f"  {'TF/RR':<10}{'plain ret/DD(N)':>22}{'squeeze ret/DD(N)':>24}")
    for tf in ("4h", "8h", "1d"):
        for rr in (2.0, 3.0, 4.0):
            p = rd("plain", plain_bo, inst, tf, rr, skip)
            q = rd("sq", sq, inst, tf, rr, skip)
            ps = f"{p['retdd']:+.2f}({p['N']})" if p else "-"
            qs = f"{q['retdd']:+.2f}({q['N']})" if q else "-"
            mark = "  <<" if (p and q and q['retdd'] > p['retdd'] and q['retdd'] > 0.3) else ""
            print(f"  {tf}/RR{rr:<5}{ps:>22}{qs:>24}{mark}")
