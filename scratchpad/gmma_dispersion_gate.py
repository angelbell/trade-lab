"""gmma_dispersion_gate.py -- daily GMMA (Guppy 12-EMA) dispersion state as a deploy gate for the
two breakout legs, vs the standing champion daily-KAMA(14)-rising gate. Reuses the leg-construction
and gate-application pattern from research/regime_kama_legs.py / research/regime_statedet.py
(same CFG, same `at()` causal reindex, same `metrics()` CAGR/DD reporting) so numbers are directly
comparable to the ledger. Entries are FIXED (canon breakout configs); only the deploy gate changes.

GMMA: short EMA spans [3,5,8,10,12,15], long EMA spans [30,35,40,45,50,60], all on daily CONFIRMED
close. Three parameter-free-ish gates:
  G1 separation : min(short group) > max(long group)                       (bull-trend state)
  G2 orderliness: count of correctly-ordered adjacent pairs (11 total, sorted 3..60) >= T
  G3 width-exp  : long-group width/close today > width k days ago (long group stretching)
All raw signals computed on confirmed day t; a further shift(1) is applied ONLY at the point of
gating trades (`at(gate.shift(1), t.time)`), identical to the kama_gate() convention in the
precedent scripts -- no lookahead.

  .venv/bin/python scratchpad/gmma_dispersion_gate.py            # full history
  .venv/bin/python scratchpad/gmma_dispersion_gate.py --smoke    # 2020-2023 mechanics check
"""
import os, sys, argparse
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from research.regime_gate_lab import CFG, SPLIT, metrics, at
from research.regime_adaptive import kama

SHORT = [3, 5, 8, 10, 12, 15]
LONG = [30, 35, 40, 45, 50, 60]
KN = 14


def gmma_emas(dc):
    return {s: dc.ewm(span=s, adjust=False).mean() for s in SHORT + LONG}


def g1_separation(emas):
    short_min = pd.concat([emas[s] for s in SHORT], axis=1).min(axis=1)
    long_max = pd.concat([emas[s] for s in LONG], axis=1).max(axis=1)
    return short_min > long_max


def g2_orderliness(emas, T):
    spans = SHORT + LONG  # ascending span order
    ok = pd.concat({f"{a}-{b}": emas[a] > emas[b] for a, b in zip(spans[:-1], spans[1:])}, axis=1)
    return ok.sum(axis=1) >= T


def g3_width_expanding(emas, close, k):
    long_df = pd.concat([emas[s] for s in LONG], axis=1)
    width = (long_df.max(axis=1) - long_df.min(axis=1)) / close
    return width > width.shift(k)


def phi_coeff(a, b):
    """Matthews correlation coefficient between two aligned boolean series."""
    a, b = a.align(b, join="inner")
    a = a.fillna(False).astype(bool); b = b.fillna(False).astype(bool)
    tp = (a & b).sum(); tn = ((~a) & (~b)).sum()
    fp = (a & (~b)).sum(); fn = ((~a) & b).sum()
    num = tp * tn - fp * fn
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / den if den > 0 else np.nan


def show(name, t, ref=None):
    m = metrics(t)
    if m is None:
        print(f"    {name:<26} (too few)"); return None
    flag = ""
    if ref is not None and m["cdd"] > ref:
        flag = "  > KAMA"
    print(f"    {name:<26} n={m['n']:>4} totR={m['totR']:+5.0f} CAGR={m['cagr']:+5.1f}% DD={m['dd']:4.1f}% "
          f"CAGR/DD={m['cdd']:5.2f} | IS={m['isr']:+.2f} OOS={m['oos']:+.2f} | yr+ {m['py']}{flag}")
    return m["cdd"]


def run_instrument(name, csv, tf, rr, fwd, start=None, end=None):
    d = resample(load_mt5_csv(csv), tf)
    if start or end:
        d = d.loc[start:end]
    t = run(d, SimpleNamespace(**{**CFG, "csv": csv, "tf": tf, "rr": rr, "fwd": fwd}))
    t = t.sort_values("time").reset_index(drop=True)
    dc = d["close"].resample("1D").last().dropna()

    km = kama(dc, KN)
    kama_up = (km > km.shift(1))

    emas = gmma_emas(dc)
    g1 = g1_separation(emas)
    g2s = {T: g2_orderliness(emas, T) for T in (9, 10, 11)}
    g3s = {k: g3_width_expanding(emas, dc, k) for k in (3, 5, 10)}
    kama_and_g1 = kama_up & g1

    print(f"\n{'='*92}\n=== {name} ({os.path.basename(csv)} {tf}, RR{rr}, fwd{fwd}) "
          f"days={len(dc)} trades={len(t)} IS<{SPLIT} ===")

    print("  -- phi(GMMA gate, KAMA-rising) + ON% (full-history daily series, raw i.e. pre use-shift) --")
    print(f"    KAMA-rising                ON%={kama_up.mean()*100:5.1f}%")
    print(f"    G1 separation              ON%={g1.mean()*100:5.1f}%  phi={phi_coeff(g1, kama_up):+.3f}")
    for T, g in g2s.items():
        print(f"    G2 orderliness T={T:<2}        ON%={g.mean()*100:5.1f}%  phi={phi_coeff(g, kama_up):+.3f}")
    for k, g in g3s.items():
        print(f"    G3 width-exp k={k:<2}         ON%={g.mean()*100:5.1f}%  phi={phi_coeff(g, kama_up):+.3f}")
    print(f"    KAMA & G1                  ON%={kama_and_g1.mean()*100:5.1f}%  phi={phi_coeff(kama_and_g1, kama_up):+.3f}")

    print("  -- gate swap on leg (CAGR/DD; entries FIXED, comparable to regime_statedet.py metric) --")
    show("always-on", t)
    kama_cdd = show("KAMA-rising (baseline)", t[at(kama_up.shift(1), t.time)])
    show("G1 separation", t[at(g1.shift(1), t.time)], kama_cdd)
    for T, g in g2s.items():
        show(f"G2 orderliness T={T}", t[at(g.shift(1), t.time)], kama_cdd)
    for k, g in g3s.items():
        show(f"G3 width-exp k={k}", t[at(g.shift(1), t.time)], kama_cdd)
    show("KAMA & G1", t[at(kama_and_g1.shift(1), t.time)], kama_cdd)

    yr_on = (g1.groupby(g1.index.year).mean() * 100).round(0).astype(int)
    print("  -- per-year ON% of G1 separation --")
    print("    " + ", ".join(f"{y}:{p}%" for y, p in yr_on.items()))

    return kama_cdd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="restrict to 2020-2023 to sanity-check mechanics")
    a = ap.parse_args()
    legs = [
        ("GOLD breakout 1H", "data/vantage_xauusd_h1.csv", "1h", 3.0, 500),
        ("BTC breakout 4H", "data/vantage_btcusd_h1.csv", "4h", 2.0, 300),
    ]
    start, end = ("2020-01-01", "2023-12-31") if a.smoke else (None, None)
    print("GMMA (Guppy 12-EMA) dispersion state as a deploy gate -- bar to beat = KAMA(14)-rising")
    if a.smoke:
        print("[SMOKE TEST: window 2020-01-01..2023-12-31]")
    for name, csv, tf, rr, fwd in legs:
        run_instrument(name, csv, tf, rr, fwd, start, end)
    print(f"\n{'='*92}")
    print("PASS rule: GMMA gate beats KAMA-rising on BOTH legs, or KAMA&G1 improves on KAMA with agreeing")
    print("neighbors. Prediction to check: phi(G1,KAMA) high -> redundant.")


if __name__ == "__main__":
    main()
