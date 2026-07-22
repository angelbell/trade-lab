"""Finalists head-to-head for btc15m_L's regime gate.

Round 1 (btc15m_gate_ab / _stress) said: the ratchet's sign FLIPS with the gate (helps on daily,
hurts on 4h), and on the 4h gate the whole ratchet TP2xfloor grid is a noise field. So:
  1. is the ratchet a PLATEAU on the daily gate, or is it noise there too?  (full TP2 x floor grid)
  2. per-year + IS/OOS for the three finalists:
       A = daily gate, ratchet ON   (current Pine default)
       B = 4h gate,    ratchet OFF
       C = daily AND 4h gate, ratchet OFF
  3. paired monthly bootstrap on the user's axis (equal-DD terminal wealth): P(B>A), P(C>A), P(C>B)
Run: .venv/bin/python experiments/btc15m_gate_final.py
"""
import sys, os
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, io, contextlib
from src.data_loader import load_mt5_csv
import pine_replica_btc15m as P
from pine_replica_btc15m import walk, stats, ROOT, START
from btc15m_gate_ab import build_entries, kama_rising, equal_dd_bet

RNG = np.random.default_rng(20260712)
REF_DD = 19.0          # the daily/no-ratchet reference maxDD; all bets scaled to this


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    span = (df.index[-1] - df.index[0]).days / 365.25
    noflow = pd.Series(np.nan, index=df.index)
    gD, g4 = kama_rising(df, "1D"), kama_rising(df, "4h")
    ED, E4, EB = (build_entries(df, gD), build_entries(df, g4), build_entries(df, gD & g4))

    # ---- 1. is the ratchet a plateau on the DAILY gate? -----------------------
    orig = P.RATCH
    print(f"1. ratchet TP2 x floor grid, totR/DD  (daily gate; ratchet OFF = "
          f"{stats(walk(df, ED, False, False, noflow), span)['retdd']:.2f})")
    tp2s, floors = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0], [0.0, 1.0, 1.5, 2.0, 2.5]
    print(f"{'TP2\\floor':<10}" + "".join(f"{f:>8.1f}" for f in floors))
    grid = {}
    for tp2 in tp2s:
        row = []
        for fl in floors:
            P.RATCH = (tp2, fl)
            v = stats(walk(df, ED, False, True, noflow), span)["retdd"]
            grid[(tp2, fl)] = v; row.append(v)
        print(f"{tp2:<10.1f}" + "".join(f"{v:>8.2f}" for v in row))
    g = np.array(list(grid.values()))
    print(f"   grid: min {g.min():.2f} / median {np.median(g):.2f} / max {g.max():.2f}  "
          f"| cells beating ratchet-off: {(g > 6.62).sum()}/{len(g)}")
    P.RATCH = orig

    # ---- 2. the three finalists ---------------------------------------------
    P.RATCH = (3.5, 2.0)
    fin = {"A daily + ratchet": walk(df, ED, False, True, noflow),
           "B 4h, no ratchet":  walk(df, E4, False, False, noflow),
           "C daily AND 4h, no ratchet": walk(df, EB, False, False, noflow)}
    yrs = sorted({t.year for tr in fin.values() for t, *_ in tr})
    print("\n2. finalists")
    print(f"{'cell':<28}{'n':>5}{'N/yr':>6}{'PF':>6}{'totR/yr':>8}{'maxDD%':>8}{'tot/DD':>7}"
          f"{'grn':>6}{'IS/OOS':>12}{'eqDD f':>8}{'wealth':>8}")
    for name, tr in fin.items():
        s = stats(tr, span)
        R = np.array([x[1] for x in tr])
        f, cagr, mult = equal_dd_bet(R, span, REF_DD)
        print(f"{name:<28}{s['n']:>5}{s['npy']:>6.1f}{s['pf']:>6.2f}{s['totyr']:>8.1f}"
              f"{s['ddp']:>7.1f}%{s['retdd']:>7.2f}{s['grn']:>5.0f}%"
              f"{f'{s[chr(73)+chr(83)]:+.0f}/{s[chr(79)+chr(79)+chr(83)]:+.0f}':>12}"
              f"{100*f:>7.2f}%{mult:>7.2f}x")
    print(f"\n{'per-year totR':<28}" + "".join(f"{y:>7}" for y in yrs))
    for name, tr in fin.items():
        by = {y: sum(x[1] for x in tr if x[0].year == y) for y in yrs}
        print(f"{name:<28}" + "".join(f"{by[y]:>+7.0f}" for y in yrs))

    # ---- 3. paired monthly bootstrap on terminal wealth at equal DD ----------
    def monthly(tr):
        s = pd.Series([x[1] for x in tr], index=pd.DatetimeIndex([x[0] for x in tr]))
        return s.resample("ME").sum()
    M = {k: monthly(v) for k, v in fin.items()}
    idx = M["A daily + ratchet"].index
    for v in M.values(): idx = idx.union(v.index)
    W = {k: v.reindex(idx).fillna(0).values for k, v in M.items()}
    m = len(idx)
    print(f"\n3. paired monthly bootstrap ({m} months, 3000 draws), equal-DD terminal wealth")
    draws = {k: [] for k in fin}
    for _ in range(3000):
        k_idx = RNG.integers(0, m, m)
        for k in fin:
            draws[k].append(equal_dd_bet(W[k][k_idx], m / 12, REF_DD)[2])
    for k in fin:
        d = np.array(draws[k])
        print(f"   {k:<28} wealth median {np.median(d):>5.2f}x  "
              f"[10th {np.percentile(d,10):>4.2f}x, 90th {np.percentile(d,90):>5.2f}x]")
    A, B, C = (np.array(draws[k]) for k in fin)
    print(f"   P(B > A) = {100*(B > A).mean():.0f}%   P(C > A) = {100*(C > A).mean():.0f}%   "
          f"P(C > B) = {100*(C > B).mean():.0f}%")
    P.RATCH = orig


if __name__ == "__main__":
    main()
