"""Stress the gate A/B result (btc15m_gate_ab.py): is "4h gate, ratchet OFF" real, or is the
ranking noise / a lone spike?

Tests
  1. per-year totR for all 6 cells (does the 4h gate really fix the bleed years?)
  2. ratchet parameter sweep ON THE 4H GATE (TP2 x floor): is the sign flip structural or a
     bad parameter pair?  A real effect should be a plateau, not one cell.
  3. WHY does the ratchet hurt on 4h?  It exits earlier -> the machine re-arms sooner -> a
     DIFFERENT trade population.  Split the ratchet's effect into (a) the direct effect on the
     trades both variants take and (b) the trades only the ratchet variant takes.
  4. paired monthly bootstrap: P(4h-off beats daily-ON) on totR/DD and on equal-DD wealth.
Run: .venv/bin/python scratchpad/btc15m_gate_stress.py
"""
import sys, os
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, io, contextlib
from src.data_loader import load_mt5_csv
import pine_replica_btc15m as P
from pine_replica_btc15m import walk, stats, ROOT, START
from btc15m_gate_ab import build_entries, kama_rising, equal_dd_bet

RNG = np.random.default_rng(20260712)


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    span = (df.index[-1] - df.index[0]).days / 365.25
    noflow = pd.Series(np.nan, index=df.index)
    gD, g4 = kama_rising(df, "1D"), kama_rising(df, "4h")
    ED, E4 = build_entries(df, gD), build_entries(df, g4)

    cells = {("daily", "off"): walk(df, ED, False, False, noflow),
             ("daily", "ON"):  walk(df, ED, False, True,  noflow),
             ("4h",    "off"): walk(df, E4, False, False, noflow),
             ("4h",    "ON"):  walk(df, E4, False, True,  noflow)}

    # ---- 1. per-year ---------------------------------------------------------
    yrs = sorted({t.year for tr in cells.values() for t, *_ in tr})
    print("1. per-year totR")
    print(f"{'cell':<16}" + "".join(f"{y:>7}" for y in yrs) + f"{'maxDD(R)':>10}{'grn':>6}")
    for (g, r), tr in cells.items():
        by = {y: sum(x[1] for x in tr if x[0].year == y) for y in yrs}
        s = stats(tr, span)
        print(f"{g+' ratchet-'+r:<16}" + "".join(f"{by[y]:>+7.0f}" for y in yrs)
              + f"{s['ddR']:>10.1f}{s['grn']:>5.0f}%")

    # ---- 2. ratchet sweep on the 4h gate -------------------------------------
    print("\n2. ratchet sweep on the 4h gate (totR/DD; base = ratchet off = "
          f"{stats(cells[('4h','off')], span)['retdd']:.2f})")
    tp2s, floors = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0], [0.0, 1.0, 1.5, 2.0, 2.5]
    print(f"{'TP2\\floor':<10}" + "".join(f"{f:>8.1f}" for f in floors))
    orig = P.RATCH
    for tp2 in tp2s:
        row = []
        for fl in floors:
            P.RATCH = (tp2, fl)
            row.append(stats(walk(df, E4, False, True, noflow), span)["retdd"])
        print(f"{tp2:<10.1f}" + "".join(f"{v:>8.2f}" for v in row))
    row = []
    for fl in floors:
        P.RATCH = (3.5, fl)
        row.append(stats(walk(df, ED, False, True, noflow), span)["retdd"])
    print(f"{'daily@3.5':<10}" + "".join(f"{v:>8.2f}" for v in row)
          + f"   <- contrast: same floor sweep on the daily gate (off = "
          f"{stats(cells[('daily','off')], span)['retdd']:.2f})")
    P.RATCH = orig

    # ---- 3. decompose the ratchet's effect on the 4h gate ---------------------
    P.RATCH = (3.5, 2.0)
    off, on = cells[("4h", "off")], cells[("4h", "ON")]
    toff = {x[0]: x[1] for x in off}
    ton = {x[0]: x[1] for x in on}
    shared = sorted(set(toff) & set(ton))
    only_on = sorted(set(ton) - set(toff))
    only_off = sorted(set(toff) - set(ton))
    d_shared = sum(ton[t] - toff[t] for t in shared)
    print("\n3. where does the ratchet's damage on 4h come from? (totR decomposition)")
    print(f"   shared trades      n={len(shared):>4}  ratchet effect on them  {d_shared:>+7.1f}R")
    print(f"   only in ratchet-ON n={len(only_on):>4}  their totR             "
          f"{sum(ton[t] for t in only_on):>+7.1f}R  (extra trades the earlier exits re-arm into)")
    print(f"   only in ratchet-off n={len(only_off):>3}  their totR             "
          f"{sum(toff[t] for t in only_off):>+7.1f}R  (trades lost to the re-arm shift)")
    print(f"   net                                                  "
          f"{sum(ton.values()) - sum(toff.values()):>+7.1f}R")

    # ---- 4. paired monthly bootstrap: 4h-off vs daily-ON ----------------------
    print("\n4. paired monthly bootstrap (4h ratchet-off  vs  daily ratchet-ON), 4000 draws")
    def monthly(tr):
        s = pd.Series([x[1] for x in tr], index=pd.DatetimeIndex([x[0] for x in tr]))
        return s.resample("ME").sum()
    a, b = monthly(cells[("4h", "off")]), monthly(cells[("daily", "ON")])
    idx = a.index.union(b.index)
    A, B = a.reindex(idx).fillna(0).values, b.reindex(idx).fillna(0).values
    m = len(idx)
    wins_dd = wins_w = 0
    da, db = [], []
    for _ in range(4000):
        k = RNG.integers(0, m, m)                       # same months for both = paired
        for R, store in ((A[k], da), (B[k], db)):
            cum = np.cumsum(R); dd = (np.maximum.accumulate(cum) - cum).max()
            store.append(R.sum() / dd if dd > 0 else np.nan)
        if da[-1] > db[-1]: wins_dd += 1
        fa = equal_dd_bet(A[k], m / 12, 19.0)[2]
        fb = equal_dd_bet(B[k], m / 12, 19.0)[2]
        if fa > fb: wins_w += 1
    da, db = np.array(da), np.array(db)
    print(f"   totR/DD  4h-off median {np.nanmedian(da):.2f} [5th {np.nanpercentile(da,5):.2f}] "
          f"vs daily-ON {np.nanmedian(db):.2f} [5th {np.nanpercentile(db,5):.2f}]")
    print(f"   P(4h-off > daily-ON): totR/DD {100*wins_dd/4000:.0f}%   "
          f"equal-DD final wealth {100*wins_w/4000:.0f}%")


if __name__ == "__main__":
    main()
