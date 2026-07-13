"""Stress the leading result from btc15m_weekly_strength_gate.py:
   arm B (4h gate, no ratchet) + weekly ER(14) >= IS-median  ->  CAGR/DD 1.40 -> 1.60, 2022 fixed.

Frozen criterion 3 from that spec is still unchecked: "the improvement must NOT be confined to
2022". Plus the standard falsifiers this lab requires before believing a gate:
  1. FULL PER-YEAR (is 2022 the only year that moves? does it cost the trend years?)
  2. THRESHOLD PLATEAU: sweep the ER cut across keep-rates 30/40/50/60/70%. A real gate is a
     plateau; an overfit one is a lone spike. (The headline used the IS median = the 50% cut.)
  3. ER LENGTH PLATEAU: 8 / 10 / 14 / 20 / 26 weeks. If only 14 works, it is a spike.
  4. RATCHET INTERACTION: the ratchet was harmful on the 4h gate. Does the weekly gate change that?
  5. THE USER'S AXIS: terminal wealth at equal DD, vs the two incumbents (A = daily+ratchet,
     B = 4h plain).
  6. GATE ON% by year (a gate that is ON 95% of the time is not a gate; one ON 10% is a different
     strategy).
Run: .venv/bin/python scratchpad/btc15m_wk_er_stress.py
"""
import sys, os, io, contextlib, warnings
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src.data_loader import load_mt5_csv
import pine_replica_btc15m as P
from pine_replica_btc15m import walk, stats
from btc15m_gate_ab import build_entries, kama_rising, equal_dd_bet
from btc15m_weekly_strength_gate import er, cagr_dd

ROOT, START, REF_DD = "/home/angelbell/dev/auto-trade", "2018-10-01", 19.0
YRS = list(range(2018, 2027))


def wk_er(df, n):
    wk = df["close"].resample("W").last().dropna()
    return er(wk, n).shift(1).reindex(df.index, method="ffill")


def peryear(tr):
    return {y: sum(x[1] for x in tr if x[0].year == y) for y in YRS}


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    span = (df.index[-1] - df.index[0]).days / 365.25
    noflow = pd.Series(np.nan, index=df.index)
    gD, g4 = kama_rising(df, "1D"), kama_rising(df, "4h")
    half = df.index[len(df) // 2]

    E4 = build_entries(df, g4)
    trB = walk(df, E4, False, False, noflow)
    trA = walk(df, build_entries(df, gD), False, True, noflow)
    eb = np.array([e[0] for e in E4])
    is_eb = eb[df.index[eb] < half]

    e14 = wk_er(df, 14).values.astype(float)
    thr = np.nanmedian(e14[is_eb])
    gate = np.isfinite(e14) & (e14 >= thr)
    trG = walk(df, build_entries(df, g4 & gate), False, False, noflow)
    print(f"weekly ER(14) IS-median threshold = {thr:.3f}   "
          f"(gate ON on {100*gate.mean():.0f}% of 15m bars)")

    # ---- 1. full per-year ----------------------------------------------------
    print("\n1. per-year totR   (criterion 3: the gain must not be 2022-only)")
    print(f"{'':<26}" + "".join(f"{y:>7}" for y in YRS))
    for nm, tr in (("A daily+ratchet", trA), ("B 4h plain", trB), ("B + weekly-ER14", trG)):
        b = peryear(tr)
        print(f"{nm:<26}" + "".join(f"{b[y]:>+7.0f}" for y in YRS))
    bB, bG = peryear(trB), peryear(trG)
    d = {y: bG[y] - bB[y] for y in YRS}
    print(f"{'delta (gated - plain)':<26}" + "".join(f"{d[y]:>+7.0f}" for y in YRS))
    print(f"   years improved: {[y for y in YRS if d[y] > 2]}   "
          f"years hurt (< -2R): {[y for y in YRS if d[y] < -2]}")

    # ---- 2/3. threshold and length plateaus ---------------------------------
    print(f"\n2+3. PLATEAU test -- CAGR/DD over ER-length x keep-rate  (base B = "
          f"{cagr_dd(trB, span):.2f}; a real gate is a plateau, not a spike)")
    lens, keeps = [8, 10, 14, 20, 26], [0.7, 0.6, 0.5, 0.4, 0.3]
    print(f"{'ER weeks':<10}" + "".join(f"{int(100*k)}% keep".rjust(10) for k in keeps))
    for L in lens:
        e = wk_er(df, L).values.astype(float)
        row = []
        for k in keeps:
            q = np.nanquantile(e[is_eb], 1 - k)
            g = np.isfinite(e) & (e >= q)
            row.append(cagr_dd(walk(df, build_entries(df, g4 & g), False, False, noflow), span))
        print(f"{L:<10}" + "".join(f"{v:>10.2f}" for v in row))

    # ---- 4. ratchet interaction ---------------------------------------------
    print("\n4. ratchet interaction on the gated leg (it was HARMFUL on the plain 4h gate)")
    for rn, ur in (("off", False), ("ON", True)):
        tr = walk(df, build_entries(df, g4 & gate), False, ur, noflow)
        s = stats(tr, span)
        print(f"   ratchet {rn:<4} n={s['n']:>4} PF {s['pf']:.2f} totR/yr {s['totyr']:>+5.1f} "
              f"maxDD {s['ddp']:>4.1f}% totR/DD {s['retdd']:>5.2f} CAGR/DD {cagr_dd(tr, span):.2f}")

    # ---- 5. the user's axis: terminal wealth at equal DD ---------------------
    print(f"\n5. equal-DD terminal wealth (bet scaled so every arm runs maxDD {REF_DD}%), {span:.1f}yr")
    arms = (("A daily+ratchet", trA), ("B 4h plain", trB), ("B + weekly-ER14", trG),
            ("B + weekly-ER14 + ratchet",
             walk(df, build_entries(df, g4 & gate), False, True, noflow)))
    for nm, tr in arms:
        R = np.array([x[1] for x in tr])
        f, cagr, mult = equal_dd_bet(R, span, REF_DD)
        s = stats(tr, span)
        print(f"   {nm:<28} f={100*f:>5.2f}%  CAGR={cagr:>6.1f}%  wealth={mult:>6.2f}x  "
              f"(N/yr {s['npy']:.0f}, PF {s['pf']:.2f})")

    # ---- 6. gate ON% by year -------------------------------------------------
    print("\n6. gate ON% by year (a gate ON ~always is not a gate)")
    g = pd.Series(gate, index=df.index)
    print("   " + "  ".join(f"{y}:{100*g[g.index.year == y].mean():.0f}%" for y in YRS))


if __name__ == "__main__":
    main()
