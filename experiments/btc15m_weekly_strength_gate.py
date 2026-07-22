"""Can a WEEKLY-scale trend-STRENGTH condition fix btc15m_L's 2022 bleed?  (user hypothesis)

Why this is a legitimately open cell, and what is off the table:
  * ADX / ATR / vol-regime gates are DEAD (killed by the B->A discipline; the ledger says do not
    retest). Not used here.
  * ER (Kaufman efficiency ratio) is the ONE strength component that ever worked on BTC -- it is
    KAMA's own input. On btc15m_L it was tested at the 72h scale only (slope across quartiles but
    even Q1 positive -> soft-filtering predicted to fail the book).
  * WEEKLY scale is untested on this leg, and is exactly the granularity where the pullback leg's
    30-week cycle gate is the survivor. -> open cell.

WHAT TODAY'S DECOMPOSITION SAYS ABOUT THE TARGET (btc15m_gate_split.py):
  the 4h-gate leg's 2022 bleed (-14R) is NOT in the "4h up / daily down" bear-rally trades (-4R);
  it is in the trades where BOTH gates already agree (-10R). So a DIRECTIONAL weekly gate cannot
  fix it. Only a STRENGTH/WEAKNESS condition can. That is precisely the user's hypothesis.

CONDITIONERS (all on the PRIOR COMPLETED weekly/daily bar; no lookahead)
  Wk-ER14 / Wk-ER8      weekly efficiency ratio (|net move| / sum|moves|) over 14 / 8 weeks
  Wk-KAMA-rising        weekly KAMA(14) rising
  Wk-slope10            10-week SMA slope, ATR-normalised
  Wk-30MA-ratio         close / 30-week MA  (the btc_pull cycle gate's variable)
  Wk-HL-structure       weekly higher-high AND higher-low over the last 3 weekly swings
  D-ER14 / D-ER30       daily efficiency ratio (contrast: is weekly really the right scale?)

TEST (frozen before running)
  Each conditioner is turned into a gate by thresholding at its IS median / IS terciles, then the
  ENTRY SET IS REBUILT with the gate (re-arm coupling matters -- filtering the trade list after the
  fact is not the same machine). Reported per arm: n, PF, totR/yr, maxDD, totR/DD, CAGR/DD, per-year.

  PASS requires ALL of:
    1. OOS totR/DD > base OOS totR/DD                     (not just IS)
    2. random-drop null >= 90th %ile on CAGR/DD           (drop the same COUNT of armed entries at
       random, 300 draws -- this is what kills "any filter that removes a bad year looks good")
    3. the improvement is NOT confined to 2022            (>= 2 other years must not get worse)
    4. IS and OOS agree in sign
Run: .venv/bin/python experiments/btc15m_weekly_strength_gate.py
"""
import sys, os, io, contextlib, warnings
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src.data_loader import load_mt5_csv
import pine_replica_btc15m as P
from pine_replica_btc15m import walk, stats
from btc15m_gate_ab import build_entries, kama_rising

ROOT = "/home/angelbell/dev/auto-trade"
START = "2018-10-01"
NDRAW = 300
RNG = np.random.default_rng(20260712)


def er(s, n):
    """Kaufman efficiency ratio: |net move over n| / sum of |bar moves| over n.  0=chop, 1=trend."""
    mom = (s - s.shift(n)).abs()
    vol = s.diff().abs().rolling(n).sum()
    return (mom / vol).replace([np.inf, -np.inf], np.nan)


def conditioners(df):
    """Weekly/daily context, evaluated on the PRIOR COMPLETED bar, ffilled onto the 15m grid."""
    wk = df["close"].resample("W").last().dropna()
    dl = df["close"].resample("1D").last().dropna()
    out = {}

    def put(name, s):
        out[name] = s.shift(1).reindex(df.index, method="ffill")

    put("Wk-ER14", er(wk, 14))
    put("Wk-ER8", er(wk, 8))
    from breakout_wave import kama_adaptive
    kw = kama_adaptive(wk, 14)
    put("Wk-KAMA-rising", (kw > kw.shift(1)).astype(float))
    sma10 = wk.rolling(10).mean()
    put("Wk-slope10", (sma10 - sma10.shift(1)) / wk.rolling(10).std())
    put("Wk-30MA-ratio", wk / wk.rolling(30).mean())
    hh = wk.rolling(3).max() > wk.rolling(3).max().shift(3)
    hl = wk.rolling(3).min() > wk.rolling(3).min().shift(3)
    put("Wk-HL-structure", (hh & hl).astype(float))
    put("D-ER14", er(dl, 14))
    put("D-ER30", er(dl, 30))
    return out


def run(df, mask, ratchet, span, noflow):
    E = build_entries(df, mask)
    tr = walk(df, E, False, ratchet, noflow)
    return E, tr, stats(tr, span)


def cagr_dd(tr, span, risk=0.01):
    R = np.array([x[1] for x in tr])
    if len(R) == 0: return 0.0
    eq = np.cumprod(1 + risk * R); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max()
    cagr = (eq[-1] ** (1 / span) - 1)
    return cagr / dd if dd > 0 else np.nan


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    span = (df.index[-1] - df.index[0]).days / 365.25
    noflow = pd.Series(np.nan, index=df.index)
    gD, g4 = kama_rising(df, "1D"), kama_rising(df, "4h")
    C = conditioners(df)
    half = df.index[len(df) // 2]
    yrs = list(range(2018, 2027))

    for arm_name, base_gate, ratchet in (("A: daily gate + ratchet", gD, True),
                                         ("B: 4h gate, no ratchet", g4, False)):
        E0, tr0, s0 = run(df, base_gate, ratchet, span, noflow)
        by0 = {y: sum(x[1] for x in tr0 if x[0].year == y) for y in yrs}
        print(f"\n{'='*100}\n{arm_name}   BASE: n={s0['n']} PF {s0['pf']:.2f} totR/yr {s0['totyr']:+.1f} "
              f"maxDD {s0['ddp']:.1f}% totR/DD {s0['retdd']:.2f} CAGR/DD {cagr_dd(tr0, span):.2f} "
              f"| 2022 {by0[2022]:+.0f}R")
        print(f"{'conditioner':<20}{'keep%':>7}{'n':>5}{'PF':>6}{'totR/yr':>9}{'maxDD':>7}"
              f"{'totR/DD':>9}{'CAGR/DD':>9}{'2022':>7}{'IS/OOS totR/DD':>16}{'rnd-drop %ile':>15}")

        for cname, cs in C.items():
            v = cs.values.astype(float)
            fin = np.isfinite(v)
            # threshold = IS median of the conditioner AT THE BASE ENTRY BARS (no lookahead: IS only)
            eb = np.array([e[0] for e in E0])
            is_eb = eb[df.index[eb] < half]
            vals = v[is_eb]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 30: continue
            thr = np.median(vals)
            keep = fin & (v >= thr)                       # "strong enough" -> trade
            E1, tr1, s1 = run(df, base_gate & keep, ratchet, span, noflow)
            if s1["n"] < 40: continue
            by1 = {y: sum(x[1] for x in tr1 if x[0].year == y) for y in yrs}
            cd = cagr_dd(tr1, span)

            # IS/OOS totR/DD
            def sub(tr, lo, hi):
                R = np.array([x[1] for x in tr if lo <= x[0] < hi])
                if len(R) < 5: return np.nan
                cum = np.cumsum(R); dd = (np.maximum.accumulate(cum) - cum).max()
                return R.sum() / dd if dd > 0 else np.nan
            t0, t1 = df.index[0], df.index[-1]
            is_v, oos_v = sub(tr1, t0, half), sub(tr1, half, t1)

            # random-drop null: remove the same COUNT of armed entries at random
            ndrop = len(E0) - len(E1)
            null = []
            if ndrop > 0:
                for _ in range(NDRAW):
                    k = RNG.choice(len(E0), len(E0) - ndrop, replace=False)
                    Er = [E0[i] for i in sorted(k)]
                    null.append(cagr_dd(walk(df, Er, False, ratchet, noflow), span))
                null = np.array(null)
                pct = 100 * (cd > null).mean()
            else:
                pct = np.nan
            print(f"{cname:<20}{100*s1['n']/s0['n']:>6.0f}%{s1['n']:>5}{s1['pf']:>6.2f}"
                  f"{s1['totyr']:>9.1f}{s1['ddp']:>6.1f}%{s1['retdd']:>9.2f}{cd:>9.2f}"
                  f"{by1[2022]:>+7.0f}"
                  f"{f'{is_v:.1f}/{oos_v:.1f}':>16}{pct:>14.0f}%")

        print(f"  (base per-year: " + "  ".join(f"{y}:{by0[y]:+.0f}" for y in yrs if by0[y] != 0) + ")")


if __name__ == "__main__":
    main()
