"""Size UP when the HIGHER TIMEFRAME trend is strong -- the classic rule, tested on arm B.

Established today (btc15m_adaptive_rr_diag.py): entry-time "strength" does NOT change how FAR a
trade runs (the MFE distribution has the same shape; every group's optimal target is RR4.5) -- it
changes WHETHER the trade works at all. So strength belongs on SIZE, not on the target.
The leg already does a weak version of this (PDH: full size above the prior-day high, half inside).

*** LOOKAHEAD FIX (the diagnostic had it, this does not) ***
A 4H swing high is only KNOWN at the close of its confirm bar. Reindexing the 4H series onto 15m
bars with ffill hands it to the 15m bars INSIDE that 4H bar -- up to 4 hours early. Every HTF
series here is .shift(1) on its own timeframe before being mapped down.

Machine: arm B = 4h-KAMA gate, pullback limit frac 0.30, RR4.5, no ratchet, cost $15 RT.
Sizing arms (the trade SET is identical whenever every weight > 0 -> a clean apples-to-apples test):
  base    PDH soft:  w = 1.0 above the prior-day high, 0.5 inside
  A       HH4H soft: w = 1.0 above the last CONFIRMED 4H swing high, else weak_w
  B       both:      w = 1.0 if both, 0.5 if one, 0.25 if neither
  C       HH4H hard: w = 1.0 / 0.0 (a gate -- expected to lose; included so the claim is falsifiable)
Falsifiers: per-year · IS/OOS · plateau over weak_w · CIRCULAR BLOCK BOOTSTRAP vs base (the arbiter
that killed the weekly-ER gate, the ratchet-on-B and RR3 today) · redundancy vs PDH.
Run: .venv/bin/python experiments/btc15m_htf_size.py
"""
import sys, os, io, contextlib, warnings
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src.data_loader import load_mt5_csv
import pine_replica_btc15m as P, btc15m_gate_ab as G
from pine_replica_btc15m import stats
from btc15m_gate_ab import kama_rising, equal_dd_bet
from breakout_wave import swings_zigzag
from trend_leg_aging import atr as atr_fn

ROOT, START, REF_DD = "/home/angelbell/dev/auto-trade", "2018-10-01", 19.0
RR = 4.5
YRS = list(range(2018, 2027))


def hh4h_series(df):
    """Last CONFIRMED 4H swing high, shifted one 4H bar so it is only used AFTER it is known."""
    h4 = df.resample("4h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    a4 = atr_fn(h4["high"].values, h4["low"].values, h4["close"].values)
    sw = swings_zigzag(h4["high"].values, h4["low"].values, a4, 2.0)
    s = pd.Series(np.nan, index=h4.index)
    for (ci, pi, px, kind) in sw:
        if kind == +1:
            s.iloc[ci] = px                    # known at the CLOSE of bar ci
    return s.ffill().shift(1).reindex(df.index, method="ffill").values   # shift -> no lookahead


def build(df, gate, wfun):
    """Entry list with a caller-supplied size weight (the ZigZag/EMA/break logic is unchanged)."""
    G.RR = RR; P.RR = RR
    E = G.build_entries(df, gate)              # gives (i, e, stop, tgt, w_pdh)
    return [(i, e, s, t, wfun(i, e, w)) for (i, e, s, t, w) in E]


def walk(df, E):
    h, l, c = (df[k].values for k in ("high", "low", "close"))
    busy = -1; out = []
    for (i, e, stop0, tgt, w) in E:
        if i <= busy or w <= 0: continue
        lim = e - G.FRAC * (e - stop0)
        if lim <= stop0 or lim >= e: continue
        fill = None
        for j in range(i + 1, min(i + 1 + P.FILLWIN, len(c))):
            if h[j] >= tgt: break
            if l[j] <= lim: fill = j; break
        if fill is None: continue
        u = lim - stop0; rew = (tgt - lim) / u
        if l[fill] <= stop0:
            out.append((df.index[fill], w * (-1.0 - G.COST / u))); busy = fill; continue
        R = None; ej = min(fill + P.FWD, len(c) - 1)
        for j in range(fill + 1, min(fill + 1 + P.FWD, len(c))):
            if l[j] <= stop0: R = -1.0; ej = j; break
            if h[j] >= tgt: R = rew; ej = j; break
        if R is None: R = (c[ej] - lim) / u
        out.append((df.index[ej], w * (R - G.COST / u))); busy = ej
    return out


def line(nm, tr, span):
    s = stats(tr, span); R = np.array([x[1] for x in tr])
    f, cg, m = equal_dd_bet(R, span, REF_DD)
    y22 = sum(x[1] for x in tr if x[0].year == 2022)
    eq = np.cumprod(1 + 0.01 * R); pk = np.maximum.accumulate(eq)
    cdd = ((eq[-1] ** (1 / span) - 1)) / ((pk - eq) / pk).max()
    print(f"{nm:<28}{s['n']:>5}{s['pf']:>6.2f}{s['totyr']:>+9.1f}{s['ddp']:>7.1f}%"
          f"{s['retdd']:>8.2f}{cdd:>8.2f}{m:>8.2f}x{y22:>+8.1f}"
          f"{f'{s[chr(73)+chr(83)]:+.0f}/{s[chr(79)+chr(79)+chr(83)]:+.0f}':>12}")
    return tr


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    span = (df.index[-1] - df.index[0]).days / 365.25
    g4 = kama_rising(df, "4h")
    c = df["close"].values
    hh4 = hh4h_series(df)
    pdh = df["high"].resample("1D").max().shift(1).reindex(df.index, method="ffill").values
    strong = lambda i: np.isfinite(hh4[i]) and c[i] > hh4[i]
    above_pdh = lambda i: np.isfinite(pdh[i]) and c[i] > pdh[i]

    n_s = sum(strong(i) for (i, *_r) in G.build_entries(df, g4))
    n_p = sum(above_pdh(i) for (i, *_r) in G.build_entries(df, g4))
    both = sum(strong(i) and above_pdh(i) for (i, *_r) in G.build_entries(df, g4))
    tot = len(G.build_entries(df, g4))
    print(f"entries={tot}  above 4H swing high={n_s} ({100*n_s/tot:.0f}%)  above PDH={n_p} "
          f"({100*n_p/tot:.0f}%)  both={both} ({100*both/tot:.0f}%)")
    print(f"redundancy: P(above PDH | above 4H high) = {100*both/max(1,n_s):.0f}%  "
          f"(if ~100%, HH4H is just PDH in disguise)\n")

    print(f"{'sizing arm':<28}{'n':>5}{'PF':>6}{'totR/yr':>9}{'maxDD':>8}{'tot/DD':>8}"
          f"{'CAGR/DD':>8}{'wealth':>9}{'2022':>8}{'IS/OOS':>12}")
    base = line("base: PDH soft 0.5", walk(df, build(df, g4, lambda i, e, w: w)), span)
    arms = {}
    for wk in (0.0, 0.25, 0.5, 0.75):
        tr = walk(df, build(df, g4, lambda i, e, w, wk=wk: 1.0 if strong(i) else wk))
        arms[f"HH4H soft {wk}"] = line(f"A: HH4H soft {wk}", tr, span)
    tr = walk(df, build(df, g4, lambda i, e, w: 1.0 if (strong(i) and above_pdh(i))
                        else (0.5 if (strong(i) or above_pdh(i)) else 0.25)))
    arms["both-ladder"] = line("B: both (1 / 0.5 / 0.25)", tr, span)

    # ---- per-year ------------------------------------------------------------
    print(f"\n{'per-year totR':<28}" + "".join(f"{y:>7}" for y in YRS))
    for nm, tr in [("base: PDH soft", base)] + [(k, v) for k, v in arms.items()]:
        by = {y: sum(x[1] for x in tr if x[0].year == y) for y in YRS}
        print(f"{nm:<28}" + "".join(f"{by[y]:>+7.0f}" for y in YRS))

    # ---- the arbiter ---------------------------------------------------------
    def monthly(tr):
        s = pd.Series([x[1] for x in tr], index=pd.DatetimeIndex([x[0] for x in tr]))
        return s.resample("ME").sum()
    Mb = monthly(base)
    print(f"\nCIRCULAR BLOCK bootstrap vs base (equal-DD wealth, 2000 draws)")
    print(f"  {'arm':<20}" + "".join(f"{f'P(win) blk{b}':>15}" for b in (1, 3, 6, 12)))
    rng = np.random.default_rng(20260713)
    for nm, tr in arms.items():
        Ma = monthly(tr)
        idx = Ma.index.union(Mb.index)
        A, B = Ma.reindex(idx).fillna(0).values, Mb.reindex(idx).fillna(0).values
        m = len(idx)
        row = []
        for blk in (1, 3, 6, 12):
            nb = int(np.ceil(m / blk)); win = 0
            for _ in range(2000):
                st = rng.integers(0, m, nb)
                k = np.concatenate([(np.arange(s, s + blk) % m) for s in st])[:m]
                if equal_dd_bet(A[k], m / 12, REF_DD)[2] > equal_dd_bet(B[k], m / 12, REF_DD)[2]:
                    win += 1
            row.append(100 * win / 2000)
        print(f"  {nm:<20}" + "".join(f"{v:>14.0f}%" for v in row))


if __name__ == "__main__":
    main()
