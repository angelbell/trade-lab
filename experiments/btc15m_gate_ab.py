"""btc15m_L: KAMA gate timeframe A/B -- DAILY (Pine default) vs 4H (aggressive) vs BOTH.

Everything except the gate is imported from pine_replica_btc15m (the Pine-faithful replica):
same ZigZag(2xATR)/EMA80/Pattern-B detector, same pullback limit (frac 0.30, fillWin 200),
same PDH soft sizing (0.5 inside prior-day range), same exits (stop=L2, target=e+4R,
optional TP2 3.5R -> stop +2.0R ratchet), same $15 round-trip cost, same re-arm rule.
Flow exit is NOT used (retracted: broker-clock lookahead).

Gate (no lookahead in either case): KAMA(14) on the resampled close, rising is judged on the
PRIOR COMPLETED bar of the gate timeframe -- (kama > kama.shift(1)).shift(1), ffill onto 15m.
This mirrors request.security(..., f_kama(close,14)[1] > ...[2], lookahead_off).

Run: .venv/bin/python experiments/btc15m_gate_ab.py
"""
import sys, os
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta, io, contextlib
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag, kama_adaptive
from pine_replica_btc15m import walk, stats, ROOT, START, BO, FRAC, RR, COST, PDH_W


def kama_rising(df, tf):
    ck = df["close"].resample(tf).last().dropna()
    k = kama_adaptive(ck, 14)
    return ((k > k.shift(1)).shift(1)).reindex(df.index, method="ffill").fillna(False).values


def build_entries(df, gate):
    """Identical to pine_replica_btc15m.build_entries except for the regime gate."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    a = ta.atr(df["high"], df["low"], df["close"], 14).values
    es = df["close"].ewm(span=80, adjust=False).mean().values
    pdh = df["high"].resample("1D").max().shift(1).reindex(df.index, method="ffill").values
    kreg = gate
    sw = swings_zigzag(h, l, a, 2.0)
    E = []
    for t in range(2, len(sw)):
        (cL2, iL2, pL2, kL2), (cH1, iH1, pH1, kH1), (cL0, iL0, pL0, kL0) = sw[t], sw[t-1], sw[t-2]
        if not (kL2 == -1 and kH1 == +1 and kL0 == -1): continue
        if pL2 <= pL0 or pH1 - pL0 <= 0: continue
        if not np.isnan(es[cL2]) and pH1 < es[cL2]: continue
        e_i = None
        for j in range(cL2 + 1, min(cL2 + 1 + BO, len(c))):
            if c[j] > pH1: e_i = j; break
        if e_i is None or not kreg[e_i]: continue
        e = c[e_i]; stop = pL2
        if e - stop <= 0: continue
        w = 1.0 if (np.isfinite(pdh[e_i]) and e > pdh[e_i]) else PDH_W
        E.append((e_i, e, stop, e + RR * (e - stop), w))
    E.sort(key=lambda x: x[0])
    seen = set(); U = []
    for en in E:
        if en[0] in seen: continue
        seen.add(en[0]); U.append(en)
    return U


def equal_dd_bet(R, span, target_dd):
    lo, hi = 0.001, 0.30
    for _ in range(60):
        mid = (lo + hi) / 2
        eq = np.cumprod(1 + mid * R); pk = np.maximum.accumulate(eq)
        if ((pk - eq) / pk).max() * 100 > target_dd: hi = mid
        else: lo = mid
    eq = np.cumprod(1 + lo * R)
    return lo, (eq[-1] ** (1 / span) - 1) * 100, eq[-1]


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    span = (df.index[-1] - df.index[0]).days / 365.25
    noflow = pd.Series(np.nan, index=df.index)

    gD, g4 = kama_rising(df, "1D"), kama_rising(df, "4h")
    gates = {"daily (Pine default)": gD, "4h (aggressive)": g4, "daily AND 4h": gD & g4}
    print(f"span {span:.1f}yr | gate ON% of bars: "
          + " | ".join(f"{k} {100*v.mean():.0f}%" for k, v in gates.items()) + "\n")

    print(f"{'gate':<22}{'ratchet':<9}{'n':>5}{'N/yr':>6}{'PF':>6}{'totR/yr':>9}{'maxDD%':>8}"
          f"{'tot/DD':>8}{'meanR':>8}{'CAGR%':>7}{'grn':>6}{'IS/OOS':>12}")
    res = {}
    for gname, g in gates.items():
        E = build_entries(df, g)
        for rname, ur in (("off", False), ("ON", True)):
            tr = walk(df, E, False, ur, noflow)
            s = stats(tr, span)
            res[(gname, rname)] = (tr, s)
            print(f"{gname:<22}{rname:<9}{s['n']:>5}{s['npy']:>6.1f}{s['pf']:>6.2f}"
                  f"{s['totyr']:>9.2f}{s['ddp']:>7.1f}%{s['retdd']:>8.2f}{s['meanR']:>+8.3f}"
                  f"{s['cagr']:>7.1f}{s['grn']:>5.0f}%"
                  f"{f'{s[chr(73)+chr(83)]:+.0f}/{s[chr(79)+chr(79)+chr(83)]:+.0f}':>12}")

    # ---- per-year totR (the claim to test: does the 4h gate fix the bleed years?) ----
    print("\nper-year totR (risk-normalised R, ratchet ON):")
    yrs = sorted({t.year for tr, _ in res.values() for t, *_ in tr})
    print(f"{'gate':<22}" + "".join(f"{y:>8}" for y in yrs))
    for gname in gates:
        tr, _ = res[(gname, "ON")]
        by = {y: sum(r for t, r, w in tr if t.year == y) for y in yrs}
        cnt = {y: sum(1 for t, r, w in tr if t.year == y) for y in yrs}
        print(f"{gname:<22}" + "".join(f"{by[y]:>+8.0f}" for y in yrs))
        print(f"{'  (n)':<22}" + "".join(f"{cnt[y]:>8}" for y in yrs))

    # ---- equal-DD comparison: scale the bet so every cell runs the same maxDD ----
    ref_dd = res[("daily (Pine default)", "off")][1]["ddp"]
    print(f"\nequal-DD comparison (bet scaled so each cell runs maxDD {ref_dd:.1f}% "
          f"= the daily/no-ratchet reference), {span:.1f}yr:")
    for (gname, rname), (tr, s) in res.items():
        R = np.array([x[1] for x in tr])
        f, cagr, mult = equal_dd_bet(R, span, ref_dd)
        print(f"  {gname:<22} ratchet {rname:<4} f={100*f:>5.2f}%  CAGR={cagr:>6.1f}%  "
              f"final={mult:>6.2f}x")

    # ---- overlap: are the two gates selecting the same trades? ----
    dD = {t for t, *_ in res[("daily (Pine default)", "ON")][0]}
    d4 = {t for t, *_ in res[("4h (aggressive)", "ON")][0]}
    print(f"\ntrade overlap (fill timestamps): daily {len(dD)}, 4h {len(d4)}, "
          f"shared {len(dD & d4)} ({100*len(dD & d4)/len(dD | d4):.0f}% of the union)")


if __name__ == "__main__":
    main()
