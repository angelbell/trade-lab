"""OVERFIT AUDIT of the proposed btc15m_L v2 (before anyone believes today's numbers).

v2 = 4h-KAMA gate + pullback limit frac0.30 + RR4.5 + no ratchet
     + size ladder (above 4H swing high AND above PDH = 1.0 / one = 0.5 / neither = 0.25)
v1 = the shipped Pine: daily-KAMA gate + RR4 + TP2 ratchet + PDH soft 0.5

The PBO grid is THE SPACE WE ACTUALLY SEARCHED TODAY -- anything narrower would flatter us:
    gate_tf {1D, 4h} x rr {3.0,3.5,4.0,4.5,5.0} x weak_w {0.25,0.5,0.75,1.0}
    x hh_k {1.5,2.0,2.5} x ratchet {off,on}   = 240 configs
(weak_w = 1.0 means "no HH4H sizing", i.e. the v1 sizing -- so v1 lives inside the grid too.)

A. Deflated Sharpe over a trial-count sweep (functions imported from research/overfit_audit.py)
B. PBO via CSCV on the 240-config monthly matrix -- when we pick the IS-best config, how often is
   it OOS-below-median? PBO ~0.5 = we are p-hacking; PBO < ~0.3 = the choice carries information.
C. Block-bootstrap CI on CAGR/DD + a mean-removed null (is the edge distinguishable from luck?)
Run: .venv/bin/python scratchpad/btc15m_v2_audit.py
"""
import sys, os, io, contextlib, warnings, itertools
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src.data_loader import load_mt5_csv
import pine_replica_btc15m as P, btc15m_gate_ab as G
from pine_replica_btc15m import stats
from btc15m_gate_ab import kama_rising, equal_dd_bet
from breakout_wave import swings_zigzag
from trend_leg_aging import atr as atr_fn
from research.overfit_audit import psr, sr0, cscv, cdd_R, block_resample

ROOT, START = "/home/angelbell/dev/auto-trade", "2018-10-01"


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    span = (df.index[-1] - df.index[0]).days / 365.25
    h, l, c = (df[k].values for k in ("high", "low", "close"))
    pdh = df["high"].resample("1D").max().shift(1).reindex(df.index, method="ffill").values
    h4 = df.resample("4h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    a4 = atr_fn(h4["high"].values, h4["low"].values, h4["close"].values)

    def hh_series(k):
        sw = swings_zigzag(h4["high"].values, h4["low"].values, a4, k)
        s = pd.Series(np.nan, index=h4.index)
        for (ci, pi, px, kind) in sw:
            if kind == +1:
                s.iloc[ci] = px
        return s.ffill().shift(1).reindex(df.index, method="ffill").values

    HH = {k: hh_series(k) for k in (1.5, 2.0, 2.5)}
    GATES = {"1D": kama_rising(df, "1D"), "4h": kama_rising(df, "4h")}

    def run(gate_tf, rr, weak_w, hh_k, ratchet):
        G.RR = rr; P.RR = rr
        E = G.build_entries(df, GATES[gate_tf])
        hh = HH[hh_k]
        busy = -1; out = []
        for (i, e, stop0, tgt, w_pdh) in E:
            if i <= busy: continue
            if weak_w >= 1.0:
                w = w_pdh                                  # v1 sizing (PDH soft only)
            else:
                s_hh = np.isfinite(hh[i]) and c[i] > hh[i]
                s_pd = np.isfinite(pdh[i]) and c[i] > pdh[i]
                w = 1.0 if (s_hh and s_pd) else (0.5 if (s_hh or s_pd) else weak_w)
            lim = e - G.FRAC * (e - stop0)
            if lim <= stop0 or lim >= e or w <= 0: continue
            fill = None
            for j in range(i + 1, min(i + 1 + P.FILLWIN, len(c))):
                if h[j] >= tgt: break
                if l[j] <= lim: fill = j; break
            if fill is None: continue
            u = lim - stop0; rew = (tgt - lim) / u
            if l[fill] <= stop0:
                out.append((df.index[fill], w * (-1.0 - G.COST / u))); busy = fill; continue
            R = None; ej = min(fill + P.FWD, len(c) - 1); cur = stop0; rt = False
            for j in range(fill + 1, min(fill + 1 + P.FWD, len(c))):
                if l[j] <= cur: R = (cur - lim) / u; ej = j; break
                if h[j] >= tgt: R = rew; ej = j; break
                if ratchet and not rt and h[j] >= lim + 3.5 * u:
                    cur = max(cur, lim + 2.0 * u); rt = True
            if R is None: R = (c[ej] - lim) / u
            out.append((df.index[ej], w * (R - G.COST / u))); busy = ej
        return pd.DataFrame(out, columns=["t", "R"])

    # ---------------- build the grid ----------------
    grid = list(itertools.product(["1D", "4h"], [3.0, 3.5, 4.0, 4.5, 5.0],
                                  [0.25, 0.5, 0.75, 1.0], [1.5, 2.0, 2.5], [False, True]))
    print(f"running the {len(grid)}-config search grid (the space we actually walked today)...")
    cols, srs, meta = {}, [], []
    for n, (g, rr, ww, hk, rt) in enumerate(grid):
        t = run(g, rr, ww, hk, rt)
        if len(t) < 20: continue
        s = t.copy(); s["m"] = s.t.dt.to_period("M")
        cols[f"c{n}"] = s.groupby("m").R.sum()
        srs.append(t.R.mean() / t.R.std(ddof=1))
        meta.append((g, rr, ww, hk, rt))
    M = pd.concat(cols, axis=1).fillna(0.0)
    print(f"  {M.shape[1]} configs x {M.shape[0]} months\n")

    v1 = run("1D", 4.0, 1.0, 2.0, True)
    v2 = run("4h", 4.5, 0.25, 2.0, False)

    # ---------------- A. Deflated Sharpe ----------------
    print("=" * 78)
    print("A. DEFLATED SHARPE (per-trade). DSR>0.95 = survives the trial haircut.")
    V = float(np.var(srs))
    Ns = [1, 10, 25, 50, 100, 200, 240]
    print(f"  trial-SR variance across the grid V={V:.5f}")
    print(f"  {'leg':<26}{'n':>5}{'SR/tr':>7}{'t':>6}{'skew':>6}{'kurt':>6}  "
          + " ".join(f"DSR@{N:<4}" for N in Ns))
    for nm, t in (("v1 (shipped Pine)", v1), ("v2 (proposed)", v2)):
        r = t.R.values
        _, sr, g1, g4 = psr(r, 0.0)
        ds = [psr(r, sr0(N, V))[0] for N in Ns]
        print(f"  {nm:<26}{len(r):>5}{sr:>7.3f}{sr*np.sqrt(len(r)):>6.2f}{g1:>6.2f}{g4:>6.1f}  "
              + " ".join(f"{d:>7.2f}" for d in ds))

    # ---------------- B. PBO ----------------
    print("\n" + "=" * 78)
    print("B. PBO via CSCV on the 240-config grid (0.5 = pure p-hacking, <0.3 = the pick informs)")
    pbo, oos_sr, oos_loss = cscv(M.values)
    npbo, _, _ = cscv(np.random.default_rng(0).normal(size=M.shape))
    print(f"  PBO(real grid) = {pbo:.2f}   PBO(pure-noise control) = {npbo:.2f}  "
          f"gap = {pbo-npbo:+.2f}")
    print(f"  mean OOS Sharpe of the IS-best config = {oos_sr:+.3f}   "
          f"P(the IS-best config is OOS-negative) = {oos_loss:.2f}")

    # ---------------- C. block-bootstrap CI + null ----------------
    print("\n" + "=" * 78)
    print("C. block-bootstrap CI on CAGR/DD (12-trade blocks) + mean-removed null")
    rng = np.random.default_rng(7)
    for nm, t in (("v1 (shipped Pine)", v1), ("v2 (proposed)", v2)):
        r = t.R.values
        obs = cdd_R(r, span)[2]
        bs = [cdd_R(block_resample(r, 12, rng), span)[2] for _ in range(2000)]
        r0 = r - r.mean()
        null = [cdd_R(block_resample(r0, 12, rng), span)[2] for _ in range(2000)]
        p = float(np.mean(np.array(null) >= obs))
        print(f"  {nm:<26} CAGR/DD obs {obs:>5.2f}  CI[5th {np.percentile(bs,5):>5.2f}, "
              f"50th {np.percentile(bs,50):>5.2f}, 95th {np.percentile(bs,95):>5.2f}]  "
              f"null p={p:.4f}")


if __name__ == "__main__":
    main()
