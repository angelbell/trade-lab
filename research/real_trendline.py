"""real_trendline.py -- mechanize a genuine MULTI-TOUCH descending trendline (connect real swing
highs, require N touches) + a RETEST entry, vs LuxAlgo's single-anchor fixed-slope ray.

Honest prior: this is still a DETECTOR ("the lever is never the detector"), and gold_bo already is
"break the ZigZag swing-high on a confirmed close, structural stop at the higher-low" -- a better line
likely just correlates MORE with it. So we LEAD WITH THE CHEAP KILL: mfe/mae (vs random) + correlation
vs gold_bo. If redundant (corr>=~0.6, the LuxAlgo outcome) we stop. The RETEST is isolated as a
separately-transferable sub-finding (if it improves entry quality it could be bolted onto gold_bo).

Causal: ZigZag pivots acted on at confirm_idx (no lookahead); line uses only confirmed highs; break =
confirmed close; next-bar fill. Touch judgment is an explicit tol*ATR band + min_touch count (SWEPT).

  .venv/bin/python research/real_trendline.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import resample, swings_zigzag
from research.regime_gate_lab import metrics
from research.portfolio_kama import get_legs

RNG = np.random.default_rng(7)
CSV, TF, COST = "data/vantage_xauusd_h1.csv", "1h", 0.30


def active_resistance(d, atr, sw, tol, min_touch, M, max_slope=0.0):
    """causal descending/flat MULTI-TOUCH resistance line; returns per-bar projected level (NaN=none)."""
    n = len(d)
    slope = np.full(n, np.nan); x0 = np.full(n, np.nan); y0 = np.full(n, np.nan)
    wid = np.full(n, -1, int)                                    # window id (per anchor reconfigure)
    highs = [(c, p, pr) for (c, p, pr, k) in sw if k == +1]      # (confirm_idx, pivot_idx, price)
    w = 0
    for idx in range(len(highs)):
        c_k = highs[idx][0]
        recent = highs[max(0, idx - M + 1):idx + 1]
        if len(recent) < 2:
            continue
        best = None
        for a in range(len(recent)):
            for b in range(a + 1, len(recent)):
                xa, ya = recent[a][1], recent[a][2]; xb, yb = recent[b][1], recent[b][2]
                if xb == xa:
                    continue
                s = (yb - ya) / (xb - xa)
                if s > max_slope:                              # descending/flat only
                    continue
                ok = True; touches = 0
                for (_, xp, yp) in recent:
                    L = ya + s * (xp - xa)
                    band = tol * (atr[xp] if not np.isnan(atr[xp]) else atr[c_k])
                    if yp > L + band:
                        ok = False; break
                    if abs(yp - L) <= band:
                        touches += 1
                if ok and touches >= min_touch:
                    key = (touches, xb - xa)
                    if best is None or key > best[0]:
                        best = (key, s, xa, ya)
        if best is not None:
            _, s, xa, ya = best
            end = highs[idx + 1][0] if idx + 1 < len(highs) else n
            rng = np.arange(c_k, end)
            slope[rng] = s; x0[rng] = xa; y0[rng] = ya; wid[rng] = w; w += 1
    L = y0 + slope * (np.arange(n) - x0)
    return L, wid


def detect(d, atr, L, wid, mode, W, buf, tol):
    """signal bars (enter next bar). break: first confirmed close above line per window.
    retest: after a break, pullback touches the now-support line then closes back above it."""
    C = d["close"].values; Lo = d["low"].values; n = len(C)
    sig = -1; broken = False; rt = None; out = []
    for i in range(n):
        if np.isnan(L[i]):
            sig = -1; broken = False; rt = None; continue
        if wid[i] != sig:                    # new anchor window -> re-arm
            sig = wid[i]; broken = False; rt = None
        if not broken:
            if C[i] > L[i] + buf * atr[i]:
                broken = True
                if mode == "break":
                    out.append(i)
                else:
                    rt = {"bk": i, "touched": False}
        else:
            if mode == "retest" and rt is not None:
                if i - rt["bk"] > W:
                    rt = None
                elif not rt["touched"] and Lo[i] <= L[i] + tol * atr[i]:
                    rt["touched"] = True
                elif rt["touched"] and C[i] > L[i]:
                    out.append(i); rt = None
    return out


def make_trades(d, entries, recent_low, atr, rr, fwd=400):
    op = d["open"].values; H = d["high"].values; Lo = d["low"].values; C = d["close"].values
    tm = d.index; n = len(C); rows = []; busy = -1
    for si in entries:
        ei = si + 1
        if ei <= busy or ei >= n or np.isnan(recent_low[si]):
            continue
        e = op[ei]; stop = recent_low[si]
        if e <= stop:
            continue
        risk = e - stop; tgt = e + rr * risk; R = None; xj = None
        for j in range(ei, min(ei + fwd, n)):
            if Lo[j] <= stop: R, xj = -1.0, j; break
            if H[j] >= tgt: R, xj = rr, j; break
        if R is None:
            xj = min(ei + fwd, n - 1); R = (C[xj] - e) / risk
        R -= COST / risk
        rows.append((tm[ei], R, 1)); busy = xj
    return pd.DataFrame(rows, columns=["time", "R", "dir"])


def mm(d, entries, atr, fwd=100):
    C = d["close"].values; H = d["high"].values; Lo = d["low"].values; n = len(C)
    fe, ae = [], []
    for si in entries:
        i = si + 1
        if i >= n or np.isnan(atr[i]) or atr[i] == 0:
            continue
        e = C[si]; a = atr[i]; hh = H[i:i + fwd]; ll = Lo[i:i + fwd]
        if len(hh) == 0:
            continue
        fe.append((hh.max() - e) / a); ae.append((e - ll.min()) / a)
    fe = np.array(fe); ae = np.clip(np.array(ae), 1e-9, None)
    if len(fe) == 0:
        return 0, np.nan, np.nan, np.nan
    return len(fe), fe.mean(), ae.mean(), fe.mean() / ae.mean()


def main():
    d = resample(load_mt5_csv(CSV), TF)
    atr = ta.atr(d["high"], d["low"], d["close"], length=14).values
    h, l = d["high"].values, d["low"].values
    n = len(d)
    print(f"real_trendline -- gold {TF} {d.index[0].date()}->{d.index[-1].date()} (n={n})  "
          f"prior: detector~0 lift + redundancy w/ gold_bo likely -> CHEAP KILL first")

    ZZ, TOL, MT, M, W, BUF = 2.0, 0.5, 2, 6, 12, 0.05
    sw = swings_zigzag(h, l, atr, ZZ)
    recent_low = np.full(n, np.nan)
    for (c, p, pr, k) in sw:
        if k == -1 and c < n:
            recent_low[c] = pr
    recent_low = pd.Series(recent_low).ffill().values
    L, wid = active_resistance(d, atr, sw, TOL, MT, M)
    print(f"  active-resistance bars: {np.isfinite(L).mean()*100:.0f}% of chart  "
          f"(zz_k{ZZ} tol{TOL}ATR min_touch{MT} M{M})")

    br = detect(d, atr, L, wid, "break", W, BUF, TOL)
    rt = detect(d, atr, L, wid, "retest", W, BUF, TOL)
    print(f"\n  -- entries: break n={len(br)}  retest n={len(rt)} --")

    # ===== PROBE 1: CHEAP KILL -- mfe/mae vs random, + correlation vs gold_bo =====
    print("\n" + "=" * 78 + "\n1. CHEAP KILL: exit-agnostic mfe/mae (vs random long) + corr vs gold_bo")
    valid = np.arange(60, n - 2)
    randbars = list(np.sort(RNG.choice(valid, min(len(br), len(valid)), replace=False)))
    for tag, ents in [("trendline-break", br), ("trendline-retest", rt), ("random-long", randbars)]:
        nn, fe, ae, r = mm(d, ents, atr)
        print(f"    {tag:<18} n={nn:>4}  MFE={fe:.2f} MAE={ae:.2f}  MFE/MAE={r:.2f}")

    legs = get_legs(); gold_bo = legs["gold_bo"]
    cstart = gold_bo.time.min()
    tb = make_trades(d, br, recent_low, atr, rr=2.0); tb = tb[tb.time >= cstart]
    tr = make_trades(d, rt, recent_low, atr, rr=2.0); tr = tr[tr.time >= cstart]
    gb = gold_bo[gold_bo.time >= cstart]

    def ann(x): return x.assign(y=x.time.dt.year).groupby("y").R.sum()
    def mon(x): return x.assign(m=x.time.dt.to_period("M")).groupby("m").R.sum()
    for tag, t in [("break", tb), ("retest", tr)]:
        if len(t) < 10:
            print(f"    corr({tag}, gold_bo): n={len(t)} too few"); continue
        A = pd.concat([ann(t), ann(gb)], axis=1).fillna(0)
        Mo = pd.concat([mon(t), mon(gb)], axis=1).fillna(0)
        print(f"    corr({tag}, gold_bo)  annual={A.iloc[:,0].corr(A.iloc[:,1]):+.2f}  "
              f"monthly={Mo.iloc[:,0].corr(Mo.iloc[:,1]):+.2f}  (>=~0.6 => redundant)")

    # ===== PROBE 2: RETEST vs BREAK-ONLY (transferable even if detector redundant) =====
    print("\n" + "=" * 78 + "\n2. RETEST vs BREAK-ONLY (win%, meanR, fakeout=immediate stop-out)")
    for tag, t in [("break-only", tb), ("break+retest", tr)]:
        if len(t) < 10:
            print(f"    {tag:<14} n={len(t)} too few"); continue
        fake = (t.R <= -0.99).mean() * 100
        m = metrics(t)
        print(f"    {tag:<14} n={m['n']:>4} win={(t.R>0).mean()*100:>3.0f}% meanR={t.R.mean():+.3f} "
              f"fakeout={fake:>3.0f}% | CAGR/DD={m['cdd']:5.2f} IS={m['isr']:+.2f} OOS={m['oos']:+.2f}")

    # ===== PROBE 2b: is retest a REAL false-break FILTER, or just a luck-sorter? =====
    # The open lead: break+retest lifted CAGR/DD 0.36->0.87 but fakeout% was ~unchanged,
    # so the lift may be mere trade-thinning, not false-break removal. Decisive test:
    # randomly drop break-only down to the retest count many times; where does the ACTUAL
    # retest result fall in that null? >=~90%ile => real filter; mid-pack => luck-sorter.
    print("\n" + "=" * 78 + "\n2b. RETEST vs RANDOM-EQUAL-DROP null (is it a filter or a luck-sorter?)")
    if len(tr) >= 10 and len(tb) > len(tr):
        k = len(tr)
        act_cdd = metrics(tr)["cdd"]; act_mr = tr.R.mean()
        nrep = 3000
        ds_cdd = np.empty(nrep); ds_mr = np.empty(nrep)
        tb_sorted = tb.sort_values("time").reset_index(drop=True)
        for i in range(nrep):
            samp = tb_sorted.iloc[np.sort(RNG.choice(len(tb_sorted), k, replace=False))]
            mi = metrics(samp)
            ds_cdd[i] = mi["cdd"] if mi else np.nan; ds_mr[i] = samp.R.mean()
        pc_cdd = (ds_cdd < act_cdd).mean() * 100
        pc_mr = (ds_mr < act_mr).mean() * 100
        print(f"    retest keeps {k}/{len(tb)} break trades.  random-drop null n={nrep}")
        print(f"    CAGR/DD : actual={act_cdd:5.2f}  null mean={np.nanmean(ds_cdd):5.2f} "
              f"[{np.nanpercentile(ds_cdd,5):.2f},{np.nanpercentile(ds_cdd,95):.2f}]  -> {pc_cdd:4.1f}%ile")
        print(f"    meanR   : actual={act_mr:+.3f}  null mean={ds_mr.mean():+.3f} "
              f"[{np.percentile(ds_mr,5):+.3f},{np.percentile(ds_mr,95):+.3f}]  -> {pc_mr:4.1f}%ile")
        print("    VERDICT: >=~90%ile both => REAL false-break filter; mid-pack => luck-sorter (bury).")
    else:
        print(f"    skip: tr={len(tr)} tb={len(tb)} (need tr>=10 and tb>tr)")

    # ===== PROBE 3: standalone surface (plateau?) + vs gold_bo 1.05 =====
    print("\n" + "=" * 78 + "\n3. SURFACE (break, RR2): meanR / CAGR-DD over tol x min_touch x zz_k -- plateau?")
    print(f"  {'zz_k':>5}{'mt':>4} | " + "  ".join(f"tol{t}(mR/cdd)" for t in (0.25, 0.5, 1.0)))
    for zz in (1.5, 2.0, 3.0):
        sw2 = swings_zigzag(h, l, atr, zz)
        rl = np.full(n, np.nan)
        for (c, p, pr, k) in sw2:
            if k == -1 and c < n: rl[c] = pr
        rl = pd.Series(rl).ffill().values
        for mt in (2, 3):
            cells = []
            for tol in (0.25, 0.5, 1.0):
                L2, wid2 = active_resistance(d, atr, sw2, tol, mt, M)
                e2 = detect(d, atr, L2, wid2, "break", W, BUF, tol)
                t2 = make_trades(d, e2, rl, atr, rr=2.0)
                m = metrics(t2)
                cells.append(f"{t2.R.mean():+.2f}/{m['cdd']:4.2f}" if m else "  --   ")
            print(f"  {zz:>5}{mt:>4} | " + "  ".join(f"{c:<13}" for c in cells))
    print("\n  refs: gold_bo CAGR/DD=1.05.  PASS = beats gold_bo character AND low corr (<~0.4) + plateau.")
    print("=" * 78)


if __name__ == "__main__":
    main()
