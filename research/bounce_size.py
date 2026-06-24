"""bounce_size.py -- characterize the RAW bounce off HTF S/R (no strategy, no cost).

Question: when price tags a higher-TF support/resistance level, how many pips does
it actually rebound, on average?  This sets the *ceiling* for any bounce edge:
if the typical rebound is only a few pips, no entry rule can beat a ~3-pip spread.

Definition (lookahead is FINE here -- we are describing the data, not trading it):
  * HTF pivots (15m+1h fractals) = proven S/R, confirmed k bars later (htf_pivots).
  * keep the most-recent `sr_keep` pivots as the active levels.
  * a TAG = price's low comes within tol*ATR of the nearest support below price
    (mirror for resistance). Dedupe: ignore tags within `horizon` bars of a prior one.
  * reversal low = min low over the `dip` bars right after the tag (bottom of the dip).
  * bounce peak    = max high over `horizon` bars after the tag.
  * bounce_pips    = (peak - reversal_low) / PIP  (mirror for resistance).
We report the DISTRIBUTION (median/quartiles), not just the mean, and the share of
tags that rebound past a few realistic thresholds.

  .venv/bin/python research/bounce_size.py --csv data/vantage_xauusd_m5.csv --split is
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from scalp_lab import htf_pivots, SPLITS, PIP


def merged_pivots(d, piv_tf, k):
    sps, sxs, rps, rxs = [], [], [], []
    for rule in (t.strip() for t in piv_tf.split(",") if t.strip()):
        a, b, c2, d2 = htf_pivots(d, rule, k)
        sps.append(a); sxs.append(b); rps.append(c2); rxs.append(d2)
    spos = np.concatenate(sps); sp = np.concatenate(sxs)
    rpos = np.concatenate(rps); rp = np.concatenate(rxs)
    o1 = np.argsort(spos, kind="stable"); o2 = np.argsort(rpos, kind="stable")
    return spos[o1], sp[o1], rpos[o2], rp[o2]


def bounces(d, levels_pos, levels_px, side, tol_atr, sr_keep, dip, horizon):
    """side=+1 support (bounce up), -1 resistance (bounce down)."""
    hv, lv = d["high"].values, d["low"].values
    atr = ta.atr(d["high"], d["low"], d["close"], length=14).values
    n = len(hv)
    active = []          # (pos, price) most-recent kept pivots
    ip = 0
    out = []
    last_tag = -10**9
    i = 30
    while i < n - horizon:
        while ip < len(levels_pos) and levels_pos[ip] <= i:
            active.append((levels_pos[ip], levels_px[ip])); ip += 1
            if len(active) > sr_keep:
                active.pop(0)
        if np.isnan(atr[i]) or not active or i - last_tag < horizon:
            i += 1; continue
        tol = tol_atr * atr[i]
        if side > 0:
            cands = [lv_ for _, lv_ in active if lv_ <= lv[i] + tol]      # support at/below the dip
            lvl = max(cands) if cands else None
            tag = lvl is not None and lv[i] <= lvl + tol
        else:
            cands = [lv_ for _, lv_ in active if lv_ >= hv[i] - tol]
            lvl = min(cands) if cands else None
            tag = lvl is not None and hv[i] >= lvl - tol
        if not tag:
            i += 1; continue
        if side > 0:
            rev_low = lv[i:i + dip].min()
            peak = hv[i:i + horizon].max()
            b = (peak - rev_low) / PIP
        else:
            rev_hi = hv[i:i + dip].max()
            trough = lv[i:i + horizon].min()
            b = (rev_hi - trough) / PIP
        out.append(b)
        last_tag = i
        i += 1
    return np.asarray(out)


def describe(b, label):
    if len(b) == 0:
        print(f"  {label:<12} no tags"); return
    q = np.percentile(b, [10, 25, 50, 75, 90])
    print(f"  {label:<12} tags={len(b):>4}  mean={b.mean():5.1f}  "
          f"median={q[2]:5.1f}  p25/p75={q[1]:.0f}/{q[3]:.0f}  p10/p90={q[0]:.0f}/{q[4]:.0f} pips")
    for thr in (10, 20, 30, 50):
        print(f"        >= {thr:>2}pip: {(b >= thr).mean()*100:4.0f}%", end="")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", default="is", choices=["is", "val", "test"])
    ap.add_argument("--piv-tf", default="15min,1h")
    ap.add_argument("--piv-k", type=int, default=3)
    ap.add_argument("--sr-keep", type=int, default=2)
    ap.add_argument("--tol-atr", type=float, default=0.5)
    ap.add_argument("--dip", type=int, default=6, help="bars after tag to find the reversal low")
    ap.add_argument("--horizon", type=int, default=48, help="bars to measure the bounce peak (48=4h)")
    p = ap.parse_args()

    s, e = SPLITS[p.split]
    d = load_mt5_csv(p.csv).loc[s:e]
    spos, sp, rpos, rp = merged_pivots(d, p.piv_tf, p.piv_k)
    print(f"\n=== bounce size  split={p.split}  {d.index[0]} -> {d.index[-1]}  "
          f"piv={p.piv_tf} k{p.piv_k} keep{p.sr_keep} tol{p.tol_atr}ATR dip{p.dip} horizon{p.horizon} ===")
    for H in sorted({12, 48, p.horizon}):
        print(f"  -- horizon {H} bars ({H*5//60}h{H*5%60:02d}m) --")
        describe(bounces(d, spos, sp, +1, p.tol_atr, p.sr_keep, p.dip, H), f"support  H{H}")
        describe(bounces(d, rpos, rp, -1, p.tol_atr, p.sr_keep, p.dip, H), f"resist   H{H}")


if __name__ == "__main__":
    main()
