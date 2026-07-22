"""Is btc15m_L arm B's drawdown an EXIT problem or an ENTRY/REGIME problem?  (user hypothesis)

B = 4h KAMA gate, no ratchet, RR4 far fixed target. maxDD 21.4% @1% risk; 2022 = -14R.

The test that separates the two stories, per trade:
    MFE = the best unrealised R the trade ever showed (peak favourable excursion, in R of the
          realised risk u = lim - stop), measured on the actual bar path.
  EXIT problem   -> losers built real profit and gave it back (high MFE among losers; the money
                    was on the table and the exit rule left it there).
  ENTRY problem  -> losers went straight to the stop (MFE near 0); no exit rule could have saved
                    them, because there was never anything to take.
Reported: whole sample, the maxDD window, and 2022 alone. Plus the CEILING: what the leg would
earn with an oracle exit (take the MFE) -- the most any exit change could possibly buy.
Run: .venv/bin/python experiments/btc15m_B_dd_anatomy.py
"""
import sys, os, io, contextlib, warnings
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src.data_loader import load_mt5_csv
import pine_replica_btc15m as P
from btc15m_gate_ab import build_entries, kama_rising

ROOT, START = "/home/angelbell/dev/auto-trade", "2018-10-01"


def walk_mfe(df, E):
    """Same walk as the replica (ratchet off) but recording MFE and the exit reason."""
    h, l, c = (df[k].values for k in ("high", "low", "close"))
    busy = -1
    out = []
    for (i, e, stop0, tgt, w) in E:
        if i <= busy: continue
        lim = e - P.FRAC * (e - stop0)
        if lim <= stop0 or lim >= e: continue
        fill = None
        for j in range(i + 1, min(i + 1 + P.FILLWIN, len(c))):
            if h[j] >= tgt: break
            if l[j] <= lim: fill = j; break
        if fill is None: continue
        u = lim - stop0
        if l[fill] <= stop0:
            out.append(dict(t=df.index[fill], R=w * (-1.0 - P.COST / u), mfe=0.0,
                            why="stop", w=w)); busy = fill; continue
        R, why, mfe = None, "time", 0.0
        exit_j = min(fill + P.FWD, len(c) - 1)
        for j in range(fill + 1, min(fill + 1 + P.FWD, len(c))):
            mfe = max(mfe, (h[j] - lim) / u)
            if l[j] <= stop0: R, why, exit_j = -1.0, "stop", j; break
            if h[j] >= tgt: R, why, exit_j = (tgt - lim) / u, "target", j; break
        if R is None: R = (c[exit_j] - lim) / u
        out.append(dict(t=df.index[fill], R=w * (R - P.COST / u), mfe=mfe, why=why, w=w))
        busy = exit_j
    return pd.DataFrame(out)


def dd_window(R, t):
    cum = np.cumsum(R); pk = np.maximum.accumulate(cum)
    ddv = pk - cum
    end = int(np.argmax(ddv))
    start = int(np.argmax(cum[:end + 1])) if end > 0 else 0
    return start, end, ddv[end]


def report(d, tag):
    los = d[d.R <= 0]
    win = d[d.R > 0]
    if len(d) == 0: return
    print(f"\n{tag}: n={len(d)}  win%={100*len(win)/len(d):.0f}  totR={d.R.sum():+.1f}")
    print(f"  losers n={len(los)}  their MFE (peak unrealised R before dying):")
    for lo, hi, nm in ((0, 0.5, "never got going (MFE < 0.5R)"), (0.5, 1.0, "0.5-1.0R"),
                       (1.0, 2.0, "1.0-2.0R"), (2.0, 99, "2R+ (real giveback)")):
        s = los[(los.mfe >= lo) & (los.mfe < hi)]
        print(f"    {nm:<32} {len(s):>4}  ({100*len(s)/max(1,len(los)):>3.0f}% of losers)  "
              f"totR {s.R.sum():>+7.1f}")
    print(f"  loser MFE: median {los.mfe.median():.2f}R  mean {los.mfe.mean():.2f}R")
    print(f"  winners: median MFE {win.mfe.median():.2f}R (target is 5.7R off the limit)")


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    E = build_entries(df, kama_rising(df, "4h"))
    d = walk_mfe(df, E)
    R = d.R.values
    s, e, dd = dd_window(R, d.t.values)
    print(f"arm B (4h gate, no ratchet): n={len(d)} totR={R.sum():+.0f} maxDD={dd:.1f}R")
    print(f"the maxDD window: {d.t.iloc[s].date()} -> {d.t.iloc[e].date()} "
          f"({e-s} trades, {(d.t.iloc[e]-d.t.iloc[s]).days} days)")

    report(d, "WHOLE SAMPLE")
    report(d.iloc[s:e + 1], f"THE MAX-DD WINDOW ({d.t.iloc[s].date()} -> {d.t.iloc[e].date()})")
    report(d[d.t.dt.year == 2022], "2022 ALONE")

    # ---- the ceiling: how much can ANY exit change possibly buy? -------------
    print("\nCEILING -- what an ORACLE exit (sell at the exact MFE) would earn:")
    for tag, sub in (("whole sample", d), ("2022", d[d.t.dt.year == 2022])):
        real = sub.R.sum()
        oracle = (sub.w * (sub.mfe - P.COST / 1)).sum()   # cost term approximate; MFE in R
        # a fairer oracle: cap at the actual target (can't do better than the far target rule)
        capped = (sub.w * np.minimum(sub.mfe, 5.7)).sum()
        print(f"  {tag:<14} actual {real:>+7.1f}R   oracle-at-MFE {capped:>+7.1f}R   "
              f"headroom {capped-real:>+7.1f}R")
    print("  (the oracle is unattainable -- it is the LIMIT of what any exit rule could add.)")


if __name__ == "__main__":
    main()
