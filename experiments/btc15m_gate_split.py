"""Decompose the 4h gate: the 4h-gate leg (B) = the trades the daily gate also allows (C-like)
PLUS the trades where 4h is rising but the daily KAMA is NOT (the "bear-rally" trades).
If the 2022 bleed lives entirely in that second bucket, a SOFT gate (half size when only 4h
agrees, mirroring the PDH soft sizing already in the Pine) should keep B's return and kill its
bleed. Frozen before running: soft-0.5 is adopted over B only if it beats B on equal-DD wealth
AND does not make 2022 worse; a "kills the bleed but costs the return" outcome = report as-is.
Run: .venv/bin/python experiments/btc15m_gate_split.py
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

REF_DD = 19.0


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    span = (df.index[-1] - df.index[0]).days / 365.25
    noflow = pd.Series(np.nan, index=df.index)
    gD, g4 = kama_rising(df, "1D"), kama_rising(df, "4h")

    E4 = build_entries(df, g4)
    trB = walk(df, E4, False, False, noflow)

    # which of B's entries had the daily gate DOWN at the break bar?
    bar_of = {df.index[e[0]]: e[0] for e in E4}
    # map fill-time -> entry bar: re-walk bookkeeping is internal, so tag by entry bar order
    ent_by_bar = {e[0]: e for e in E4}
    daily_up_at = {b: bool(gD[b]) for b in ent_by_bar}

    # rebuild B with a per-trade daily flag by walking each entry subset separately is unsafe
    # (re-arm couples trades), so instead: run B once, and label each trade by the entry that
    # produced it -- walk() consumes E in order and skips entries while busy, so replay the
    # same skip logic to recover the entry bar of every executed trade.
    def replay_labels(df, E, use_ratchet):
        h, l, c, o = (df[k].values for k in ("high", "low", "close", "open"))
        busy = -1; labels = []
        FRAC, RR, COST, FILLWIN, FWD = P.FRAC, P.RR, P.COST, P.FILLWIN, P.FWD
        for (i, e, stop0, tgt, w) in E:
            if i <= busy: continue
            lim = e - FRAC * (e - stop0)
            if lim <= stop0 or lim >= e: continue
            fill = None
            for j in range(i + 1, min(i + 1 + FILLWIN, len(c))):
                if h[j] >= tgt: break
                if l[j] <= lim: fill = j; break
            if fill is None: continue
            u = lim - stop0; reward = tgt - lim
            if l[fill] <= stop0:
                labels.append((df.index[fill], i)); busy = fill; continue
            cur = stop0; ratched = False; R = None; exit_j = min(fill + FWD, len(c) - 1)
            for j in range(fill + 1, min(fill + 1 + FWD, len(c))):
                if l[j] <= cur: R = 1; exit_j = j; break
                if h[j] >= tgt: R = 1; exit_j = j; break
                if use_ratchet and not ratched and h[j] >= lim + P.RATCH[0] * u:
                    cur = max(cur, lim + P.RATCH[1] * u); ratched = True
            labels.append((df.index[fill], i)); busy = exit_j
        return labels

    lab = replay_labels(df, E4, False)
    assert len(lab) == len(trB) and all(a[0] == b[0] for a, b in zip(lab, trB)), "label mismatch"

    yrs = sorted({t.year for t, *_ in trB})
    both = [(t, r) for (t, r, w), (_, i) in zip(trB, lab) if daily_up_at[i]]
    only4 = [(t, r) for (t, r, w), (_, i) in zip(trB, lab) if not daily_up_at[i]]
    print(f"B (4h gate, no ratchet): n={len(trB)}, totR={sum(x[1] for x in trB):+.0f}")
    print(f"  bucket 1 'daily agrees'   n={len(both):>4}  totR={sum(x[1] for x in both):+7.1f}  "
          f"meanR {np.mean([x[1] for x in both]):+.3f}")
    print(f"  bucket 2 'only 4h up'     n={len(only4):>4}  totR={sum(x[1] for x in only4):+7.1f}  "
          f"meanR {np.mean([x[1] for x in only4]):+.3f}   <- the bear-rally trades")
    print(f"\n{'per-year totR':<26}" + "".join(f"{y:>7}" for y in yrs))
    for nm, bk in (("  bucket 1 daily agrees", both), ("  bucket 2 only 4h up", only4)):
        by = {y: sum(r for t, r in bk if t.year == y) for y in yrs}
        print(f"{nm:<26}" + "".join(f"{by[y]:>+7.0f}" for y in yrs))

    # ---- soft gate: half size when only the 4h gate agrees -------------------
    print("\nsoft gate (size x0.5 when the daily KAMA disagrees), vs the finalists:")
    rows = {}
    for wsoft in (0.0, 0.25, 0.5, 0.75, 1.0):
        E = [(i, e, s, t, w * (1.0 if daily_up_at[i] else wsoft)) for (i, e, s, t, w) in E4]
        E = [x for x in E if x[4] > 0]
        tr = walk(df, E, False, False, noflow)
        st = stats(tr, span)
        R = np.array([x[1] for x in tr])
        f, cagr, mult = equal_dd_bet(R, span, REF_DD)
        by22 = sum(x[1] for x in tr if x[0].year == 2022)
        rows[wsoft] = (st, f, mult, by22)
        tag = {0.0: " (= C, daily AND 4h)", 1.0: " (= B, plain 4h)"}.get(wsoft, "")
        print(f"  w={wsoft:<5}{tag:<22} n={st['n']:>4} PF {st['pf']:.2f} totR/yr {st['totyr']:>+5.1f}"
              f"  maxDD {st['ddp']:>4.1f}%  tot/DD {st['retdd']:>5.2f}  grn {st['grn']:>3.0f}%"
              f"  eqDD f {100*f:.2f}%  wealth {mult:>5.2f}x  2022 {by22:>+5.1f}R")


if __name__ == "__main__":
    main()
