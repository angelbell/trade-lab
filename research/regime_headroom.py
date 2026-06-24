"""regime_headroom.py -- WHERE does the mech->oracle regime headroom live, and is it
predictable from observable entry-time features?

regime_ceiling.py: same fixed gold-1H-breakout entries, deploy-timing alone moves CAGR/DD
0.35 (always-on) -> 0.69 (mech SMA150+slope gate) -> 1.54 (ORACLE = perfect forward-regime
foresight). The oracle SKIPS trades whose forward regime turns out bad. This asks: do those
oracle-SKIP trades share an OBSERVABLE entry-time signature?
  - if a feature separates SKIP from KEEP in IS *and* the gate holds OOS + plateaus -> the
    headroom is partly CAPTURABLE (candidate gate).
  - if nothing separates out-of-sample -> the headroom is UNPREDICTABLE from price = human
    judgment / luck, and we stop. (a clean null is a real answer.)

Discipline: features are all daily + shift(1) = no lookahead. The oracle is used ONLY as the
SKIP/KEEP label (the target), never as a gate input. Feature selection is IS-only; any winner
is judged on OOS + plateau, and must then generalize to BTC (separate follow-up).

  .venv/bin/python research/regime_headroom.py
"""
import os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from research.regime_gate_lab import CFG, SPLIT, er, metrics, at

M = 20   # oracle forward-regime horizon (trading days), matches regime_ceiling


def daily_features(d):
    dc = d["close"].resample("1D").last().dropna()
    dh = d["high"].resample("1D").max().reindex(dc.index)
    dl = d["low"].resample("1D").min().reindex(dc.index)
    a14 = dc.diff().abs().rolling(14).mean()                       # daily ATR proxy
    s50, s150 = dc.rolling(50).mean(), dc.rolling(150).mean()
    f = {
        "slope150": (s150 - s150.shift(10)) / (10 * a14),          # trend steepness
        "slope50":  (s50 - s50.shift(10)) / (10 * a14),
        "ER20":     er(dc, 20),                                    # trend efficiency
        "dist150":  (dc - s150) / a14,                             # extension above MA (ATR)
        "volrel":   a14 / a14.rolling(100).mean(),                 # vol regime (>1 = elevated)
        "range10":  (dh.rolling(10).max() - dl.rolling(10).min()) / a14,  # expansion/consolidation
    }
    adx = ta.adx(dh, dl, dc, length=14)
    f["adx14"] = adx["ADX_14"] if adx is not None and "ADX_14" in adx else pd.Series(np.nan, index=dc.index)
    oracle_keep = dc.shift(-M) > dc                                # LABEL ONLY (lookahead)
    return f, oracle_keep


def grpstat(g):
    return f"n={len(g):>4}  meanR={g.R.mean():+.2f}  totR={g.R.sum():+5.0f}  win={(g.R>0).mean()*100:>3.0f}%"


def main():
    csv = "data/vantage_xauusd_h1.csv"
    d = resample(load_mt5_csv(csv), "1h")
    t = run(d, SimpleNamespace(**{**CFG, "csv": csv, "tf": "1h"})).sort_values("time").reset_index(drop=True)

    feats, oracle_keep = daily_features(d)
    t["keep"] = at(oracle_keep, t.time)
    for name, s in feats.items():
        t[name] = s.shift(1).reindex(t.time, method="ffill").values   # shift(1) = no lookahead

    print(f"\n=== regime_headroom :: gold 1H breakout (entries FIXED)  refs: mech 0.69 / oracle 1.54 ===")
    print(f"  always-on: {grpstat(t)}\n")
    print(f"  -- the headroom: oracle KEEP vs SKIP (fwd {M}d regime) --")
    print(f"    KEEP : {grpstat(t[t.keep])}")
    print(f"    SKIP : {grpstat(t[~t.keep])}   <- the trades a perfect timer avoids (the prize)")

    # ---- feature separation, IS ONLY ----
    is_t = t[t.time.dt.year < SPLIT].copy()
    print(f"\n  -- feature separation on IS ({len(is_t)} trades, <{SPLIT}): quartile meanR | SKIP% --")
    rank = []
    for name in list(feats):
        sub = is_t[[name, "R", "keep"]].dropna()
        if len(sub) < 20:
            continue
        q = pd.qcut(sub[name], 4, labels=False, duplicates="drop")
        mr = sub.groupby(q).R.mean()
        sk = sub.groupby(q).apply(lambda g: (~g.keep).mean() * 100)
        spread = mr.iloc[-1] - mr.iloc[0]
        rank.append((abs(spread), name, mr, sk, spread))
    rank.sort(reverse=True)
    for _, name, mr, sk, spread in rank:
        mrs = " ".join(f"{v:+.2f}" for v in mr.values)
        sks = " ".join(f"{v:3.0f}" for v in sk.values)
        print(f"    {name:<9} meanR[Q1..Q4] {mrs}  | SKIP% {sks}  | spread {spread:+.2f}")

    # ---- top feature -> candidate gate, OOS + plateau ----
    if not rank:
        print("\n  no rankable features"); return
    _, top, mr, _, spread = rank[0]
    hi_good = spread > 0                                  # favourable = high side if Q4>Q1
    print(f"\n  -- candidate gate on TOP feature '{top}' ({'high' if hi_good else 'low'} = favourable) --")
    print(f"    {'thr(IS pct)':<12} {'gate result (full sample)':<55} keep%")
    for pct in (0.3, 0.4, 0.5, 0.6, 0.7):
        thr = is_t[top].quantile(pct)
        gated = t[t[top] >= thr] if hi_good else t[t[top] <= thr]
        m = metrics(gated)
        keeprate = len(gated) / len(t) * 100
        if m is None:
            print(f"    p{pct*100:.0f}={thr:+.2f}  (too few)"); continue
        print(f"    p{int(pct*100)}={thr:+6.2f}  n={m['n']:>4} CAGR={m['cagr']:+5.1f}% DD={m['dd']:4.1f}% "
              f"CAGR/DD={m['cdd']:5.2f} IS={m['isr']:+.2f} OOS={m['oos']:+.2f} yr+{m['py']:>4}  {keeprate:3.0f}%")
    print("\n  read: a real edge = the gate beats mech 0.69 across the threshold sweep (plateau) with OOS>=IS-ish,")
    print("        not a lone spike. then it MUST generalize to BTC before believing (steep-slope did not).")


if __name__ == "__main__":
    main()
