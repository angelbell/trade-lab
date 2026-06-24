"""Free-screen triple set (zero new strategy code) for three fresh ideas.

Each test is a near-instant data screen meant to KILL an idea cheaply, before
any strategy is built. Pre-registered decision rules are printed inline.

  Screen 1  Cross-asset lead-lag (Idea 4): is leader[t-1] -> lagger[t] corr
            distinguishable from 0 on the SAME Vantage feed/clock?
            KILL rule: |lag-1 corr| < 0.05 everywhere -> no exploitable lag.

  Screen 2  USDX-as-gold-gate redundancy (Idea 2): how much does USDX *already*
            live inside gold's own price? Measured by contemporaneous corr.
            KILL rule: |contemp corr| high (say >0.5) -> the gate is redundant
            (gold price already encodes the dollar), like the dead macro-DXY gate.

  Screen 3  Weekend-gap behavior on gold (Idea 7): after a weekend gap, does
            gold FADE (fill) or CONTINUE? And does BTC's weekend move (the only
            instrument that priced the weekend) predict gold's Monday direction?
            KILL rule: no monotone, year-stable fade/continue tendency that
            clears a realistic spread.

Run:  .venv/bin/python research/free_screens.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_mt5_csv  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
GOLD = DATA / "vantage_xauusd_h1.csv"
BTC = DATA / "vantage_btcusd_h1.csv"
USDX = DATA / "vantage_usdx.r_h1.csv"


def logret(s: pd.Series) -> pd.Series:
    return np.log(s / s.shift(1))


def load_close(path: Path) -> pd.Series:
    return load_mt5_csv(path)["close"]


# ----------------------------------------------------------------------------
def screen1_leadlag():
    print("=" * 74)
    print("SCREEN 1 — Cross-asset lead-lag (Idea 4)")
    print("  KILL if |lag-1 corr| < 0.05 everywhere. Same feed/clock, inner-join.")
    print("=" * 74)

    g, b, u = load_close(GOLD), load_close(BTC), load_close(USDX)
    series = {"GOLD": g, "BTC": b, "USDX": u}

    for tf_name, rule in [("H1", "1h"), ("Daily", "1D")]:
        rets = {k: logret(v.resample(rule).last()) for k, v in series.items()}
        df = pd.DataFrame(rets).dropna()
        print(f"\n  [{tf_name}]  overlapping bars n={len(df)}  "
              f"({df.index.min().date()} -> {df.index.max().date()})")
        names = list(df.columns)
        print(f"    {'leader[t-1] -> lagger[t]':28s}  corr      (contemp for ref)")
        for L in names:
            for M in names:
                if L == M:
                    continue
                lag1 = df[L].shift(1).corr(df[M])
                contemp = df[L].corr(df[M])
                flag = "  <-- !" if abs(lag1) >= 0.05 else ""
                print(f"    {L:>5s}[t-1] -> {M:<5s}[t]        "
                      f"{lag1:+.3f}     ({contemp:+.3f}){flag}")


# ----------------------------------------------------------------------------
def screen2_usdx_redundancy():
    print("\n" + "=" * 74)
    print("SCREEN 2 — USDX-as-gold-gate redundancy (Idea 2)")
    print("  KILL if |contemp corr| high -> gold price already encodes the dollar.")
    print("=" * 74)

    g, u = load_close(GOLD), load_close(USDX)
    for tf_name, rule in [("H1", "1h"), ("Daily", "1D")]:
        gr, ur = logret(g.resample(rule).last()), logret(u.resample(rule).last())
        df = pd.DataFrame({"gold": gr, "usdx": ur}).dropna()
        c = df["gold"].corr(df["usdx"])
        print(f"\n  [{tf_name}]  n={len(df)}  contemp corr(gold, usdx) = {c:+.3f}")
        # Year-by-year stability of the inverse relationship.
        by = df.groupby(df.index.year).apply(
            lambda x: x["gold"].corr(x["usdx"]))
        print("    per-year:", "  ".join(f"{y}:{v:+.2f}" for y, v in by.items()))
        verdict = "REDUNDANT (info already in gold price)" if abs(c) > 0.5 \
            else "possible orthogonal info — worth a gate test"
        print(f"    -> {verdict}")


# ----------------------------------------------------------------------------
def screen3_weekend_gap():
    print("\n" + "=" * 74)
    print("SCREEN 3 — Weekend-gap behavior on gold (Idea 7)")
    print("  Fade vs continue after weekend gap; + BTC weekend as predictor.")
    print("=" * 74)

    g = load_mt5_csv(GOLD)
    b = load_close(BTC)

    # Detect session gaps: consecutive bars with a time jump > 6h (weekend/holiday).
    idx = g.index
    dt_h = idx.to_series().diff().dt.total_seconds() / 3600.0
    gap_mask = dt_h > 6.0
    gap_pos = np.where(gap_mask.values)[0]
    gap_pos = gap_pos[gap_pos > 0]

    rows = []
    closes = g["close"].values
    opens = g["open"].values
    for p in gap_pos:
        prev_close = closes[p - 1]
        gap_open = opens[p]
        gap_pct = (gap_open - prev_close) / prev_close
        # forward return: gap-open -> close 24 bars later (~1 trading day on H1).
        fwd_end = min(p + 24, len(closes) - 1)
        fwd = (closes[fwd_end] - gap_open) / gap_open
        # BTC weekend move over the same wall-clock gap window.
        t_prev, t_gap = idx[p - 1], idx[p]
        bw = np.nan
        try:
            b_prev = b.asof(t_prev)
            b_gap = b.asof(t_gap)
            if pd.notna(b_prev) and pd.notna(b_gap) and b_prev > 0:
                bw = (b_gap - b_prev) / b_prev
        except Exception:
            pass
        rows.append((idx[p], gap_pct, fwd, bw))

    gp = pd.DataFrame(rows, columns=["t", "gap", "fwd", "btc_wknd"]).set_index("t")
    gp = gp.dropna(subset=["gap", "fwd"])
    print(f"\n  weekend/holiday gaps detected: n={len(gp)}  "
          f"median |gap|={gp['gap'].abs().median()*100:.3f}%")

    # Fade vs continue: sign agreement of gap and forward move.
    cont = np.sign(gp["gap"]) == np.sign(gp["fwd"])
    print(f"  continuation rate (gap & fwd same sign) = {cont.mean()*100:.1f}%  "
          f"(50% = coin flip)")
    print(f"  mean fwd | up-gap   = {gp.loc[gp.gap>0,'fwd'].mean()*100:+.3f}%  "
          f"(n={int((gp.gap>0).sum())})")
    print(f"  mean fwd | down-gap = {gp.loc[gp.gap<0,'fwd'].mean()*100:+.3f}%  "
          f"(n={int((gp.gap<0).sum())})")

    # Bigger gaps: does the tendency strengthen (monotone)?
    big = gp[gp["gap"].abs() > gp["gap"].abs().median()]
    cont_big = np.sign(big["gap"]) == np.sign(big["fwd"])
    print(f"  continuation rate on LARGER-than-median gaps = {cont_big.mean()*100:.1f}%")

    # Per-year stability of continuation rate.
    by = gp.groupby(gp.index.year).apply(
        lambda x: (np.sign(x["gap"]) == np.sign(x["fwd"])).mean())
    print("  per-year continuation rate:",
          "  ".join(f"{y}:{v*100:.0f}%" for y, v in by.items()))

    # BTC weekend as predictor of gold gap direction / forward move.
    sub = gp.dropna(subset=["btc_wknd"])
    if len(sub) > 20:
        c_dir = sub["btc_wknd"].corr(sub["gap"])
        c_fwd = sub["btc_wknd"].corr(sub["fwd"])
        print(f"\n  BTC weekend move vs gold gap    corr = {c_dir:+.3f}  (n={len(sub)})")
        print(f"  BTC weekend move vs gold fwd    corr = {c_fwd:+.3f}")


if __name__ == "__main__":
    screen1_leadlag()
    screen2_usdx_redundancy()
    screen3_weekend_gap()
    print("\n" + "=" * 74)
    print("Reminder: these are KILL-screens. A pass here = 'not dead yet', NOT an edge.")
    print("=" * 74)
