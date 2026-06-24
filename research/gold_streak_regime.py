"""gold_streak_regime.py -- the user's correction: the up/down continuation asymmetry (up-runs continue,
down-runs bounce) is BULL BETA, not a streak effect. A naive full-history test conflates the two and would
unfairly judge 'short the persistent decline' as dead. Remove the confound: measure streak-continuation
WITHIN the prevailing trend regime (1H EMA80, H17's gate), so 'down-streak in an actual DOWNTREND' is tested
apart from gold's secular up-drift.

The fair question: once we are ALREADY in a downtrend (regime-conditioned), does a persistent N-bar decline
CONTINUE (user's intuition holds intra-downtrend) or still bounce? If down-runs continue only in downtrends
AND up-runs continue only in uptrends by the same amount => the 'continuation' is pure trend-alignment (beta),
streak count adds nothing. If the regime-matched streak forward move clears cost AND beats the regime's plain
drift => the streak carries marginal info. Regime known at entry (1H EMA80 shift(1)+ffill = no lookahead).

Honesty: gross, cost ~$1.3-2.1 RT noted; descriptive. compares streak-conditional fwd vs the SAME-regime
unconditional drift (the beta baseline) -- the marginal streak effect is the difference. In-sample.
  .venv/bin/python research/gold_streak_regime.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from research.volume_reversal_screen import resample
from research.gold_streak_continuation import atr, run_lengths


def regime_1h(d):
    """up-regime = 1H close > 1H EMA80 (H17 gate), aligned to M15, no lookahead."""
    c1 = d["close"].resample("1h", label="left", closed="left").last().dropna()
    ema = c1.ewm(span=80, adjust=False).mean()
    up = (c1 > ema).shift(1).reindex(d.index, method="ffill").fillna(False).values
    return up


def screen(d, H):
    c = d["close"].values
    a = atr(d).values
    rl = run_lengths(c)
    up_reg = regime_1h(d)
    med = np.nanmedian(a)
    n = len(c)
    valid = np.zeros(n, bool); valid[14:n - H] = True
    valid &= np.isfinite(a) & (a > 0)
    fwd_all = np.full(n, np.nan); fwd_all[:n - H] = c[H:] - c[:n - H]
    print(f"\n== forward {H} bars ({H*15}min); regime = 1H price vs EMA80 (ATR med ${med:.1f}) ==")
    print(f"  {'streak':>10} {'regime':>9} {'n':>6} {'driftFwd$':>10} {'streakFwd$':>11} {'marginal$':>10} {'cont%':>7}")
    for side, sgn in (("down", -1), ("up", +1)):
        for L in (3, 4, 5):
            for rname, rmask in (("downtrend", ~up_reg), ("uptrend", up_reg)):
                if side == "down":
                    sm = (rl <= -L)
                else:
                    sm = (rl >= L)
                base = valid & rmask                          # all bars in this regime (the drift baseline)
                cond = base & sm                              # + the streak condition
                if cond.sum() < 80:
                    continue
                drift = sgn * np.nanmean(fwd_all[base])       # regime's plain drift, signed to side
                sf = sgn * np.nanmean(fwd_all[cond])          # streak-conditional fwd, signed to side
                contp = (sgn * fwd_all[cond] > 0).mean() * 100
                marg = sf - drift                            # marginal streak effect (beta removed)
                tag = "+streak" if marg > 0.1 else ("-streak" if marg < -0.1 else "~beta")
                print(f"  {side+str(L)+'+':>10} {rname:>9} {cond.sum():>6} {drift:>+10.2f} {sf:>+11.2f} "
                      f"{marg:>+10.2f} {contp:>6.1f}%  {tag}")


def main():
    full = resample(load_mt5_csv("data/vantage_xauusd_m5.csv"), "15min")
    for era, lo in [("2019+", 2019), ("2024+ (current vol)", 2024)]:
        d = full[full.index.year >= lo]
        print(f"\n############ GOLD M15 {era}  n={len(d)} ############")
        for H in (4, 8):
            screen(d, H)
    print("\n  read: 'driftFwd' = the regime's plain trend drift (the BETA baseline). 'streakFwd' = forward")
    print("  after the N-bar run. 'marginal' = streakFwd - drift = what the STREAK adds beyond the trend.")
    print("  marginal ~0 => the run carries NO info beyond regime (continuation = pure beta, user's caveat).")
    print("  down-streak in a DOWNtrend: streakFwd>0 (cont%>50) would be the user's 'decline keeps going'.")


if __name__ == "__main__":
    main()
