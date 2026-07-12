"""B: does the trend-presence gate (ADX/ATR) survive the overfit gauntlet vs KAMA?

Screen-first: on a COMMON BTC breakout base (demo Donchian proxy @4h), build a FAMILY
of gate configs (ADX/ATR/KAMA/ER x thresholds) and run edge_harness.audit ->
DSR (trial-count haircut), PBO/CSCV (is the ADX-winner just config-selection noise?),
bootCI + null-p. Plateau + per-config CAGR/DD/green printed too.
If ADX fails PBO here it's dead; if it passes, escalate to the REAL btc_bo leg.
"""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import research.edge_harness as EH
from research.edge_harness import audit
from research.overfit_audit import cdd_R
from research.regime_discriminator import _trades, f_adx, f_atr_reg, f_kama_d, f_er

df, idx, R, times = _trades("BTC", EH.demo_breakout, "4h", 2.0, 1.0, "rr", None, 0.5)
yrs = (pd.Timestamp(times[-1]) - pd.Timestamp(times[0])).days / 365.25
yr = np.array([t.year for t in times])
def feat(fn): return np.asarray(fn(df), float)[idx]
adx, atr, kama, er = feat(f_adx), feat(f_atr_reg), feat(f_kama_d), feat(f_er)

def gate(vals, thr):
    m = (vals >= thr) & ~np.isnan(vals)
    return list(zip(times[m], R[m])), m

configs = {"always_on": list(zip(times, R))}
masks = {"always_on": np.ones(len(R), bool)}
for t in (20, 24, 28, 32):      configs[f"ADX>{t}"], masks[f"ADX>{t}"] = gate(adx, t)
for t in (1.0, 1.1, 1.2, 1.3):  configs[f"ATRreg>{t}"], masks[f"ATRreg>{t}"] = gate(atr, t)
configs["KAMAd_rising"], masks["KAMAd_rising"] = gate(kama, 0.0)
for t in (0.25, 0.30, 0.35):    configs[f"ER>{t}"], masks[f"ER>{t}"] = gate(er, t)

print(f"BASE BTC breakout @4h: N={len(R)} {yrs:.1f}yr meanR={R.mean():+.3f} CAGR/DD={cdd_R(R,yrs)[2]:+.2f}")
print(f"\n  {'config':<16}{'N':>5}{'meanR':>8}{'CAGR/DD':>9}{'green/yr':>10}")
for nm, m in masks.items():
    Rm = R[m]; ym = yr[m]
    if len(Rm) < 12: print(f"  {nm:<16}{len(Rm):>5}  (too few)"); continue
    uy = np.unique(ym); green = sum(1 for y in uy if Rm[ym == y].sum() > 0)
    print(f"  {nm:<16}{len(Rm):>5}{Rm.mean():>+8.3f}{cdd_R(Rm, yrs)[2]:>+9.2f}{f'{green}/{len(uy)}':>10}")

print("\n### AUDIT (flagship = the first-screen winner ADX>28; PBO haircuts the 12-config selection) ###")
audit(configs, flagship="ADX>28")
print("\n### AUDIT (flagship = incumbent KAMAd_rising, for comparison) ###")
audit(configs, flagship="KAMAd_rising")
