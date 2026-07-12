"""B2: ATR_regime>1.2 is the real winner. Audit it as flagship; is it redundant with
KAMA or complementary? IS/OOS + per-year (regime-concentration?) + gate overlap + combo."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import research.edge_harness as EH
from research.edge_harness import audit
from research.overfit_audit import cdd_R
from research.regime_discriminator import _trades, f_adx, f_atr_reg, f_kama_d

df, idx, R, times = _trades("BTC", EH.demo_breakout, "4h", 2.0, 1.0, "rr", None, 0.5)
yrs = (pd.Timestamp(times[-1]) - pd.Timestamp(times[0])).days / 365.25
yr = np.array([t.year for t in times])
def feat(fn): return np.asarray(fn(df), float)[idx]
adx, atr, kama = feat(f_adx), feat(f_atr_reg), feat(f_kama_d)

def gm(vals, thr): return (vals >= thr) & ~np.isnan(vals)
def lst(m): return list(zip(times[m], R[m]))
def isoos(m):
    med = np.median(np.unique(yr)); Rm, ym = R[m], yr[m]
    IS = Rm[ym < med].mean(); OOS = Rm[ym >= med].mean(); return IS, OOS

mA = gm(atr, 1.2); mK = gm(kama, 0.0); mD = gm(adx, 28); mBoth = mA & mK; mEither = mA | mK

print("=== gate comparison (BTC breakout @4h base) ===")
print(f"  {'gate':<18}{'N':>5}{'meanR':>8}{'CAGR/DD':>9}{'IS/OOS':>16}{'green':>7}")
for nm, m in [("ATRreg>1.2", mA), ("KAMAd_rising", mK), ("ADX>28", mD),
              ("ATR&KAMA", mBoth), ("ATR|KAMA", mEither)]:
    Rm, ym = R[m], yr[m]; uy = np.unique(ym); g = sum(1 for y in uy if Rm[ym == y].sum() > 0)
    IS, OOS = isoos(m)
    print(f"  {nm:<18}{len(Rm):>5}{Rm.mean():>+8.3f}{cdd_R(Rm, yrs)[2]:>+9.2f}{f'{IS:+.2f}/{OOS:+.2f}':>16}{f'{g}/{len(uy)}':>7}")

ov = (mA & mK).sum() / mA.sum(); print(f"\n  overlap: {ov:.0%} of ATRreg>1.2 trades are also KAMA-rising")

print("\n=== per-year meanR (ATRreg>1.2) — regime-concentration check ===")
for y in np.unique(yr):
    m = mA & (yr == y);
    if m.sum() >= 3: print(f"  {y}: n={m.sum():>3}  meanR={R[m].mean():+.3f}  totR={R[m].sum():+.1f}")

configs = {"always_on": lst(np.ones(len(R), bool))}
for t in (1.0, 1.1, 1.2, 1.3, 1.4): configs[f"ATRreg>{t}"] = lst(gm(atr, t))
configs["KAMAd_rising"] = lst(mK)
for t in (24, 28, 32): configs[f"ADX>{t}"] = lst(gm(adx, t))
print("\n### AUDIT flagship = ATRreg>1.2 ###")
audit(configs, flagship="ATRreg>1.2")
