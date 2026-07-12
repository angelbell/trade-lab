"""What N (trades/yr), PF, and risk-fraction f give >=50% prob of 2x in a year?
Fixed-fractional compounding: each trade risks fraction f; R-multiple r changes
equity by (1+f*r). Median-doubling (the 50% line) requires N*g >= ln2 where
g = E[ln(1+f*r)] (variance-free; variance only sets the spread, reported too)."""
import numpy as np
LN2 = np.log(2.0)

def p_from_pf(pf, b):           # PF = p*b/((1-p)*1) -> p
    return pf / (pf + b)

def g_per_trade(p, b, f):       # geometric log-growth per trade
    return p * np.log(1 + f * b) + (1 - p) * np.log(1 - f)

def need_N(pf, b, f):
    p = p_from_pf(pf, b); g = g_per_trade(p, b, f)
    return (LN2 / g) if g > 0 else np.inf, p

for b, label in [(1.0, "RR 1:1  (matches the bounce: tgt=stop)"),
                 (2.0, "RR 2:1")]:
    print(f"\n===== {label} — trades/year needed for MEDIAN 2x =====")
    print("  PF (winrate)      f=1%   f=2%   f=3%   f=5%   f=10%")
    for pf in (1.3, 1.5, 1.8, 2.0, 2.5, 3.0):
        p = p_from_pf(pf, b)
        cells = []
        for f in (0.01, 0.02, 0.03, 0.05, 0.10):
            N, _ = need_N(pf, b, f)
            cells.append(f"{N:5.0f}" if np.isfinite(N) else "  inf")
        print(f"  {pf:>3.1f} ({p*100:>4.1f}% win)   " + "  ".join(cells))

# --- full distribution for a few realistic (N, PF, f) targets (RR 1:1) ---
def mc(pf, b, f, N, trials=200000, seed=0):
    rng = np.random.default_rng(seed)
    p = p_from_pf(pf, b)
    wins = rng.random((trials, N)) < p
    r = np.where(wins, b, -1.0)
    mult = np.prod(1 + f * r, axis=1)
    return mult

print("\n\n===== distribution of the 1-year multiple (RR 1:1, Monte Carlo) =====")
print(f"  {'config':<34} P(>=2x) P(>=5x) P(<=0.5x)  median  p25   p75")
for pf, f, N in [(2.0, 0.02, 107), (2.0, 0.05, 45), (2.0, 0.10, 24),
                 (2.5, 0.05, 34), (2.5, 0.10, 18),
                 (3.0, 0.10, 15), (1.5, 0.05, 79), (1.5, 0.10, 46)]:
    m = mc(pf, 1.0, f, N)
    p = p_from_pf(pf, 1.0)
    tag = f"PF{pf} ({p*100:.0f}%w) f={int(f*100)}% N={N}"
    print(f"  {tag:<34} {(m>=2).mean()*100:5.1f}% {(m>=5).mean()*100:5.1f}% "
          f"{(m<=0.5).mean()*100:6.1f}%   {np.median(m):5.2f} {np.percentile(m,25):4.2f} {np.percentile(m,75):5.2f}")
