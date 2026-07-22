"""DO TREND LEGS AGE?  (generalisation of the user's "BTC falls for ~10 weekly bars")

The question, stated so it can be killed:
    Is an OLD trend leg more likely to end in the next bar than a YOUNG one?
    (aging / rising hazard)   vs   (memoryless: every bar has the same reversal odds)
If legs are memoryless everywhere, then every "this trend is getting long in the tooth" rule --
time stops, cycle-age gates, size-down-with-age, "take profit, it has run far enough" -- is dead
by construction, and the lab's "fixed far target beats cutting winners" law gets its mechanism.

WHY THE OBVIOUS TEST IS WRONG
  A ZigZag leg cannot end until price retraces k*ATR, so short legs are impossible BY CONSTRUCTION
  -> the raw hazard always rises at young ages -> fake "aging". A geometric/exponential null does
  not see this bias.
THE NULL WE USE INSTEAD
  Run THE SAME DETECTOR on synthetic series built by BLOCK-SHUFFLING the bars (each bar keeps its
  own return and its high/low geometry; blocks of 20 keep volatility clustering). This destroys
  trend memory but reproduces the detector's bias exactly. Anything left over is real.

STATISTIC
  Weibull shape k of the leg-duration distribution (MLE, loc=0):
      k > 1 = aging (hazard rises with age)   k = 1 = memoryless   k < 1 = the opposite.
  Reported as k_obs vs the null distribution of k from the shuffled series -> p = P(k_null >= k_obs).

CELLS  gold/BTC {1h, 4h, 1d} x {up legs, down legs}, 6 FX pairs {4h, 1d} x {up, down}.
Run: .venv/bin/python experiments/trend_leg_aging.py
"""
import sys, os, io, contextlib, warnings
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import weibull_min
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag

ROOT = "/home/angelbell/dev/auto-trade"
NNULL = 150
BLOCK = 20
ZZK = 2.0
RNG = np.random.default_rng(20260713)

CELLS = [("gold", "data/vantage_xauusd_h1.csv", ["1h", "4h", "1d"]),
         ("BTC", "data/vantage_btcusd_h1.csv", ["1h", "4h", "1d"]),
         ("EURUSD", "data/vantage_eurusd_h4.csv", ["4h", "1d"]),
         ("GBPUSD", "data/vantage_gbpusd_h4.csv", ["4h", "1d"]),
         ("AUDUSD", "data/vantage_audusd_h4.csv", ["4h", "1d"]),
         ("NZDUSD", "data/vantage_nzdusd_h4.csv", ["4h", "1d"]),
         ("USDCAD", "data/vantage_usdcad_h4.csv", ["4h", "1d"]),
         ("USDJPY", "data/vantage_usdjpy_h4.csv", ["4h", "1d"])]


def atr(h, l, c, n=14):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n).mean().values


def legs_from(h, l, c):
    a = atr(h, l, c)
    sw = swings_zigzag(h, l, a, ZZK)
    P = [(p[1], p[3]) for p in sw]                  # (pivot bar index, kind: +1 high / -1 low)
    up, dn = [], []
    for (b0, k0), (b1, k1) in zip(P[:-1], P[1:]):
        d = b1 - b0
        if d <= 0: continue
        (up if k1 == +1 else dn).append(d)          # leg ENDING at a high = an up leg
    return np.array(up), np.array(dn)


def shape_k(L):
    if len(L) < 12: return np.nan
    try:
        k, _, _ = weibull_min.fit(L.astype(float), floc=0)
        return k
    except Exception:
        return np.nan


def block_shuffle(h, l, c):
    """Rebuild a synthetic series: each bar keeps its own return + its high/low geometry;
    blocks of BLOCK bars are resampled -> volatility clustering survives, trend memory does not."""
    lc = np.log(c)
    r = np.diff(lc, prepend=lc[0])
    hi = np.log(h / c)                              # >= 0
    lo = np.log(l / c)                              # <= 0
    n = len(c)
    nb = int(np.ceil(n / BLOCK))
    starts = RNG.integers(0, max(1, n - BLOCK), nb)
    idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
    r2, hi2, lo2 = r[idx], hi[idx], lo[idx]
    c2 = np.exp(np.log(c[0]) + np.cumsum(r2))
    return c2 * np.exp(hi2), c2 * np.exp(lo2), c2


def resample(df, tf):
    if tf == "1h": return df
    rule = {"4h": "4h", "1d": "1D"}[tf]
    o = df.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    return o.dropna()


def main():
    print(f"ZigZag {ZZK}xATR legs; Weibull shape k  (k>1 = AGING, k=1 = memoryless, k<1 = the reverse)")
    print(f"null = the SAME detector on {NNULL} block-shuffled series (block={BLOCK} bars): this "
          f"reproduces the detector's minimum-leg bias, so only real memory survives\n")
    print(f"{'cell':<14}{'dir':<6}{'legs':>6}{'med':>6}{'sd':>7}{'k_obs':>8}"
          f"{'k_null (med)':>14}{'null 95th':>11}{'p':>7}  verdict")
    rows = []
    for name, path, tfs in CELLS:
        with contextlib.redirect_stderr(io.StringIO()):
            base = load_mt5_csv(os.path.join(ROOT, path))
        for tf in tfs:
            d = resample(base, tf)
            if len(d) < 500: continue
            h, l, c = d["high"].values, d["low"].values, d["close"].values
            up, dn = legs_from(h, l, c)
            nulls_up, nulls_dn = [], []
            for _ in range(NNULL):
                h2, l2, c2 = block_shuffle(h, l, c)
                u2, d2 = legs_from(h2, l2, c2)
                nulls_up.append(shape_k(u2)); nulls_dn.append(shape_k(d2))
            for dirn, L, nulls in (("UP", up, np.array(nulls_up)), ("DOWN", dn, np.array(nulls_dn))):
                ko = shape_k(L)
                nn = nulls[np.isfinite(nulls)]
                if not np.isfinite(ko) or len(nn) < 30:
                    continue
                p = float((nn >= ko).mean())
                v = "AGING" if p < 0.05 else ("younger-dies" if p > 0.95 else "memoryless")
                print(f"{name+' '+tf:<14}{dirn:<6}{len(L):>6}{np.median(L):>6.0f}{L.std():>7.1f}"
                      f"{ko:>8.2f}{np.median(nn):>14.2f}{np.percentile(nn,95):>11.2f}{p:>7.3f}  {v}")
                rows.append(dict(cell=f"{name} {tf}", dir=dirn, n=len(L), k=ko,
                                 knull=np.median(nn), p=p, verdict=v))
    R = pd.DataFrame(rows)
    print(f"\nSUMMARY over {len(R)} cells:")
    print(f"  AGING (p<0.05)      : {(R.p < 0.05).sum():>3}   (chance would give ~{0.05*len(R):.0f})")
    print(f"  memoryless          : {((R.p >= 0.05) & (R.p <= 0.95)).sum():>3}")
    print(f"  younger-dies (p>.95): {(R.p > 0.95).sum():>3}")
    print(f"  k_obs vs k_null: mean {R.k.mean():.2f} vs {R.knull.mean():.2f}  "
          f"(median diff {np.median(R.k - R.knull):+.3f})")
    print(f"\n  by direction:  UP  k-knull median {np.median(R[R['dir']=='UP'].k - R[R['dir']=='UP'].knull):+.3f}"
          f"   DOWN {np.median(R[R['dir']=='DOWN'].k - R[R['dir']=='DOWN'].knull):+.3f}")
    R.to_csv(os.path.join(ROOT, "experiments/out_trend_leg_aging.csv"), index=False)


if __name__ == "__main__":
    main()
