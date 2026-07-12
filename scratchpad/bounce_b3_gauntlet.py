"""bounce_b3_gauntlet.py -- B3 (gold 15m deep-Fib bounce, revived at real cost $0.3) decisive tests
derived from its remaining kill-risks + first improvement lever:

  1. BOUNDARY: extend the frac sweep past the old 0.85 edge (0.70..0.93) -- plateau or spike?
  2. REGIME: per-year totR + rerun on TRUE-M5-density span only (2018-09+) -- 2025-concentration?
     (the m5 file's pre-2018 rows are sparse (H1/daily); the full-file numbers count them)
  3. REDUNDANCY: annual-R correlation vs the gold_bo 15m canon leg (same span)
  4. IMPROVEMENT (death-reverse-engineered): stop buffer below the obvious L0 (buf x ATR@setup)
     -- L0 is where everyone's stop sits; does escaping the sweep pay more than the RR it costs?
All net $0.3 rt + stop-slip 0.5x overshoot (same model as the revival run).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag, run as run_bo, resample as resample_bo
from radar_gate_race import BASE

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}
W, FWD = 40, 300
SP, SS = 0.3, 0.5


def build(df):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    a = ta.atr(df["high"], df["low"], df["close"], 14).values
    es = df["close"].ewm(span=80, adjust=False).mean().values
    dc = df["close"].resample("1D").last().dropna()
    sma = dc.rolling(150).mean()
    g = ((dc > sma) & (sma > sma.shift(10))).shift(1).reindex(df.index, method="ffill").fillna(False).values
    sw = swings_zigzag(h, l, a, 2.0)
    imps = []
    for t in range(1, len(sw)):
        cH, iH, pH, kH = sw[t]
        cL, iL, pL, kL = sw[t - 1]
        if kH != +1 or kL != -1 or pH - pL <= 0:
            continue
        if not np.isnan(es[cH]) and pH < es[cH]:
            continue
        imps.append((cH, pL, pH))
    return dict(df=df, h=h, l=l, c=c, a=a, g=g, imps=imps)


def one(B, frac, buf=0.0, sp=SP, ss=SS):
    h, l, c, a, g, df = B["h"], B["l"], B["c"], B["a"], B["g"], B["df"]
    busy = -1
    out = []
    for (cH, L0, H1) in B["imps"]:
        if cH <= busy:
            continue
        lim = H1 - frac * (H1 - L0)
        stop = L0 - buf * a[cH]
        if lim <= stop or not g[min(cH, len(g) - 1)]:
            continue
        fill_j = None
        for j in range(cH + 1, min(cH + 1 + W, len(c))):
            if l[j] <= stop:
                break
            if l[j] <= lim:
                fill_j = j
                break
        if fill_j is None:
            continue
        entry, risk, reward = lim, lim - stop, H1 - lim
        if risk <= 0 or reward <= 0:
            continue
        exit_j = min(fill_j + FWD, len(c) - 1)
        r = None
        for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
            if l[j] <= stop:
                r = -1.0 - ss * (stop - l[j]) / risk
                exit_j = j
                break
            if h[j] >= H1:
                r = reward / risk
                exit_j = j
                break
        if r is None:
            r = (c[exit_j] - entry) / risk
        out.append((df.index[fill_j], df.index[exit_j], r - sp / risk))
        busy = exit_j
    return out


def card(tag, tr, span):
    if len(tr) < 12:
        print(f"  {tag:>12}: n={len(tr)} too few")
        return
    R = np.array([r for _, _, r in tr])
    ts = [t for t, _, _ in tr]
    yr = np.array([t.year for t in ts])
    yrs = np.unique(yr)
    half = yrs[len(yrs) // 2]
    pf = R[R > 0].sum() / abs(R[R <= 0].sum())
    cum = np.cumsum(R)
    dd = (np.maximum.accumulate(cum) - cum).max()
    hold = np.median([(b - a).total_seconds() / 3600 for a, b in
                      [(t0, t1) for t0, t1, _ in tr]])
    g = sum(R[yr == y].sum() > 0 for y in yrs)
    print(f"  {tag:>12}: N={len(R):4d} N/yr={len(R)/span:5.1f} win={(R>0).mean()*100:4.1f}% "
          f"PF={pf:4.2f} meanR={R.mean():+.3f} IS/OOS={R[yr<half].mean():+.2f}/{R[yr>=half].mean():+.2f} "
          f"totR/yr={R.sum()/span:+5.1f} maxDD={dd:5.1f}R ret/DD={R.sum()/dd:5.2f} grn={g}/{len(yrs)} "
          f"holdMed={hold:.1f}h")


def main():
    gold = load_mt5_csv("data/vantage_xauusd_m5.csv")
    for span_tag, dfm in [("FULL-file", gold), ("true-M5 2018-09+", gold.loc["2018-09-14":])]:
        df = dfm.resample("15min").agg(AGG).dropna()
        span = (df.index[-1] - df.index[0]).days / 365.25
        B = build(df)
        print(f"\n===== {span_tag} ({span:.1f}yr, impulses={len(B['imps'])}) net $0.3 + slip0.5 =====")
        for frac in (0.70, 0.75, 0.786, 0.80, 0.85, 0.88, 0.90, 0.93):
            card(f"frac{frac}", one(B, frac), span)

    # per-year + stop-buffer lever + correlation on the true-M5 span
    df = gold.loc["2018-09-14":].resample("15min").agg(AGG).dropna()
    span = (df.index[-1] - df.index[0]).days / 365.25
    B = build(df)
    print("\n--- per-year totR (true-M5 span) ---")
    for frac in (0.786, 0.85, 0.90):
        tr = one(B, frac)
        R = np.array([r for _, _, r in tr])
        yr = np.array([t.year for t, _, _ in tr])
        print(f"  frac{frac}: " + "  ".join(f"{y}:{R[yr==y].sum():+.0f}" for y in np.unique(yr)))

    print("\n--- IMPROVEMENT: stop buffer below L0 (buf x ATR@setup) ---")
    for frac in (0.786, 0.85):
        for buf in (0.0, 0.15, 0.30):
            card(f"f{frac}/b{buf}", one(B, frac, buf=buf), span)

    print("\n--- REDUNDANCY: annual-R corr vs gold_bo 15m canon (same span, $0.3) ---")
    t_bo = run_bo(df, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                         "ext_cap": 8.0, "pullback_frac": 0.25}))
    Rbo = t_bo["R"].values - 0.3 / t_bo["risk"].values
    s_bo = pd.Series(Rbo, index=t_bo["time"]).groupby(lambda i: i.year).sum()
    for frac in (0.786, 0.85):
        tr = one(B, frac)
        s_bn = pd.Series([r for _, _, r in tr],
                         index=pd.DatetimeIndex([t for t, _, _ in tr])).groupby(lambda i: i.year).sum()
        idx = sorted(set(s_bo.index) | set(s_bn.index))
        a = s_bo.reindex(idx, fill_value=0.0)
        b = s_bn.reindex(idx, fill_value=0.0)
        # same-day position overlap share
        days_bo = set(pd.DatetimeIndex(t_bo["time"]).date)
        days_bn = set(t.date() for t, _, _ in tr)
        ov = len(days_bo & days_bn) / max(len(days_bn), 1)
        print(f"  frac{frac}: annual corr={np.corrcoef(a, b)[0,1]:+.2f}  "
              f"(bounce entry-days overlapping gold_bo days: {ov*100:.0f}%)")


if __name__ == "__main__":
    main()
