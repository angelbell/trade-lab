"""line_reaction.py -- does price REACT to a line, and does the reaction rate
depend on (a) how far it just fell into the line and (b) the timeframe?

This is a top-of-funnel SCREEN, not a strategy. It answers the user's hypothesis:
"ラインへの反応率は、直近の下落幅と時間軸に影響される" — the reaction rate off a
support line is conditioned by the recent drop magnitude and the TF.

Definitions (all CAUSAL -- no lookahead; the tag is decided at the close of bar i
using only info known by then; the excursion is measured from i+1 onward):
  * LINE:
      - ema200 : the 200-EMA, read as ema[i-1] (previous bar's value) so it is
                 fully known when bar i closes.
      - swing  : the most-recent CONFIRMED swing low (fractal, confirmed n bars
                 later via src.strategy._confirmed_swings -> already lagged).
  * TAG (support / long bounce): approaching from ABOVE, low[i] <= level + tol*ATR
        while the prior close was above the level. Deduped within `horizon` bars.
  * DROP magnitude (conditioning var #1): (max high over the prior L bars
        - low[i]) / ATR[i]  -- how deep the fall into the line was, in ATR units.
        Known at the close of bar i.
  * DROP speed (conditioning var #2): drop / (bars from that recent high to i)
        -- ATR per bar. A steep capitulation vs a slow grind into the line.
  * REACTION (the rate): first-touch race from i+1 -- does price reach
        e + react*ATR (bounce) before e - react*ATR (fail)?  Same-bar both-touch
        is counted as FAIL (conservative). e = close[i].
  * MFE/MAE: peak favourable / adverse excursion over `fwd` bars / ATR (the screen).

Reports, per TF and split by DROP tercile: N, reaction-rate, MFE/MAE ratio,
median MFE/MAE. Median + std are shown where a distribution is summarised.

  .venv/bin/python research/line_reaction.py --csv data/vantage_xauusd_h1.csv \
      --tf 1h,2h,4h,8h,1d --line ema200
  .venv/bin/python research/line_reaction.py --csv data/vantage_xauusd_m15.csv \
      --tf 15m --line swing
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from src.strategy import _confirmed_swings

_ALIAS = {"15m": "15min", "30m": "30min", "1h": "1h", "2h": "2h", "4h": "4h",
          "8h": "8h", "12h": "12h", "1d": "1D", "1w": "1W"}


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = _ALIAS.get(tf.lower(), tf)
    out = pd.DataFrame({
        "open":  df["open"].resample(rule).first(),
        "high":  df["high"].resample(rule).max(),
        "low":   df["low"].resample(rule).min(),
        "close": df["close"].resample(rule).last(),
    }).dropna()
    return out


def level_series(d: pd.DataFrame, line: str, ema_n: int, frac_n: int) -> pd.Series:
    """Return the CAUSAL support-line level per bar (long-bounce side)."""
    if line == "ema200":
        ema = ta.ema(d["close"], length=ema_n)
        return ema.shift(1)                       # previous bar's EMA = known at i
    elif line == "swing":
        _, _, last_sl, _ = _confirmed_swings(d["high"], d["low"], frac_n)
        return last_sl                            # already confirmed/lagged
    raise ValueError(line)


def daily_gate(base: pd.DataFrame, kind: str) -> pd.Series:
    """CAUSAL daily uptrend gate (bool, daily index, shifted 1 day)."""
    dc = base["close"].resample("1D").last().dropna()
    if kind == "ema200d":
        g = dc > ta.ema(dc, length=200)
    elif kind == "kama":
        k = ta.kama(dc, length=14, fast=2, slow=30)
        g = k.diff() > 0
    else:
        raise ValueError(kind)
    return g.shift(1)                                 # known as of the next day


def collect(d: pd.DataFrame, level: pd.Series, gate: pd.Series, args) -> pd.DataFrame:
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    atr = ta.atr(d["high"], d["low"], d["close"], length=args.atr).values
    lev = level.values
    gt = gate.values if gate is not None else np.ones(len(c), dtype=bool)
    n = len(c)
    fwd, L, tolk, react = args.fwd, args.lookback, args.tol_atr, args.react_atr
    rows = []
    last_tag = -10**9
    start = max(args.atr + 2, L + 1)
    i = start
    while i < n - 1:
        a = atr[i]
        lv = lev[i]
        if np.isnan(a) or a <= 0 or np.isnan(lv) or i - last_tag < fwd or not gt[i]:
            i += 1; continue
        tol = tolk * a
        # support tag: low pierces to within tol of the line, prior close was above
        if not (l[i] <= lv + tol and c[i - 1] > lv):
            i += 1; continue
        e = c[i]
        win = h[i - L:i]
        drop = (win.max() - l[i]) / a                  # depth of the fall into the line (ATR)
        bars_since_high = L - int(np.argmax(win))       # bars from the recent high to the tag
        speed = drop / max(bars_since_high, 1)          # steepness of the fall (ATR/bar)
        fh, fl = h[i + 1:i + 1 + fwd], l[i + 1:i + 1 + fwd]
        if len(fh) == 0:
            i += 1; continue
        mfe = (fh.max() - e) / a
        mae = (e - fl.min()) / a
        up_hit, dn_hit = e + react * a, e - react * a
        outcome = 0                                   # first-touch race
        for j in range(i + 1, min(i + 1 + fwd, n)):
            hit_dn = l[j] <= dn_hit
            hit_up = h[j] >= up_hit
            if hit_dn:                                # same-bar both -> fail (conservative)
                outcome = 0; break
            if hit_up:
                outcome = 1; break
        rows.append((d.index[i], drop, speed, mfe, mae, outcome))
        last_tag = i
        i += 1
    return pd.DataFrame(rows, columns=["time", "drop", "speed", "mfe", "mae", "bounce"])


def report_tf(tf: str, t: pd.DataFrame) -> None:
    if len(t) < 12:
        print(f"  {tf:<4} n={len(t):<4} (too few tags)"); return
    ratio = t["mfe"].mean() / t["mae"].mean() if t["mae"].mean() > 0 else float("inf")
    print(f"  {tf:<4} N={len(t):<4} react={t['bounce'].mean()*100:4.0f}%  "
          f"MFE/MAE={ratio:4.2f} (med {t['mfe'].median():.2f}/{t['mae'].median():.2f}, "
          f"MFE std {t['mfe'].std():.2f})  drop med={t['drop'].median():.1f}ATR "
          f"speed med={t['speed'].median():.2f}ATR/bar")

    def split(col, unit):
        q1, q2 = t[col].quantile([1 / 3, 2 / 3])
        buckets = [("shallow", t[t[col] <= q1]),
                   ("mid    ", t[(t[col] > q1) & (t[col] <= q2)]),
                   ("deep   ", t[t[col] > q2])]
        for name, b in buckets:
            if len(b) == 0:
                continue
            r = b["mfe"].mean() / b["mae"].mean() if b["mae"].mean() > 0 else float("inf")
            lo, hi = b[col].min(), b[col].max()
            print(f"        [{col:>5}] {name} {lo:5.2f}-{hi:5.2f}{unit}  n={len(b):<4} "
                  f"react={b['bounce'].mean()*100:4.0f}%  MFE/MAE={r:4.2f}  "
                  f"medMFE={b['mfe'].median():.2f}")

    split("drop", "ATR")
    split("speed", "A/b")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tf", default="1h,2h,4h,8h,1d", help="comma list")
    ap.add_argument("--line", default="ema200", choices=["ema200", "swing"])
    ap.add_argument("--ema", type=int, default=200)
    ap.add_argument("--fractal-n", type=int, default=3)
    ap.add_argument("--tol-atr", type=float, default=0.5, help="tag tolerance (ATR)")
    ap.add_argument("--react-atr", type=float, default=1.0, help="bounce threshold (ATR)")
    ap.add_argument("--lookback", type=int, default=20, help="bars for the recent-drop high")
    ap.add_argument("--fwd", type=int, default=24, help="forward bars for reaction/excursion")
    ap.add_argument("--gate", default="none", choices=["none", "ema200d", "kama"],
                    help="daily uptrend gate: price>daily-200EMA or daily-KAMA-rising")
    ap.add_argument("--atr", type=int, default=14)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    base = load_mt5_csv(args.csv)
    if args.start or args.end:
        base = base.loc[args.start:args.end]

    gser = daily_gate(base, args.gate) if args.gate != "none" else None
    print(f"\n=== line reaction  {os.path.basename(args.csv)}  line={args.line}  gate={args.gate}"
          f"  tol={args.tol_atr}ATR react={args.react_atr}ATR fwd={args.fwd} lookback={args.lookback} ===")
    print(f"    {base.index[0].date()} -> {base.index[-1].date()}  (support / long bounce)")
    for tf in (x.strip() for x in args.tf.split(",") if x.strip()):
        d = resample(base, tf)
        lev = level_series(d, args.line, args.ema, args.fractal_n)
        gate = gser.reindex(d.index).ffill().fillna(False) if gser is not None else None
        t = collect(d, lev, gate, args)
        report_tf(tf, t)


if __name__ == "__main__":
    main()
