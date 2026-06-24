"""mfe_mae.py -- fast entry-edge screener via MFE/MAE excursion analysis.

The cheap top-of-funnel filter. Instead of building a full strategy + WFO
(days), this answers in seconds the ONE question that kills 90% of ideas:

    After this entry fires, does price travel FURTHER in our favour (MFE,
    maximum favourable excursion) than against us (MAE, maximum adverse
    excursion)?  ratio = mean(MFE)/mean(MAE).

  ratio < 1.0  -> the entry direction has NEGATIVE edge. No exit can save it.
                  DISCARD (e.g. BTC 4H-structure breakout = 0.89).
  ratio 1.0-1.2-> marginal; only worth it if exits/sizing are great.
  ratio > 1.2  -> directional edge EXISTS; worth the deeper test
                  (e.g. GOLD 4H breakout = 1.51).

Then a stage-2 read: a fixed TP/SL exit's per-trade expectancy + per-year and
IS/OOS split, so you can see if the edge is robust or just regime-luck.

LIMITS (do not over-trust):
  * In-sample. ratio>1.2 means "worth pursuing", NOT "confirmed edge" --
    GOLD passed the screen yet was IS break-even per-year (regime-dependent).
  * MFE is the PEAK; you can't actually capture it. It says the raw material
    is there, not that a realisable profit is.
  * Keep --fwd (forward window) and the TF the same when comparing ideas.

Examples:
  # GOLD 4H breakout, long-only above its 200-bar SMA (the edge we found)
  .venv/bin/python mfe_mae.py --csv data/vantage_xauusd_h1.csv --tf 4h \
      --entry breakout --period 55 --side long --sma 200

  # BTC 4H-structure swing breakout (the one that screened DEAD)
  .venv/bin/python mfe_mae.py --csv data/vantage_btcusd_h1.csv --tf 1h \
      --entry swing --htf 4h --side long --sma-daily 150

  # USDJPY Bollinger mean-reversion (a different family to screen)
  .venv/bin/python mfe_mae.py --csv data/vantage_usdjpy_h1.csv --tf 1h \
      --entry meanrev --period 20 --k 2.0 --side long
"""

import argparse

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.data_loader import load_mt5_csv
from src.strategy import _confirmed_swings


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule in ("1h", "h1", ""):
        return df
    o = {"4h": "4h", "h4": "4h", "1d": "1D", "d1": "1D", "1w": "1W"}.get(rule.lower(), rule)
    out = pd.DataFrame({
        "open":  df["open"].resample(o).first(),
        "high":  df["high"].resample(o).max(),
        "low":   df["low"].resample(o).min(),
        "close": df["close"].resample(o).last(),
    }).dropna()
    return out


def build_signals(d: pd.DataFrame, args) -> tuple[pd.Series, pd.Series]:
    """Return (long_sig, short_sig) booleans, throttled to the first bar of
    each signal (so we count distinct entries, not every bar in the state)."""
    c, h, l = d["close"], d["high"], d["low"]

    if args.entry == "breakout":
        hi = h.rolling(args.period).max().shift(1)
        lo = l.rolling(args.period).min().shift(1)
        long_sig, short_sig = (c > hi), (c < lo)

    elif args.entry == "swing":
        # HTF fractal swing structure (the structure_sma style)
        hh = h.resample(args.htf).max()
        ll = l.resample(args.htf).min()
        lsh, psh, lsl, psl = _confirmed_swings(hh, ll, args.fractal_n)
        up = (lsl > psl); dn = (lsh < psh)
        to = lambda s: s.shift(1).reindex(d.index, method="ffill")
        up_l, dn_l = to(up).fillna(False), to(dn).fillna(False)
        sh, sl = to(lsh), to(lsl)
        long_sig  = up_l & ~dn_l & (c > sh)
        short_sig = dn_l & ~up_l & (c < sl)

    elif args.entry == "meanrev":
        mid = c.rolling(args.period).mean()
        sd  = c.rolling(args.period).std()
        long_sig  = c < (mid - args.k * sd)   # oversold -> fade up
        short_sig = c > (mid + args.k * sd)    # overbought -> fade down

    else:
        raise ValueError(f"unknown entry: {args.entry}")

    long_sig  = long_sig.fillna(False)
    short_sig = short_sig.fillna(False)

    # regime filter: same-TF SMA (--sma) or daily SMA (--sma-daily)
    if args.sma > 0:
        sma = c.rolling(args.sma).mean().shift(1)
        long_sig  &= (c > sma).fillna(False)
        short_sig &= (c < sma).fillna(False)
    if args.sma_daily > 0:
        dc = c.resample("1D").last().dropna()           # dropna = gap-safe (matches iMA)
        ds = dc.rolling(args.sma_daily).mean()
        bull = (dc > ds).shift(1).reindex(d.index, method="ffill").fillna(False)
        bear = (dc < ds).shift(1).reindex(d.index, method="ffill").fillna(False)
        long_sig  &= bull
        short_sig &= bear

    # throttle to the first bar of each run
    long_sig  = long_sig  & ~long_sig.shift(1).fillna(False)
    short_sig = short_sig & ~short_sig.shift(1).fillna(False)
    return long_sig, short_sig


def screen(d: pd.DataFrame, sig: pd.Series, side: str, args) -> None:
    atr = ta.atr(d["high"], d["low"], d["close"], length=args.atr).values
    c, h, l = d["close"].values, d["high"].values, d["low"].values
    idx = np.where(sig.values)[0]
    N, TP, SL = args.fwd, args.tp, args.sl

    mfe, mae, trades = [], [], []
    for i in idx:
        if i + 1 >= len(c) or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        e = c[i]
        fh, fl = h[i + 1:i + 1 + N], l[i + 1:i + 1 + N]
        if len(fh) == 0:
            continue
        if side == "long":
            mfe.append((fh.max() - e) / atr[i]); mae.append((e - fl.min()) / atr[i])
            stop, tgt = e - SL * atr[i], e + TP * atr[i]
            R = None
            for j in range(i + 1, min(i + 1 + N, len(c))):
                if l[j] <= stop: R = -SL; break
                if h[j] >= tgt: R = TP; break
            if R is None: R = (c[min(i + N, len(c) - 1)] - e) / atr[i]
        else:  # short
            mfe.append((e - fl.min()) / atr[i]); mae.append((fh.max() - e) / atr[i])
            stop, tgt = e + SL * atr[i], e - TP * atr[i]
            R = None
            for j in range(i + 1, min(i + 1 + N, len(c))):
                if h[j] >= stop: R = -SL; break
                if l[j] <= tgt: R = TP; break
            if R is None: R = (e - c[min(i + N, len(c) - 1)]) / atr[i]
        R -= args.cost / atr[i] * e   # round-trip cost in ATR units
        trades.append((d.index[i], R))

    mfe, mae = np.array(mfe), np.array(mae)
    if len(mfe) == 0:
        print(f"  [{side}] no entries."); return
    ratio = mfe.mean() / mae.mean() if mae.mean() > 0 else float("inf")
    verdict = "DEAD (skip)" if ratio < 1.0 else "marginal" if ratio < 1.2 else "EDGE -> deeper test"

    t = pd.DataFrame(trades, columns=["time", "R"]); t["y"] = t["time"].dt.year
    print(f"\n  ===== {side.upper()}  ({len(mfe)} entries) =====")
    print(f"  MFE/MAE: {mfe.mean():.2f} / {mae.mean():.2f}  ratio = {ratio:.2f}   --> {verdict}")
    print(f"  fixed exit TP{TP}/SL{SL} ATR (fwd {N} bars): "
          f"win={ (t['R']>0).mean()*100:.0f}%  meanR={t['R'].mean():+.2f}  totalR={t['R'].sum():+.0f}")
    yrs = sorted(t["y"].unique())
    if len(yrs) > 1:
        half = yrs[len(yrs)//2]
        is_, oos = t[t["y"] < half]["R"], t[t["y"] >= half]["R"]
        print(f"  IS (<{half}) meanR={is_.mean():+.2f} (n={len(is_)}) | "
              f"OOS (>={half}) meanR={oos.mean():+.2f} (n={len(oos)})")
        print("  per-year meanR: " + "  ".join(
            f"{y}:{t[t['y']==y]['R'].mean():+.2f}" for y in yrs if len(t[t["y"]==y])))


def main() -> None:
    p = argparse.ArgumentParser(description="MFE/MAE entry-edge screener")
    p.add_argument("--csv", required=True, help="MT5-exported H1 CSV")
    p.add_argument("--tf", default="1h", help="trade timeframe: 1h/4h/1d")
    p.add_argument("--entry", default="breakout", choices=["breakout", "swing", "meanrev"])
    p.add_argument("--side", default="long", choices=["long", "short", "both"])
    p.add_argument("--period", type=int, default=55, help="breakout/meanrev lookback")
    p.add_argument("--k", type=float, default=2.0, help="meanrev band width (std)")
    p.add_argument("--htf", default="4h", help="swing structure timeframe")
    p.add_argument("--fractal-n", type=int, default=3, help="swing pivot bars each side")
    p.add_argument("--sma", type=int, default=0, help="same-TF SMA regime filter (0=off)")
    p.add_argument("--sma-daily", type=int, default=0, help="daily SMA regime filter (0=off)")
    p.add_argument("--atr", type=int, default=14)
    p.add_argument("--fwd", type=int, default=30, help="forward bars for excursion")
    p.add_argument("--tp", type=float, default=2.5, help="stage-2 take-profit (ATR)")
    p.add_argument("--sl", type=float, default=2.0, help="stage-2 stop (ATR)")
    p.add_argument("--cost", type=float, default=0.001, help="round-trip cost fraction")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    args = p.parse_args()

    d = load_mt5_csv(args.csv)
    if args.start or args.end:
        d = d.loc[args.start:args.end]
    d = resample(d, args.tf)

    print(f"\n=== MFE/MAE screen: {args.csv}  TF={args.tf}  entry={args.entry}"
          f"{'/swing'+args.htf if args.entry=='swing' else ''}"
          f"  filter={'sma'+str(args.sma) if args.sma else ''}"
          f"{'dailySMA'+str(args.sma_daily) if args.sma_daily else ''} ===")
    print(f"  {len(d):,} {args.tf} bars  {d.index[0].date()} -> {d.index[-1].date()}")

    long_sig, short_sig = build_signals(d, args)
    if args.side in ("long", "both"):
        screen(d, long_sig, "long", args)
    if args.side in ("short", "both"):
        screen(d, short_sig, "short", args)


if __name__ == "__main__":
    main()
