"""fvg_retest_screen.py -- faithful SMC Fair-Value-Gap RETEST entry screen.

The marketplace "SMC Flow" indicator draws an FVG as a ZONE (box) and treats it
as live support/resistance until `close` mitigates it. The doctrinal trade is
NOT "buy on the formation bar" -- it is: wait for price to RETRACE into the gap
and enter in the gap direction (bullish gap -> long on the dip into it), with a
LIMIT fill at the near edge of the zone. This screens exactly that.

Detection matches the Pine verbatim:
  bull_fvg = low > high[2] and close[1] > high[2]      zone = [high[2], low]
  bear_fvg = high < low[2] and close[1] < low[2]       zone = [high, low[2]]
Mitigation matches the Pine:
  bull mitigated when close < high[2] (zone bottom); bear when close > low[2] (zone top).

Entry: LIMIT at the near edge (bull: gap top = low[t]; bear: gap bottom = high[t]).
Filled on the first later bar that taps the zone before it is mitigated or expires.
Excursion + fixed TP/SL measured from the fill bar, in ATR units, round-trip cost.
"""
import argparse
import numpy as np
import pandas as pd
import pandas_ta as ta

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv


def resample(df, rule):
    if rule in ("1h", "h1", ""):
        return df
    o = {"4h": "4h", "h4": "4h", "1d": "1D", "d1": "1D"}.get(rule.lower(), rule)
    return pd.DataFrame({
        "open": df["open"].resample(o).first(), "high": df["high"].resample(o).max(),
        "low": df["low"].resample(o).min(), "close": df["close"].resample(o).last(),
    }).dropna()


def fvg_trades(d, side, maxwait=50, N=30, TP=2.5, SL=2.0, cost=0.001, atrlen=14, edge="near",
               no_overlap=False):
    """Return (trades_df[time,R,mfe,mae], n_no_retest) for FVG retest entries.
    no_overlap=True: one position at a time (skip fills while a trade is still open) -- honest
    sequential-1%-risk equity; kills the concurrent-exposure lumpiness that inflates DD."""
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    atr = ta.atr(d["high"], d["low"], d["close"], length=atrlen).values
    n = len(c)
    trades = []           # (fill_time, R, MFE, MAE)
    no_retest = 0         # gaps that ran away (never tapped) -> the strong ones
    open_until = -1       # bar index until which a position is held (no_overlap)

    for t in range(2, n - 1):
        if side == "long":
            if not (l[t] > h[t - 2] and c[t - 1] > h[t - 2]):
                continue
            gap_lo, gap_hi = h[t - 2], l[t]          # zone = [high[2], low]
            e = {"near": gap_hi, "mid": (gap_lo + gap_hi) / 2, "far": gap_lo}[edge]
        else:
            if not (h[t] < l[t - 2] and c[t - 1] < l[t - 2]):
                continue
            gap_lo, gap_hi = h[t], l[t - 2]          # zone = [high, low[2]]
            e = {"near": gap_lo, "mid": (gap_lo + gap_hi) / 2, "far": gap_hi}[edge]

        # ORDER MATTERS (no lookahead): a resting limit fills on TOUCH during the bar; the
        # close-based mitigation can only cancel a still-unfilled order AFTER that bar closes.
        # Checking mitigation first would use the bar's close to reject its own intrabar fill
        # (silently deleting the worst fills = optimistic bias).
        fill = None
        for j in range(t + 1, min(t + 1 + maxwait, n)):
            if side == "long":
                if l[j] <= e:                        # price reached the limit -> fills
                    fill = j; break
                if c[j] < gap_lo:                    # unfilled + mitigated -> cancel order
                    break
            else:
                if h[j] >= e:
                    fill = j; break
                if c[j] > gap_hi:
                    break
        if fill is None:
            no_retest += 1
            continue
        if no_overlap and fill <= open_until:   # a position is still open -> skip
            continue

        a = atr[fill]
        if np.isnan(a) or a <= 0 or fill + 1 >= n:
            continue
        fh, fl = h[fill + 1:fill + 1 + N], l[fill + 1:fill + 1 + N]
        if len(fh) == 0:
            continue
        exu = min(fill + N, n - 1)              # exit bar (fwd-end unless TP/SL hits first)
        # same-bar handling on the fill bar: intrabar sequence is unknown, so be conservative —
        # a stop touched on the fill bar counts as a loss; a TP touched on it is NOT credited.
        if side == "long":
            mfe = (fh.max() - e) / a; mae = (e - fl.min()) / a
            stop, tgt = e - SL * a, e + TP * a; R = None
            if l[fill] <= stop: R = -SL; exu = fill
            else:
                for k in range(fill + 1, min(fill + 1 + N, n)):
                    if l[k] <= stop: R = -SL; exu = k; break
                    if h[k] >= tgt: R = TP; exu = k; break
            if R is None: R = (c[min(fill + N, n - 1)] - e) / a
        else:
            mfe = (e - fl.min()) / a; mae = (fh.max() - e) / a
            stop, tgt = e + SL * a, e - TP * a; R = None
            if h[fill] >= stop: R = -SL; exu = fill
            else:
                for k in range(fill + 1, min(fill + 1 + N, n)):
                    if h[k] >= stop: R = -SL; exu = k; break
                    if l[k] <= tgt: R = TP; exu = k; break
            if R is None: R = (e - c[min(fill + N, n - 1)]) / a
        R -= cost / a * e
        open_until = exu
        trades.append((d.index[fill], R, mfe, mae))

    return pd.DataFrame(trades, columns=["time", "R", "mfe", "mae"]), no_retest


def run(d, side, maxwait, N, TP, SL, cost, atrlen, edge="near"):
    tdf, no_retest = fvg_trades(d, side, maxwait, N, TP, SL, cost, atrlen, edge)
    if len(tdf) == 0:
        print(f"  [{side}] no retest fills."); return
    tdf["y"] = tdf["time"].dt.year
    ratio = tdf["mfe"].mean() / tdf["mae"].mean() if tdf["mae"].mean() > 0 else float("inf")
    verdict = "DEAD" if ratio < 1.0 else "marginal" if ratio < 1.2 else "EDGE?"
    yrs = sorted(tdf["y"].unique()); half = yrs[len(yrs) // 2]
    is_, oos = tdf[tdf["y"] < half]["R"], tdf[tdf["y"] >= half]["R"]
    print(f"\n  ===== {side.upper()}  ({len(tdf)} retest fills | {no_retest} gaps ran away, no retest) =====")
    print(f"  MFE/MAE: {tdf['mfe'].mean():.2f} / {tdf['mae'].mean():.2f}  ratio={ratio:.2f}  --> {verdict}")
    print(f"  fixed TP{TP}/SL{SL} (fwd {N}): win={(tdf['R']>0).mean()*100:.0f}%  "
          f"meanR={tdf['R'].mean():+.2f}  totR={tdf['R'].sum():+.0f}")
    print(f"  IS(<{half}) meanR={is_.mean():+.2f} (n={len(is_)}) | OOS(>={half}) meanR={oos.mean():+.2f} (n={len(oos)})")
    print("  per-year meanR: " + "  ".join(
        f"{y}:{tdf[tdf['y']==y]['R'].mean():+.2f}" for y in yrs if len(tdf[tdf['y']==y])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--tf", default="4h")
    p.add_argument("--side", default="both", choices=["long", "short", "both"])
    p.add_argument("--maxwait", type=int, default=50, help="bars to wait for a retest before expiry")
    p.add_argument("--edge", default="near", choices=["near", "mid", "far"], help="limit fill edge of the gap")
    p.add_argument("--fwd", type=int, default=30)
    p.add_argument("--tp", type=float, default=2.5)
    p.add_argument("--sl", type=float, default=2.0)
    p.add_argument("--cost", type=float, default=0.001)
    p.add_argument("--atr", type=int, default=14)
    p.add_argument("--start", default=None); p.add_argument("--end", default=None)
    args = p.parse_args()

    d = load_mt5_csv(args.csv)
    if args.start or args.end:
        d = d.loc[args.start:args.end]
    d = resample(d, args.tf)
    print(f"\n=== FVG RETEST screen: {args.csv}  TF={args.tf}  maxwait={args.maxwait} ===")
    print(f"  {len(d):,} {args.tf} bars  {d.index[0].date()} -> {d.index[-1].date()}")
    for s in (["long", "short"] if args.side == "both" else [args.side]):
        run(d, s, args.maxwait, args.fwd, args.tp, args.sl, args.cost, args.atr, args.edge)


if __name__ == "__main__":
    main()
