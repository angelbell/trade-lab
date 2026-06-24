"""lemanchannel.py -- faithful port + falsification of LeManChanel (MT4, LeMan 2009).

The channel projects the NEXT bar's extreme range from PAST data (lookahead-clean):
  ho[t] = high[t] - close[t-1]              # up-excursion from prior close
  ol[t] = close[t-1] - low[t]               # down-excursion from prior close
  Up[t]   = close[t] + max(ho over last N)  # projected upper extreme  (drawn at t+1)
  Down[t] = close[t] - max(ol over last N)  # projected lower extreme
The band for bar t+1 is fixed at the close of bar t => no future leak. Channel/S-R family
(prior: ~0 lift). Indicator has no entry rule, so we test the two natural reads:
  fade  : the band = MAX-excursion extreme -> touching it = stretched -> revert (contrarian)
  break : close beyond the band -> continuation (trend)
Both sides, N sweep, cost, IS/OOS. SL/TP in ATR units (RR). One position at a time.

  .venv/bin/python lemanchannel.py --csv data/vantage_btcusd_h1.csv --tf 4h --mode fade --sweep
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

from src.data_loader import load_mt5_csv


def resample(df, rule):
    if rule.lower() in ("1h", "h1", ""):
        return df
    o = {"4h": "4h", "h4": "4h", "1d": "1D", "d1": "1D", "2h": "2h"}.get(rule.lower(), rule)
    return pd.DataFrame({"open": df["open"].resample(o).first(), "high": df["high"].resample(o).max(),
                         "low": df["low"].resample(o).min(), "close": df["close"].resample(o).last()}).dropna()


def channel(h, l, c, N):
    """projected Up/Down bands, valid (known) FOR bar t from data up to t-1 -> no lookahead."""
    n = len(c)
    ho = np.full(n, np.nan); ol = np.full(n, np.nan)
    ho[1:] = h[1:] - c[:-1]
    ol[1:] = c[:-1] - l[1:]
    up = np.full(n, np.nan); dn = np.full(n, np.nan)
    for t in range(N, n - 1):
        mho = np.nanmax(ho[t - N + 1:t + 1])
        mol = np.nanmax(ol[t - N + 1:t + 1])
        up[t + 1] = c[t] + mho                 # band for bar t+1, fixed at close of t
        dn[t + 1] = c[t] - mol
    return up, dn


def run(d, args, N):
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    atr = ta.atr(d["high"], d["low"], d["close"], length=14).values
    up, dn = channel(h, l, c, N)
    n = len(c); cost = args.cost; rr = args.rr; slm = args.sl_atr
    trades = []
    pos = 0; e_px = stop = tgt = 0.0; ei = 0
    for t in range(N + 1, n - 1):
        if pos != 0:                                   # manage open position intrabar
            if pos > 0:
                if l[t] <= stop: trades.append((d.index[ei], 1, -1.0)); pos = 0
                elif h[t] >= tgt: trades.append((d.index[ei], 1, rr - cost / (e_px - stop) * e_px)); pos = 0
            else:
                if h[t] >= stop: trades.append((d.index[ei], -1, -1.0)); pos = 0
                elif l[t] <= tgt: trades.append((d.index[ei], -1, rr - cost / (stop - e_px) * e_px)); pos = 0
            if t - ei >= args.fwd and pos != 0:        # timeout at close
                R = (c[t] - e_px) / (e_px - stop) if pos > 0 else (e_px - c[t]) / (stop - e_px)
                trades.append((d.index[ei], pos, R - cost)); pos = 0
        if pos != 0 or np.isnan(up[t]) or np.isnan(atr[t]) or atr[t] <= 0:
            continue
        risk = slm * atr[t]
        if args.mode == "fade":
            if not args.confirm:                                       # TOUCH = trade at the band (falling-knife)
                if h[t] >= up[t] and args.side in ("both", "short"):   # tag upper extreme -> short
                    e_px = up[t]; stop = e_px + risk; tgt = e_px - rr * risk; pos = -1; ei = t
                elif l[t] <= dn[t] and args.side in ("both", "long"):  # tag lower extreme -> long
                    e_px = dn[t]; stop = e_px - risk; tgt = e_px + rr * risk; pos = 1; ei = t
            else:                                                      # REJECTION confirm: tagged AND closed back inside
                if h[t] >= up[t] and c[t] < up[t] and args.side in ("both", "short"):
                    e_px = c[t]; stop = e_px + risk; tgt = e_px - rr * risk; pos = -1; ei = t
                elif l[t] <= dn[t] and c[t] > dn[t] and args.side in ("both", "long"):
                    e_px = c[t]; stop = e_px - risk; tgt = e_px + rr * risk; pos = 1; ei = t
        else:  # break (continuation): close beyond band -> enter next open
            if c[t] > up[t] and args.side in ("both", "long"):
                e_px = o[t + 1]; stop = e_px - risk; tgt = e_px + rr * risk; pos = 1; ei = t + 1
            elif c[t] < dn[t] and args.side in ("both", "short"):
                e_px = o[t + 1]; stop = e_px + risk; tgt = e_px - rr * risk; pos = -1; ei = t + 1
    t = pd.DataFrame(trades, columns=["t", "dir", "R"]).dropna()
    if len(t) == 0:
        print(f"  N={N}: no trades"); return
    wins = t.R[t.R > 0]; loss = t.R[t.R < 0]
    pf = wins.sum() / abs(loss.sum()) if len(loss) and loss.sum() else float("inf")
    t["y"] = t.t.dt.year; yrs = sorted(t.y.unique()); half = yrs[len(yrs) // 2] if len(yrs) > 1 else None
    isr = t.R[t.y < half] if half else t.R; oos = t.R[t.y >= half] if half else t.R
    print(f"  N={N:>2} n={len(t):>4} win={(t.R>0).mean()*100:>3.0f}% PF={pf:4.2f} "
          f"meanR={t.R.mean():+.2f} totR={t.R.sum():+5.0f} | IS={isr.mean():+.2f} OOS={oos.mean():+.2f}")
    if args.peryear:
        print("      per-year totR: " + " ".join(f"{y}:{g.R.sum():+.0f}({len(g)})" for y, g in t.groupby("y")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--mode", default="fade", choices=["fade", "break"])
    ap.add_argument("--side", default="both", choices=["both", "long", "short"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--sl-atr", type=float, default=1.0, help="stop distance in ATRs")
    ap.add_argument("--fwd", type=int, default=24, help="max bars to hold")
    ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--confirm", action="store_true", help="fade only on a REJECTION (tagged the band AND closed back inside)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--peryear", action="store_true")
    a = ap.parse_args()
    d = resample(load_mt5_csv(a.csv), a.tf)
    print(f"\n=== {os.path.basename(a.csv)} {a.tf} LeManChanel mode={a.mode} side={a.side} "
          f"rr{a.rr} sl{a.sl_atr}ATR cost{a.cost} {d.index[0].date()}->{d.index[-1].date()} ===")
    for N in ([6, 12, 20, 30] if a.sweep else [a.n]):
        run(d, a, N)


if __name__ == "__main__":
    main()
