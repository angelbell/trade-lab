"""polymarket_btc5m.py -- falsify the spot-side premise of the Novals83/5min-btc-polymarket system.

The system bets BTC 5-min Up/Down binaries: ~2 min before expiry, if BTC moved ~$70-100 in the
window so far, enter MOMENTUM (bet the move continues to expiry). We can't see Polymarket's odds,
but the NECESSARY condition is testable on spot alone:

  In a fixed 5-min window, conditional on |price@min3 - open| >= threshold, does the window CLOSE in
  that direction (settle same), and does price CONTINUE in the last 2 min (fwd move > 0) or REVERT?

If the last-2-min forward move is ~0 (martingale) or <0 (reversion), the momentum-into-expiry premise
has NO spot edge -> you could only profit from Polymarket mispricing (unmeasured here). If it's
meaningfully >0 and survives a binary spread/fee haircut, the premise has a basis worth a live check.

Data: Binance BTC/USDT 1m via ccxt (cached). Windows = fixed 5-min boundaries (Polymarket-style).

  .venv/bin/python research/polymarket_btc5m.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import fetch_ohlcv

START, END = "2025-06-01", "2026-06-01"     # 1 year of 1m


def main():
    d = fetch_ohlcv("BTC/USDT", "1m", START, END)
    d = d[["open", "high", "low", "close"]].copy()
    print(f"polymarket_btc5m -- BTC/USDT 1m {d.index[0]}->{d.index[-1]} ({len(d)} bars)")

    d["w"] = d.index.floor("5min")
    g = d.groupby("w")
    win = g.agg(o=("open", "first"), c=("close", "last"), n=("close", "size"),
                c3=("close", lambda s: s.iloc[2] if len(s) >= 3 else np.nan))
    win = win[win.n == 5].dropna()
    win["move3"] = win.c3 - win.o          # move by minute 3 (entry = "2 min left")
    win["final"] = win.c - win.o           # full 5-min window move (settles the binary)
    win["fwd"]   = win.c - win.c3          # forward 2-min move after entry
    n_all = len(win)
    print(f"  complete 5-min windows: {n_all}  | base P(window up) = {(win.final>0).mean()*100:.1f}%\n")

    print(f"  {'thr($)':>7}{'n':>8}{'%win':>7}{'P(settle same)':>16}{'fwd|dir mean$':>15}"
          f"{'fwd|dir bps':>13}{'cont>0%':>9}")
    for thr in (30, 50, 70, 100, 150, 200):
        m = win[win.move3.abs() >= thr]
        if len(m) < 50:
            print(f"  {thr:>7}{len(m):>8}  (too few)"); continue
        dir = np.sign(m.move3)
        settle_same = (np.sign(m.final) == dir).mean()
        fwd_dir = dir * m.fwd                          # forward move in entry direction
        fwd_bps = (fwd_dir / m.c3) * 1e4
        cont = (fwd_dir > 0).mean()
        print(f"  {thr:>7}{len(m):>8}{(m.final.gt(0)).mean()*100:>6.0f}%{settle_same*100:>15.1f}%"
              f"{fwd_dir.mean():>+15.2f}{fwd_bps.mean():>+13.2f}{cont*100:>8.0f}%")

    print("\n  read:")
    print("   - P(settle same) high just because it's ALREADY up $X at entry (binary ~already priced).")
    print("   - DECISIVE = fwd|dir (last-2-min move in the impulse direction): >0 momentum / ~0 martingale / <0 REVERSION.")
    print("   - if fwd<=0, momentum-into-expiry has NO spot edge; profit would need Polymarket mispricing (untested).")
    # binary spread/fee haircut reference: Polymarket near-expiry spread ~1-3c on a ~$0.85 share = ~120-350 bps
    print("   - a near-expiry binary spread (~1-3c) = ~120-350 bps drag -> fwd edge must clear that to be real.")


if __name__ == "__main__":
    main()
