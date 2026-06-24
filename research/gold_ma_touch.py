"""gold_ma_touch.py -- falsify: GOLD, enter when price TOUCHES the 200 MA, fixed SL 5p / TP 10p (RR2).
Gold pip = 0.1 USD (so SL=0.5 / TP=1.0 USD -- tiny). Modes: long / short / bounce (MA-slope aligned).
Touch = bar's [low,high] straddles the MA. Next-bar-open fill, intrabar SL/TP, one position at a time,
cost charged round-trip. Reports n/win/PF/net/IS-OOS, cost-stressed. Breakeven win @RR2 = 33.3%.

  .venv/bin/python research/gold_ma_touch.py --csv data/vantage_xauusd_m5.csv --tf 5min
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv

PIP = 0.1  # gold


def run(d, ma_len, sl_pips, tp_pips, mode, cost_pips):
    c = d["close"]; cv = c.values; op = d["open"].values; hi = d["high"].values; lo = d["low"].values
    ma = c.rolling(ma_len).mean().values
    ma_up = np.r_[False, ma[1:] > ma[:-1]]
    yr = d.index.year.values
    n = len(cv)
    sl, tp = sl_pips * PIP, tp_pips * PIP
    rows = []; busy = -1
    for i in range(ma_len, n - 1):
        if i + 1 <= busy or np.isnan(ma[i]):
            continue
        if not (lo[i] <= ma[i] <= hi[i]):          # bar touches the MA
            continue
        if mode == "long":   dr = 1
        elif mode == "short": dr = -1
        else:                 dr = 1 if ma_up[i] else -1   # bounce: align to MA slope
        ei = i + 1; e_px = op[ei]
        stopp = e_px - dr * sl; tgt = e_px + dr * tp
        R = None
        for j in range(ei, min(ei + 500, n)):
            if dr > 0:
                if lo[j] <= stopp: R = -1.0; break
                if hi[j] >= tgt:   R = tp / sl; break
            else:
                if hi[j] >= stopp: R = -1.0; break
                if lo[j] <= tgt:   R = tp / sl; break
        if R is None:
            R = (cv[min(ei + 500, n - 1)] - e_px) / sl * dr
        R -= cost_pips * PIP / sl                   # round-trip cost in R units
        rows.append((yr[ei], R)); busy = j if R is not None else ei
    t = pd.DataFrame(rows, columns=["y", "R"])
    return t


def stat(t):
    if len(t) == 0:
        return "n=0"
    w = t.R[t.R > 0].sum(); l = t.R[t.R < 0].sum()
    pf = w / abs(l) if l else float("inf")
    isr, oos = t[t.y < 2022].R, t[t.y >= 2022].R
    return (f"n={len(t):>5} win={(t.R>0).mean()*100:>3.0f}% PF={pf:4.2f} meanR={t.R.mean():+.2f} "
            f"totR={t.R.sum():+6.0f} | IS={isr.mean():+.2f} OOS={oos.mean():+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/vantage_xauusd_m5.csv")
    ap.add_argument("--ma", type=int, default=200)
    ap.add_argument("--sl", type=float, default=5)
    ap.add_argument("--tp", type=float, default=10)
    a = ap.parse_args()
    d = load_mt5_csv(a.csv)
    print(f"\n=== GOLD {a.ma}MA touch, SL{a.sl}/TP{a.tp}p (RR{a.tp/a.sl:.0f})  {os.path.basename(a.csv)}  "
          f"{d.index[0].date()}->{d.index[-1].date()}  [breakeven win {100/(1+a.tp/a.sl):.0f}%] ===")
    for mode in ("long", "short", "bounce"):
        print(f"  -- {mode} --")
        for cost in (0, 2, 3):
            print(f"    cost{cost}p: " + stat(run(d, a.ma, a.sl, a.tp, mode, cost)))


if __name__ == "__main__":
    main()
