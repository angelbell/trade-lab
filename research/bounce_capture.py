"""bounce_capture.py -- of the raw HTF bounce, how much is CAPTURABLE?

bounce_size.py measured the bounce from the exact dip-low to the exact peak (full
hindsight). This measures the realistic version: take the actual CONFIRMED entries
that bounce_signals(struct,mom) would fire, fill at next-bar open like the real
backtest, then for each trade record:
    MFE  = best favorable excursion (pips) before the trade's natural exit
    MAE  = worst adverse excursion (pips) before exit
    capt = what the backtest actually banked (pips, after cost)
    1R   = the stop distance (pips) -- so MFE/MAE in R is comparable across trades
Exit horizon = until SL/TP/session-close, same as the live backtest (so MFE/MAE are
what was reachable WHILE the trade was open, not unlimited hindsight).

The point: if MFE (what was on offer) is large but capt (what we kept) is small, the
leak is the EXIT. If MAE ~ MFE, entry is too late / no edge. If MFE small, the
confirmation eats the move.

  .venv/bin/python research/bounce_capture.py --csv data/vantage_xauusd_m5.csv --split is
"""
import argparse, os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from scalp_lab import bounce_signals, SPLITS, PIP


def make_params(**over):
    base = dict(level_lb=288, tol_atr=0.5, tp_lb=48, sl_lb=12, cf_lb=6, cf_win=6,
                confirm="struct,mom", htf_sr=False, trendline=False, htf_level=True,
                piv_tf="15min,1h", piv_k=3, sr_keep=2, double_bottom=False,
                db_win=48, db_gap=3, dir="both", force_exit_h=20, sl_buf_atr=0.0,
                mom_th=45.0, rej=0.6, min_rr=1.0, cost=1.4)
    base.update(over)
    return SimpleNamespace(**base)


def analyze(d, p):
    dir_, sl_px, tp_px = bounce_signals(d, p)
    op, hi, lo = d["open"].values, d["high"].values, d["low"].values
    minute = (d.index.hour * 60 + d.index.minute).values
    cost = p.cost
    n = len(op)
    rows = []
    pos = 0
    for i in range(n - 1):
        # close out an open position exactly like backtest() does
        if pos != 0:
            done_exit = None
            if minute[i] >= p.force_exit_h * 60:
                done_exit = op[i]
            elif pos > 0:
                if lo[i] <= stop: done_exit = stop
                elif hi[i] >= tp: done_exit = tp
            else:
                if hi[i] >= stop: done_exit = stop
                elif lo[i] <= tp: done_exit = tp
            # track MFE/MAE on every bar the trade is open (this bar inclusive)
            if pos > 0:
                mfe = max(mfe, (hi[i] - e_px)); mae = min(mae, (lo[i] - e_px))
            else:
                mfe = max(mfe, (e_px - lo[i])); mae = min(mae, (e_px - hi[i]))
            if done_exit is not None:
                g = (done_exit - e_px) if pos > 0 else (e_px - done_exit)
                rows.append((pos, mfe / PIP, mae / PIP, g / PIP - cost, one_r / PIP))
                pos = 0
        if pos != 0:
            continue
        if dir_[i] == 0:
            continue
        e_px = op[i + 1]; pos = int(dir_[i]); stop = sl_px[i]; tp = tp_px[i]
        one_r = (e_px - stop) if pos > 0 else (stop - e_px)
        mfe = -1e9; mae = 1e9
    return pd.DataFrame(rows, columns=["dir", "mfe", "mae", "capt", "R"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", default="is", choices=["is", "val", "test"])
    ap.add_argument("--cost", type=float, default=1.4)
    a = ap.parse_args()
    s, e = SPLITS[a.split]
    d = load_mt5_csv(a.csv).loc[s:e]
    p = make_params(cost=a.cost)
    t = analyze(d, p)
    print(f"\n=== bounce CAPTURE  split={a.split}  confirm=struct,mom htf-level keep2  "
          f"cost={a.cost}pip  {d.index[0]} -> {d.index[-1]} ===")
    print(f"  trades={len(t)}  win={(t.capt>0).mean()*100:.0f}%  net={t.capt.sum():+.0f}p  "
          f"PF={t[t.capt>0].capt.sum()/abs(t[t.capt<0].capt.sum()):.2f}")
    def q(s_, lab, unit="pip"):
        v = np.percentile(s_, [25, 50, 75])
        print(f"  {lab:<22} mean={s_.mean():6.1f}  median={v[1]:6.1f}  p25/p75={v[0]:.0f}/{v[2]:.0f} {unit}")
    q(t.mfe, "MFE (offered, favor)")
    q(t.mae, "MAE (offered, adverse)")
    q(t.capt, "captured (after cost)")
    q(t.R,   "1R stop distance")
    print(f"  -- MFE in R (how far it ran vs the risk) --")
    rr = t.mfe / t.R
    v = np.percentile(rr, [25, 50, 75])
    print(f"  MFE/R  median={v[1]:.2f}  p25/p75={v[0]:.2f}/{v[2]:.2f}   "
          f"reached 1R:{(rr>=1).mean()*100:.0f}%  1.5R:{(rr>=1.5).mean()*100:.0f}%  2R:{(rr>=2).mean()*100:.0f}%")
    print(f"  capture efficiency (capt/MFE, winners only): "
          f"{(t[t.capt>0].capt / t[t.capt>0].mfe).median()*100:.0f}% of the offered move kept")


if __name__ == "__main__":
    main()
