"""slip_audit.py -- how violent are H17's STOP-OUT bars, really?

The uniform --stop-slip stress is pessimistic (it slips EVERY stop). Reality: most
stops fill at/near the stop; only the occasional headline candle gaps through. This
audits the actual stop-out bars and measures, per stop, the OVERSHOOT = how far the
bar's adverse extreme ran BEYOND the stop price (= the worst slip you could have eaten
on that bar if you filled at the extreme instead of the stop). The distribution of
overshoot answers: how often is a stop-out a genuine violent gap vs a routine touch?

Reuses the exact H17 signal + gates from scalp_lab (validated config hardcoded).

  .venv/bin/python research/slip_audit.py --csv data/vantage_xauusd_m15.csv --split is
"""
import argparse, os, sys
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from research.scalp_lab import orb_signals, htf_trend_gate, daily_gate, vol_gate, SPLITS, PIP


def h17_params(**over):
    base = dict(asia_start_h=0, asia_end_h=7, bo_start_h=7, bo_end_h=11, force_exit_h=20,
                rr=1.0, buf_atr=0.0, sl_buf_atr=0.0, max_range_atr=0.0, min_range_atr=0.0,
                no_tp=True, fade=False, dir="both", sl_frac=1.0, rsi_max=100.0, box_trend_max=1.0,
                htf_tf="1h", htf_ema=80, htf_slope_k=0, daily_sma=0, daily_slope_k=0,
                vol_band="all", atr_win=8000, cost=1.4)
    base.update(over)
    return SimpleNamespace(**base)


def audit(d, p):
    """Replicate the scalp_lab backtester but record, on each STOP-OUT, the overshoot
    (adverse extreme beyond the stop) and the bar's range, in pips."""
    dir_, sl_px, tp_px = orb_signals(d, p)
    dir_, sl_px, tp_px = vol_gate(d, dir_, sl_px, tp_px, p)
    dir_, sl_px, tp_px = daily_gate(d, dir_, sl_px, tp_px, p)
    dir_, sl_px, tp_px = htf_trend_gate(d, dir_, sl_px, tp_px, p)
    op, hi, lo = d["open"].values, d["high"].values, d["low"].values
    minute = (d.index.hour * 60 + d.index.minute).values
    n = len(op)
    pos = 0; e_px = stop = tp = 0.0
    overs = []           # overshoot pips on stop-outs (>=0)
    gap_open = []        # 1 if the bar OPENED already beyond the stop (true gap), else 0
    n_stop = n_tp = n_eod = 0
    for i in range(n - 1):
        if pos != 0:
            if minute[i] >= p.force_exit_h * 60:
                pos = 0; n_eod += 1; continue
            if pos > 0:
                if lo[i] <= stop:
                    overs.append((stop - lo[i]) / PIP); gap_open.append(1 if op[i] < stop else 0)
                    pos = 0; n_stop += 1
                elif hi[i] >= tp:
                    pos = 0; n_tp += 1
            else:
                if hi[i] >= stop:
                    overs.append((hi[i] - stop) / PIP); gap_open.append(1 if op[i] > stop else 0)
                    pos = 0; n_stop += 1
                elif lo[i] <= tp:
                    pos = 0; n_tp += 1
        if pos != 0:
            continue
        if dir_[i] == 0:
            continue
        e_px = op[i + 1]; pos = int(dir_[i]); stop = sl_px[i]; tp = tp_px[i]
    return np.array(overs), np.array(gap_open), dict(stop=n_stop, tp=n_tp, eod=n_eod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", default="is", choices=["is", "val"])
    ap.add_argument("--max-range-atr", type=float, default=0.0, help="skip days whose Asian range > N*ATR (0=off)")
    a = ap.parse_args()
    s, e = SPLITS[a.split]
    d = load_mt5_csv(a.csv).loc[s:e]
    overs, gap_open, exits = audit(d, h17_params(max_range_atr=a.max_range_atr))
    tot_exits = exits["stop"] + exits["tp"] + exits["eod"]
    print(f"\n=== slip audit  {os.path.basename(a.csv)} split={a.split}  {d.index[0]}->{d.index[-1]} ===")
    print(f"  exits: stop={exits['stop']}  eod(forced flat)={exits['eod']}  tp={exits['tp']}  (total {tot_exits})")
    print(f"  -> {exits['eod']/max(tot_exits,1)*100:.0f}% of trades exit at the SESSION CLOSE, not the stop")
    if len(overs) == 0:
        print("  no stop-outs"); return
    print(f"\n  STOP-OUT bar overshoot (pips past the stop = worst possible slip on that bar):")
    print(f"    n={len(overs)}  mean={overs.mean():.1f}  median={np.median(overs):.1f}  "
          f"p90={np.percentile(overs,90):.1f}  p99={np.percentile(overs,99):.1f}  max={overs.max():.1f}")
    for thr in (10, 20, 40, 80):
        pct = (overs >= thr).mean() * 100
        print(f"    overshoot >= {thr:>3}p : {(overs>=thr).sum():>4} stops  ({pct:4.1f}% of stops, "
              f"{pct*exits['stop']/max(tot_exits,1):4.2f}% of ALL trades)")
    print(f"    bar OPENED already past the stop (true gap): {gap_open.sum()} / {len(overs)} "
          f"({gap_open.mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
