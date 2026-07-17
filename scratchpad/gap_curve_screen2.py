"""gap_curve_screen2.py -- "near-field vs far-field" entry screen, RE-SCALED.

v1 (scratchpad/gap_curve_screen.py) placed the stop at k*ATR(14) on the signal bar --
2-5 pips on USDJPY 5m, $1-2 on gold 5m. Cost (0.9 pip / $0.30 round trip) was then
15-36% of the stop, so EVERY cell died of cost regardless of whether the entry had any
raw edge: the screen was measuring cost survival, not edge existence, and never tested
the scale the user actually asked about (50-100 pip targets).

Fix (per coordinator's correction, everything else identical to v1):
  - Stop is no longer ATR-based. Target T is now an ABSOLUTE, FIXED distance (same units
    as scalp_null_bracket.py's null table, so this is directly comparable to it):
        gold T in {5.0, 7.5, 10.0} $/oz ; usdjpy T in {0.50, 0.75, 1.00} JPY ; btc T in
        {500, 750, 1000} $
  - Stop S = T / RR (RR grid shrunk to {0.5,0.7,1.0,1.5,2.0,3.0} -- RR<0.5 would need
    S=T/RR > 2T, absurd here since T itself is now the fixed quantity, not the stop).
  - Max hold 3d -> 10d (T=10/1.00/1000 timed out up to 68% of the time at 3d per the
    null-bracket measurement).
  - cost_R = cost/S is now reported explicitly (the whole point of the rescale).

Because S now depends on RR (unlike v1, where S=k*ATR was RR-independent), the "one
scan gives every RR" trick from v1 no longer collapses across RR: a different S changes
which bar the stop is hit on, so blocking (position-open state) must be evaluated per
(instrument,tf,side,family,T,RR) separately -- this is unavoidable and is explicitly
sanctioned by the corrected spec ("走査は(銘柄×TF×side×家族×T×RR)ごとに必要").

It IS still vectorised within a trade: for a fixed entry, T is fixed regardless of RR,
so the forward favourable/adverse excursion arrays are built ONCE per trade and reused
for every RR via searchsorted on their running cummax (monotonic non-decreasing, so
searchsorted correctly finds "first bar index where this threshold is crossed").
Same-bar stop+target conflict (index tie) resolved as stop-first (conservative).

Reuses (no reinvention): src.data_loader.load_mt5_csv, breakout_wave.resample,
breakout_wave.kama_adaptive; family-signal logic (bb2/bb25/dev2atr/rsi/run3/sweep/
bb2_range/breakout/random) is UNCHANGED from gap_curve_screen.py, imported directly.

Run:
  .venv/bin/python scratchpad/gap_curve_screen2.py --smoke
  .venv/bin/python scratchpad/gap_curve_screen2.py --instrument gold
  .venv/bin/python scratchpad/gap_curve_screen2.py --instrument gold --median-t-only
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap_curve_screen import (
    build_tf_frame, build_daily_context, family_signal, FAMILIES, SIDES, INSTRUMENTS,
)
from src.data_loader import load_mt5_csv

RR_GRID = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
T_GRID = {
    "gold":   [5.0, 7.5, 10.0],
    "btc":    [500.0, 750.0, 1000.0],
    "usdjpy": [0.50, 0.75, 1.00],
}
MEDIAN_T = {"gold": 7.5, "btc": 750.0, "usdjpy": 0.75}
MAX_HOLD_DAYS = 10


# --------------------------------------------------------------------------- simulation
def walk_and_scan2(d: pd.DataFrame, base_df: pd.DataFrame, tf: str, sig: np.ndarray,
                    side: str, T: float, RR: float, cost: float,
                    rng: np.random.Generator = None, random_target_n: int = None):
    """One forward pass for a FIXED (T, RR) bracket: S = T/RR. Position-blocking is
    evaluated for THIS bracket only (its own stop-or-10d-timeout exit event)."""
    idx = d.index
    n_bars = len(d)
    op = d["open"].values
    base_idx_arr = base_df.index
    base_low = base_df["low"].values
    base_high = base_df["high"].values
    base_close = base_df["close"].values
    base_time = base_idx_arr.values

    S = T / RR

    if sig is None:
        eligible = np.arange(n_bars - 1)
        m = min(random_target_n, len(eligible))
        cand = rng.choice(eligible, size=m, replace=False) if m > 0 else np.array([], dtype=int)
        sig_bool = np.zeros(n_bars, dtype=bool)
        sig_bool[cand] = True
        sig = sig_bool

    fire_idx = np.where(sig)[0]
    trades = []
    busy_until = None
    dropped_boundary = 0

    for i in fire_idx:
        if i + 1 >= n_bars:
            continue
        entry_time = idx[i + 1]
        if busy_until is not None and entry_time <= busy_until:
            continue
        entry_price = op[i + 1]

        if tf == "5min":
            eb = i + 1
        else:
            eb = base_idx_arr.searchsorted(entry_time)
            if eb >= len(base_idx_arr):
                dropped_boundary += 1
                continue

        cutoff = entry_time + pd.Timedelta(days=MAX_HOLD_DAYS)
        cend = base_idx_arr.searchsorted(cutoff, side="right")
        start = eb + 1
        if start >= cend:
            dropped_boundary += 1
            continue

        lo = base_low[start:cend]
        hi = base_high[start:cend]
        cl = base_close[start:cend]
        if side == "long":
            fav = hi - entry_price
            adv = entry_price - lo
        else:
            fav = entry_price - lo
            adv = hi - entry_price

        fav_cum = np.maximum.accumulate(np.maximum(fav, 0.0))
        adv_cum = np.maximum.accumulate(np.maximum(adv, 0.0))
        nwin = len(fav_cum)
        idx_tgt = np.searchsorted(fav_cum, T, side="left")
        idx_stop = np.searchsorted(adv_cum, S, side="left")

        if idx_stop < nwin and (idx_tgt >= nwin or idx_stop <= idx_tgt):
            outcome, exit_pos = "loss", idx_stop
        elif idx_tgt < nwin:
            outcome, exit_pos = "win", idx_tgt
        else:
            outcome, exit_pos = "timeout", nwin - 1

        exit_time = base_time[start + exit_pos]
        if outcome == "timeout":
            exit_close = cl[exit_pos]
            timeout_R = ((exit_close - entry_price) / S) if side == "long" else \
                        ((entry_price - exit_close) / S)
        else:
            timeout_R = np.nan

        trades.append((entry_time, outcome, S, timeout_R))
        busy_until = pd.Timestamp(exit_time).tz_localize("UTC")

    cols = ["entry_time", "outcome", "S", "timeout_R"]
    tdf = pd.DataFrame(trades, columns=cols)
    return tdf, dropped_boundary


def summarize_cell2(tdf: pd.DataFrame, RR: float, cost: float, instrument: str, tf: str,
                     side: str, family: str, T: float):
    n = len(tdf)
    if n == 0:
        return None
    win = (tdf["outcome"] == "win").values
    loss = (tdf["outcome"] == "loss").values
    timeout = (tdf["outcome"] == "timeout").values
    S = tdf["S"].values
    R_gross = np.where(win, RR, np.where(loss, -1.0, tdf["timeout_R"].values))
    R_net = R_gross - cost / S
    cost_R = cost / S  # constant per cell (S fixed = T/RR for all trades in this cell)

    span_days = (tdf["entry_time"].max() - tdf["entry_time"].min()).days
    yrs = max(span_days / 365.25, 0.5)
    per_year = n / yrs

    pos_mask = R_gross > 0
    neg_mask = R_gross < 0
    mean_win = R_gross[pos_mask].mean() if pos_mask.any() else np.nan
    mean_loss = R_gross[neg_mask].mean() if neg_mask.any() else np.nan
    eff_rr = mean_win / abs(mean_loss) if (pos_mask.any() and neg_mask.any()) else np.nan
    breakeven = 100.0 / (1.0 + eff_rr) if np.isfinite(eff_rr) else np.nan
    win_rate = pos_mask.mean() * 100.0
    gap = win_rate - breakeven if np.isfinite(breakeven) else np.nan

    pf_gross = (R_gross[pos_mask].sum() / abs(R_gross[neg_mask].sum())
                if neg_mask.any() and R_gross[neg_mask].sum() != 0 else np.nan)
    pos_n = R_net > 0
    neg_n = R_net < 0
    pf_net = (R_net[pos_n].sum() / abs(R_net[neg_n].sum())
              if neg_n.any() and R_net[neg_n].sum() != 0 else np.nan)
    mean_r_net = R_net.mean()
    unresolved_pct = timeout.mean() * 100.0

    return dict(instrument=instrument, tf=tf, side=side, family=family, T=T, RR=RR,
                n=n, per_year=per_year, mean_win=mean_win, mean_loss=mean_loss,
                eff_rr=eff_rr, breakeven=breakeven, win_rate=win_rate, gap=gap,
                pf_gross=pf_gross, pf_net=pf_net, mean_r_net=mean_r_net,
                cost_r=cost_R.mean(), unresolved_pct=unresolved_pct,
                R_net=R_net, entry_time=tdf["entry_time"].values)


# --------------------------------------------------------------------------- driver
def run(instruments, smoke=False, median_t_only=False):
    rows_main, shape_rows, qualified_year_rows = [], [], []
    t0 = time.time()

    for inst in instruments:
        conf = INSTRUMENTS[inst]
        base = load_mt5_csv(conf["csv"]).loc[conf["start"]:]
        if smoke:
            base = base.iloc[: min(len(base), 60000)]
        cost = conf["cost"]
        t_list = [MEDIAN_T[inst]] if (smoke or median_t_only) else T_GRID[inst]
        tfs = ["15min"] if smoke else ["5min", "15min"]

        for tf in tfs:
            d = build_tf_frame(base, tf)
            ctx = build_daily_context(base, d.index)
            families = ["bb2", "breakout", "random"] if smoke else FAMILIES
            for side in SIDES:
                bo_n_by_T = {}
                for family in families:
                    sig_cache = None if family == "random" else family_signal(d, ctx, family, side)
                    for T in t_list:
                        rng = np.random.default_rng(
                            abs(hash((inst, tf, side, family, T, 20260714))) % (2**32))
                        gap_curve = {}
                        for RR in RR_GRID:
                            if family == "random":
                                target_n = bo_n_by_T.get(T, 200)
                                tdf, drop_b = walk_and_scan2(
                                    d, base, tf, None, side, T, RR, cost,
                                    rng=rng, random_target_n=target_n)
                            else:
                                tdf, drop_b = walk_and_scan2(
                                    d, base, tf, sig_cache, side, T, RR, cost)
                                if family == "breakout":
                                    bo_n_by_T[T] = len(tdf)

                            print(f"  [{inst} {tf} {side} {family} T={T} RR={RR}] "
                                  f"n={len(tdf)} dropped_boundary={drop_b}", file=sys.stderr)

                            s = summarize_cell2(tdf, RR, cost, inst, tf, side, family, T)
                            if s is None:
                                continue
                            gap_curve[RR] = s
                            rows_main.append({k: v for k, v in s.items()
                                               if k not in ("R_net", "entry_time")})

                        if not gap_curve:
                            continue
                        g05 = gap_curve.get(0.5, {}).get("gap", np.nan)
                        g10 = gap_curve.get(1.0, {}).get("gap", np.nan)
                        g30 = gap_curve.get(3.0, {}).get("gap", np.nan)
                        valid_gaps = {rr: s2["gap"] for rr, s2 in gap_curve.items()
                                      if np.isfinite(s2["gap"])}
                        if valid_gaps:
                            best_rr = max(valid_gaps, key=valid_gaps.get)
                            best = gap_curve[best_rr]
                        else:
                            best_rr, best = np.nan, {}

                        if all((not np.isfinite(s2["gap"])) or s2["gap"] <= 0
                               for s2 in gap_curve.values()):
                            shape = "dead"
                        elif np.isfinite(g05) and np.isfinite(g30) and g05 > 0 and g05 > g30:
                            shape = "near"
                        elif np.isfinite(g30) and np.isfinite(g05) and g30 > g05:
                            shape = "far"
                        else:
                            shape = "dead"

                        shape_rows.append(dict(
                            instrument=inst, tf=tf, side=side, family=family, T=T,
                            gap_rr05=g05, gap_rr10=g10, gap_rr30=g30, shape=shape,
                            best_rr=best_rr, best_pf_net=best.get("pf_net", np.nan),
                            best_per_year=best.get("per_year", np.nan)))

                        for RR, s2 in gap_curve.items():
                            if RR <= 1.5 and s2["n"] >= 100 and s2["per_year"] >= 50 \
                                    and np.isfinite(s2["pf_net"]) and s2["pf_net"] >= 1.5:
                                years = pd.DatetimeIndex(s2["entry_time"]).year
                                for y in sorted(set(years)):
                                    m = years == y
                                    Rn = s2["R_net"][m]
                                    pos = Rn[Rn > 0].sum()
                                    negabs = abs(Rn[Rn < 0].sum())
                                    pf_y = pos / negabs if negabs > 0 else np.nan
                                    qualified_year_rows.append(dict(
                                        instrument=inst, tf=tf, side=side, family=family,
                                        T=T, RR=RR, year=int(y), n=int(m.sum()),
                                        mean_r_net=Rn.mean(), pf_net=pf_y))
        print(f"  -- {inst} done, elapsed {time.time()-t0:.0f}s --", file=sys.stderr)

    return (pd.DataFrame(rows_main), pd.DataFrame(shape_rows),
            pd.DataFrame(qualified_year_rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--instrument", default=None, choices=list(INSTRUMENTS))
    ap.add_argument("--median-t-only", action="store_true",
                     help="use only the median T per instrument (perf fallback)")
    args = ap.parse_args()
    instruments = [args.instrument] if args.instrument else list(INSTRUMENTS)

    main_df, shape_df, year_df = run(instruments, smoke=args.smoke,
                                      median_t_only=args.median_t_only)

    suffix = "_smoke" if args.smoke else (f"_{args.instrument}" if args.instrument else "")
    main_path = f"scratchpad/gap_curve2_main{suffix}.csv"
    shape_path = f"scratchpad/gap_curve2_shape{suffix}.csv"
    year_path = f"scratchpad/gap_curve2_qualified_years{suffix}.csv"
    main_df.to_csv(main_path, index=False)
    shape_df.to_csv(shape_path, index=False)
    year_df.to_csv(year_path, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 400)
    print("\n=== TABLE 2 (shape) ===")
    print(shape_df.round(3).to_string(index=False))
    print(f"\nwritten: {main_path} ({len(main_df)} rows), {shape_path} ({len(shape_df)} rows), "
          f"{year_path} ({len(year_df)} rows)")

    print(f"\n=== TABLE 3: cells passing the pre-registered bar "
          f"(RR<=1.5, PF_net>=1.5, n>=100, n/yr>=50) ===")
    if len(year_df) == 0:
        print("0 cells")
    else:
        qual_cells = year_df[["instrument", "tf", "side", "family", "T", "RR"]].drop_duplicates()
        print(qual_cells.to_string(index=False))
        print("\nper-year detail:")
        print(year_df.round(3).to_string(index=False))

    breakout_shape = shape_df[shape_df["family"] == "breakout"]
    bad_bo = breakout_shape[breakout_shape["shape"] != "far"]
    print(f"\nbreakout control-group check: {len(breakout_shape)} cells, "
          f"{len(bad_bo)} NOT classified 'far' (should be 0)")
    if len(bad_bo):
        print(bad_bo.to_string(index=False))


if __name__ == "__main__":
    main()
