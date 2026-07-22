"""gap_curve_screen.py -- "near-field vs far-field" entry screen.

Spec card (frozen, verbatim from the measure task): test whether ANY of 7 candidate
entries has edge at LOW RR (0.1-1.0), where "edge" = GAP = win% - breakeven% (breakeven
computed from the REALIZED payoff, not the textbook 1/(1+RR), because the 3-day time-stop
mixes in R values that are neither +RR nor -1).

Reuses: src.data_loader.load_mt5_csv, breakout_wave.resample, breakout_wave.kama_adaptive.
pandas_ta for BB/ATR/RSI/EMA.

No lookahead: every signal is evaluated on a CONFIRMED bar [i] (bar i has fully closed);
entry fills at the open of bar [i+1]; daily-based gates/levels use shift(1)+ffill (the
PRIOR completed day only). Exit monitoring always steps 5-minute bar by 5-minute bar,
starting at the bar AFTER the entry bar (mirrors breakout_wave.py's own `for j in
range(e_bar+1, ...)` convention) -- same-bar stop+target conflict resolved as STOP-FIRST
(conservative). Max hold = 3 CALENDAR days; unresolved => flat at the last 5m bar's close
within the window.

Efficiency trick (as specified): entry/stop are fixed per (instrument,tf,side,family,k),
so ONE forward scan per accepted trade yields (a) whether/when the stop was hit and
(b) the max-favorable-excursion-BEFORE-the-stop-bar (MFE, in price units). Every RR is
then a vectorised threshold test against that single MFE: win iff MFE >= RR*stopdist.
Position-blocking ("don't stack signals of the same family while a position is open") is
therefore ALSO computed once, independent of RR, using the stop-or-3-day-timeout exit
event as the "trade over" marker (early RR-target-exit is NOT tracked per-RR for blocking
purposes -- this is required by the one-pass trick and is flagged in the report).

Run:
  .venv/bin/python experiments/gap_curve_screen.py --smoke
  .venv/bin/python experiments/gap_curve_screen.py --instrument gold
  .venv/bin/python experiments/gap_curve_screen.py            # full run, all instruments
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, ".")
from breakout_wave import resample, kama_adaptive
from src.data_loader import load_mt5_csv

RR_GRID = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0]
FAMILIES = ["bb2", "bb25", "dev2atr", "rsi", "run3", "sweep", "bb2_range", "breakout", "random"]
K_GRID = [1.0, 2.0]
SIDES = ["long", "short"]
MAX_HOLD_DAYS = 3
SEED = 20260714

INSTRUMENTS = {
    "gold":    dict(csv="data/vantage_xauusd_m5.csv", start="2018-09-14", cost=0.30),
    "btc":     dict(csv="data/vantage_btcusd_m5.csv", start="2019-01-01", cost=15.0),
    "usdjpy":  dict(csv="data/vantage_usdjpy_m5.csv", start="2000-01-01", cost=0.009),
}


# --------------------------------------------------------------------------- indicators
def build_tf_frame(base_df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """tf in {'5min','15min'}. Returns a df with OHLC + all indicators needed by the
    families, computed ONLY from confirmed bars up to and including bar i."""
    d = base_df if tf == "5min" else resample(base_df, "15min")
    d = d.copy()
    d["ema20"] = ta.ema(d["close"], length=20)
    d["atr14"] = ta.atr(d["high"], d["low"], d["close"], length=14)
    d["rsi14"] = ta.rsi(d["close"], length=14)
    bb2 = ta.bbands(d["close"], length=20, lower_std=2.0, upper_std=2.0)
    d["bbl2"] = bb2.iloc[:, 0].values
    d["bbu2"] = bb2.iloc[:, 2].values
    bb25 = ta.bbands(d["close"], length=20, lower_std=2.5, upper_std=2.5)
    d["bbl25"] = bb25.iloc[:, 0].values
    d["bbu25"] = bb25.iloc[:, 2].values
    d["bearish"] = d["close"] < d["open"]
    d["bullish"] = d["close"] > d["open"]
    # donchian(20) breakout contrast, prior 20 bars EXCLUDING current bar
    d["dc_hi20"] = d["high"].shift(1).rolling(20).max()
    d["dc_lo20"] = d["low"].shift(1).rolling(20).min()
    return d


def build_daily_context(base_df: pd.DataFrame, tf_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Prior-completed-day high/low (family 6 sweep) and the range-day flag (family 7),
    all shift(1) then ffill onto the intraday TF index -- no lookahead (matches the
    daily_sma / gate_kama convention in breakout_wave.run())."""
    dd = resample(base_df, "1D")
    prior_hi = dd["high"].shift(1).reindex(tf_index, method="ffill")
    prior_lo = dd["low"].shift(1).reindex(tf_index, method="ffill")
    d_atr = ta.atr(dd["high"], dd["low"], dd["close"], length=14)
    d_kama = kama_adaptive(dd["close"], 14)
    range_day = ((d_kama - d_kama.shift(1)).abs() / d_atr) < 0.1
    range_flag = range_day.shift(1).reindex(tf_index, method="ffill").fillna(False)
    return pd.DataFrame({"prior_hi": prior_hi, "prior_lo": prior_lo, "range_flag": range_flag})


# --------------------------------------------------------------------------- signals
def family_signal(d: pd.DataFrame, ctx: pd.DataFrame, family: str, side: str) -> np.ndarray:
    """Boolean array aligned to d.index: True at bar i means the family fired
    (confirmed at bar i's close); entry is opened at open[i+1]."""
    long_ = side == "long"
    if family == "bb2":
        sig = (d["low"] <= d["bbl2"]) if long_ else (d["high"] >= d["bbu2"])
    elif family == "bb25":
        sig = (d["low"] <= d["bbl25"]) if long_ else (d["high"] >= d["bbu25"])
    elif family == "dev2atr":
        dev = (d["close"] - d["ema20"]).abs() >= 2.0 * d["atr14"]
        sig = dev & (d["close"] < d["ema20"]) if long_ else dev & (d["close"] > d["ema20"])
    elif family == "rsi":
        sig = (d["rsi14"] < 30) if long_ else (d["rsi14"] > 70)
    elif family in ("run3",):
        b3 = d["bearish"] & d["bearish"].shift(1) & d["bearish"].shift(2)
        u3 = d["bullish"] & d["bullish"].shift(1) & d["bullish"].shift(2)
        sig = b3 if long_ else u3
    elif family == "sweep":
        if long_:
            sig = (d["low"] < ctx["prior_lo"]) & (d["close"] > ctx["prior_lo"])
        else:
            sig = (d["high"] > ctx["prior_hi"]) & (d["close"] < ctx["prior_hi"])
    elif family == "bb2_range":
        base = (d["low"] <= d["bbl2"]) if long_ else (d["high"] >= d["bbu2"])
        sig = base & ctx["range_flag"]
    elif family == "breakout":
        sig = (d["close"] > d["dc_hi20"]) if long_ else (d["close"] < d["dc_lo20"])
    else:
        raise ValueError(family)
    return sig.fillna(False).values


# --------------------------------------------------------------------------- simulation
def walk_and_scan(d: pd.DataFrame, base_df: pd.DataFrame, tf: str, sig: np.ndarray,
                   side: str, k: float, cost: float, rng: np.random.Generator = None,
                   random_target_n: int = None):
    """One forward pass: apply position-blocking over `sig` (or draw a random null with
    `random_target_n` candidates if sig is None), and for every accepted entry compute
    (stopped, mfe_price, timeout_R). Returns a DataFrame of accepted trades (pre-RR)."""
    idx = d.index
    n_bars = len(d)
    atr = d["atr14"].values
    op = d["open"].values
    base_idx_arr = base_df.index
    base_low = base_df["low"].values
    base_high = base_df["high"].values
    base_close = base_df["close"].values
    base_time = base_idx_arr.values  # numpy datetime64

    if sig is None:  # random null: draw candidates from bars with valid ATR
        eligible = np.where(~np.isnan(atr) & (np.arange(n_bars) < n_bars - 1))[0]
        m = min(random_target_n, len(eligible))
        cand = rng.choice(eligible, size=m, replace=False) if m > 0 else np.array([], dtype=int)
        sig_bool = np.zeros(n_bars, dtype=bool)
        sig_bool[cand] = True
        sig = sig_bool

    fire_idx = np.where(sig)[0]
    trades = []
    busy_until = None  # np.datetime64, exit time of the currently open position
    dropped_boundary = 0
    dropped_atr = 0

    for i in fire_idx:
        if i + 1 >= n_bars:
            continue
        entry_time = idx[i + 1]
        if busy_until is not None and entry_time <= busy_until:
            continue
        sd = k * atr[i]
        if not np.isfinite(sd) or sd <= 0:
            dropped_atr += 1
            continue
        entry_price = op[i + 1]

        # map entry_time -> base 5m index
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
            stop_price = entry_price - sd
            stop_hit = lo <= stop_price
            fav = hi - entry_price
        else:
            stop_price = entry_price + sd
            stop_hit = hi >= stop_price
            fav = entry_price - lo

        hit_pos = np.argmax(stop_hit) if stop_hit.any() else -1
        if hit_pos >= 0:
            mfe = fav[:hit_pos].max() if hit_pos > 0 else 0.0
            stopped = True
            exit_time = base_time[start + hit_pos]
            timeout_R = np.nan
        else:
            mfe = fav.max()
            stopped = False
            exit_time = base_time[cend - 1]
            exit_close = cl[-1]
            timeout_R = ((exit_close - entry_price) / sd) if side == "long" else \
                        ((entry_price - exit_close) / sd)

        trades.append((entry_time, side, k, sd, stopped, max(mfe, 0.0) / sd, timeout_R))
        busy_until = pd.Timestamp(exit_time).tz_localize("UTC")

    cols = ["entry_time", "side", "k", "stopdist", "stopped", "mfe_R", "timeout_R"]
    tdf = pd.DataFrame(trades, columns=cols)
    return tdf, dropped_boundary, dropped_atr


def rr_stats(tdf: pd.DataFrame, RR: float, cost: float):
    """Vectorised per-RR outcome from the fixed (stopped, mfe_R, timeout_R) trade set."""
    win = tdf["mfe_R"].values >= RR
    stopped = tdf["stopped"].values
    timeout_R = tdf["timeout_R"].values
    R_gross = np.where(win, RR, np.where(stopped, -1.0, timeout_R))
    R_net = R_gross - cost / tdf["stopdist"].values
    unresolved = (~win) & (~stopped)  # 3-day timeout rate for THIS RR
    return R_gross, R_net, win, unresolved


def summarize_cell(tdf: pd.DataFrame, RR: float, cost: float, instrument: str, tf: str,
                    side: str, family: str, k: float):
    n = len(tdf)
    if n == 0:
        return None
    R_gross, R_net, win, unresolved = rr_stats(tdf, RR, cost)
    span_days = (tdf["entry_time"].max() - tdf["entry_time"].min()).days
    yrs = max(span_days / 365.25, 0.5)
    per_year = n / yrs

    pos_mask = R_gross > 0
    neg_mask = R_gross < 0
    mean_win = R_gross[pos_mask].mean() if pos_mask.any() else np.nan
    mean_loss = R_gross[neg_mask].mean() if neg_mask.any() else np.nan  # negative
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
    unresolved_pct = unresolved.mean() * 100.0

    return dict(instrument=instrument, tf=tf, side=side, family=family, k=k, RR=RR,
                n=n, per_year=per_year, mean_win=mean_win, mean_loss=mean_loss,
                eff_rr=eff_rr, breakeven=breakeven, win_rate=win_rate, gap=gap,
                pf_gross=pf_gross, pf_net=pf_net, mean_r_net=mean_r_net,
                unresolved_pct=unresolved_pct, R_net=R_net, entry_time=tdf["entry_time"].values)


# --------------------------------------------------------------------------- driver
def run(instruments, smoke=False):
    rows_main = []
    shape_rows = []
    qualified_year_rows = []
    t0 = time.time()

    for inst in instruments:
        conf = INSTRUMENTS[inst]
        base = load_mt5_csv(conf["csv"]).loc[conf["start"]:]
        if smoke:
            base = base.iloc[: min(len(base), 60000)]
        cost = conf["cost"]
        tfs = ["15min"] if smoke else ["5min", "15min"]
        for tf in tfs:
            d = build_tf_frame(base, tf)
            ctx = build_daily_context(base, d.index)
            families = ["bb2", "breakout", "random"] if smoke else FAMILIES
            ks = [2.0] if smoke else K_GRID
            sides = SIDES
            for side in sides:
                # cache breakout's accepted-n per k to feed the random null
                bo_n_by_k = {}
                for family in families:
                    for k in ks:
                        rng = np.random.default_rng(
                            abs(hash((inst, tf, side, k, SEED))) % (2**32))
                        if family == "random":
                            target_n = bo_n_by_k.get(k, 200)
                            tdf, drop_b, drop_a = walk_and_scan(
                                d, base, tf, None, side, k, cost, rng=rng,
                                random_target_n=target_n)
                        else:
                            sig = family_signal(d, ctx, family, side)
                            tdf, drop_b, drop_a = walk_and_scan(
                                d, base, tf, sig, side, k, cost)
                            if family == "breakout":
                                bo_n_by_k[k] = len(tdf)

                        print(f"  [{inst} {tf} {side} {family} k={k}] "
                              f"n={len(tdf)} dropped(boundary={drop_b},atr={drop_a})",
                              file=sys.stderr)

                        gap_curve = {}
                        for RR in RR_GRID:
                            s = summarize_cell(tdf, RR, cost, inst, tf, side, family, k)
                            if s is None:
                                continue
                            gap_curve[RR] = s
                            rows_main.append({kk: vv for kk, vv in s.items()
                                               if kk not in ("R_net", "entry_time")})

                        if not gap_curve:
                            continue
                        g05 = gap_curve.get(0.5, {}).get("gap", np.nan)
                        g10 = gap_curve.get(1.0, {}).get("gap", np.nan)
                        g30 = gap_curve.get(3.0, {}).get("gap", np.nan)
                        valid_gaps = {rr: s["gap"] for rr, s in gap_curve.items()
                                      if np.isfinite(s["gap"])}
                        if valid_gaps:
                            best_rr = max(valid_gaps, key=valid_gaps.get)
                            best = gap_curve[best_rr]
                        else:
                            best_rr, best = np.nan, {}
                        if not (np.isfinite(g05) or np.isfinite(g30)):
                            shape = "dead"
                        elif all((not np.isfinite(s["gap"])) or s["gap"] <= 0
                                  for s in gap_curve.values()):
                            shape = "dead"
                        elif np.isfinite(g05) and np.isfinite(g30) and g05 > 0 and g05 > g30:
                            shape = "near"
                        elif np.isfinite(g30) and np.isfinite(g05) and g30 > g05:
                            shape = "far"
                        else:
                            shape = "dead"
                        shape_rows.append(dict(
                            instrument=inst, tf=tf, side=side, family=family, k=k,
                            gap_rr05=g05, gap_rr10=g10, gap_rr30=g30, shape=shape,
                            best_rr=best_rr, best_pf_net=best.get("pf_net", np.nan),
                            best_per_year=best.get("per_year", np.nan)))

                        # table3 qualification: RR<=1.5, pf_net>=1.5, n>=100, per_year>=20
                        for RR, s in gap_curve.items():
                            if RR <= 1.5 and s["n"] >= 100 and s["per_year"] >= 20 \
                                    and np.isfinite(s["pf_net"]) and s["pf_net"] >= 1.5:
                                years = pd.DatetimeIndex(s["entry_time"]).year
                                for y in sorted(set(years)):
                                    m = years == y
                                    Rn = s["R_net"][m]
                                    pos = Rn[Rn > 0].sum()
                                    negabs = abs(Rn[Rn < 0].sum())
                                    pf_y = pos / negabs if negabs > 0 else np.nan
                                    qualified_year_rows.append(dict(
                                        instrument=inst, tf=tf, side=side, family=family,
                                        k=k, RR=RR, year=int(y), n=int(m.sum()),
                                        mean_r_net=Rn.mean(), pf_net=pf_y))
        print(f"  -- {inst} done, elapsed {time.time()-t0:.0f}s --", file=sys.stderr)

    return (pd.DataFrame(rows_main), pd.DataFrame(shape_rows),
            pd.DataFrame(qualified_year_rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--instrument", default=None, choices=list(INSTRUMENTS))
    args = ap.parse_args()
    instruments = [args.instrument] if args.instrument else list(INSTRUMENTS)

    main_df, shape_df, year_df = run(instruments, smoke=args.smoke)

    suffix = "_smoke" if args.smoke else (f"_{args.instrument}" if args.instrument else "")
    main_path = f"experiments/gap_curve_main{suffix}.csv"
    shape_path = f"experiments/gap_curve_shape{suffix}.csv"
    year_path = f"experiments/gap_curve_qualified_years{suffix}.csv"
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

    n_qual_cells = shape_df.assign(
        qual=False).shape[0]  # placeholder, real check below
    print(f"\n=== TABLE 3: cells passing the pre-registered bar "
          f"(RR<=1.5, PF_net>=1.5, n>=100, n/yr>=20) ===")
    if len(year_df) == 0:
        print("0 cells")
    else:
        qual_cells = year_df[["instrument", "tf", "side", "family", "k", "RR"]].drop_duplicates()
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
