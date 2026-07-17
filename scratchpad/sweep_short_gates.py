"""sweep_short_gates.py -- frozen spec-card measurement: BTC 15m SHORT, "PDH sweep /
false break" (high[i] > PDH, close[i] < PDH -> sell at [i+1] open). Fills walked bar-by-
bar on 5m data. Two stop modes (absolute $ / structural high+0.25*ATR), RR grid, 4 single
regime gates (+none) + 2-way combos with the best single gate, structural filters F1/F2/F3,
pre-registered pass bar, random-drop null, per-year table.

REUSE (no reinvention): src.data_loader.load_mt5_csv, breakout_wave.kama_adaptive,
research.portfolio_alloc.cagr_dd_trades. pandas_ta for ATR/SMA.

NO-LOOKAHEAD:
  - PDH/PDL: d15.high/low.resample("1D").max()/min(), shift(1), reindex(ffill) onto the
    15m index -- so the value in effect during day D is day D-1's completed high/low.
  - Entry pattern read off CONFIRMED bar [i] (its own close). Order = market at [i+1] open.
  - Gates (G1 daily KAMA14, G2 4h KAMA14, G3 daily SMA150, G4 weekly close vs 30w MA): each
    computed on its own HTF close series, boolean flag SHIFTED by 1 period BEFORE being
    reindexed(ffill) onto the 15m index -- i.e. the flag in effect at bar i reflects the
    prior COMPLETED HTF period only, never the one bar i sits inside.
  - Fill walk: 5m bars starting at the entry bar itself (its own open IS the entry price,
    so its own high/low occur at/after the fill -- safe) through entry_time+10d. Same-bar
    stop+target conflict -> stop wins (idx_stop <= idx_tgt tie broken toward stop).
  - Position-exclusivity ("hold中は新規に建てない") is evaluated PER (stop-config, RR, gate,
    filter) combination separately via a sequential busy_until scan over that combo's own
    accepted candidates -- it is combo-specific because exit time (hence whether a later
    candidate is "already in a position") depends on the stop/RR/gate/filter being tested.

PERFORMANCE DESIGN: for a fixed entry, the forward favourable/adverse excursion arrays
(fav_cum, adv_cum -- running cumulative max, monotonic non-decreasing) depend only on
entry_price, NOT on stop-size or RR. They are built ONCE per raw candidate (the expensive
5m-slice step) and then reused for all 4 stop-configs x 6 RR = 24 (S,T) pairs via
np.searchsorted (each O(log window) once monotonic). This turns an O(candidates x windows x
combos) walk into O(candidates x windows) + O(candidates x combos x log(window)).

Run:
  .venv/bin/python scratchpad/sweep_short_gates.py --smoke 2>&1 | tee scratchpad/out_sweep_short_gates_smoke.txt
  .venv/bin/python scratchpad/sweep_short_gates.py         2>&1 | tee scratchpad/out_sweep_short_gates.txt
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, ".")
from src.data_loader import load_mt5_csv
from breakout_wave import kama_adaptive
from research.portfolio_alloc import cagr_dd_trades

START = "2018-10-01"
RR_GRID = [1.0, 1.5, 2.0, 3.0, 4.5, 6.0]
ABS_STOPS = [500.0, 750.0, 1000.0]
COST = 15.0
MAX_HOLD_DAYS = 10
RISK_PCT = 0.01          # for CAGR/DD (table 4), per CLAUDE.md 1% default
ANCHOR_STOP, ANCHOR_RR = "S750", 1.5   # reference cell for "best single gate" selection
                                        # (matches the background screen's flagged best cell
                                        # T=$750/RR1.5 -- see spec background section)
F2_THRESH = [0.0, 0.25, 0.5, 1.0]
F3_THRESH = [0.0, 0.3, 0.5, 0.7]

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 500)


# --------------------------------------------------------------------------- data / gates
def load_data(smoke: bool):
    d15 = load_mt5_csv("data/vantage_btcusd_m15.csv").loc[START:]
    d5 = load_mt5_csv("data/vantage_btcusd_m5.csv").loc[START:]
    if smoke:
        d15 = d15.loc[:"2020-12-31"]
        d5 = d5.loc[:"2021-01-10"]  # a little extra tail so 10d holds near the cutoff resolve
    return d15, d5


def build_gates(d15: pd.DataFrame) -> dict:
    idx = d15.index
    close = d15["close"]

    dc = close.resample("1D").last().dropna()
    k1 = kama_adaptive(dc, 14)
    g1 = (k1 < k1.shift(1)).shift(1)

    dc4 = close.resample("4h").last().dropna()
    k2 = kama_adaptive(dc4, 14)
    g2 = (k2 < k2.shift(1)).shift(1)

    sma150 = dc.rolling(150).mean()
    g3 = (sma150 < sma150.shift(1)).shift(1)

    w1 = load_mt5_csv("data/vantage_btcusd_w1.csv")
    wsma30 = w1["close"].rolling(30).mean()
    g4 = (w1["close"] < wsma30).shift(1)

    gates = {
        "G1_dKAMA14dn": g1.reindex(idx, method="ffill").fillna(False).infer_objects(copy=False).values,
        "G2_4hKAMA14dn": g2.reindex(idx, method="ffill").fillna(False).infer_objects(copy=False).values,
        "G3_dSMA150dn": g3.reindex(idx, method="ffill").fillna(False).infer_objects(copy=False).values,
        "G4_wClose_lt_30wMA": g4.reindex(idx, method="ffill").fillna(False).infer_objects(copy=False).values,
    }
    return gates


# --------------------------------------------------------------------------- candidates
class Candidates:
    """One row per raw PDH-sweep signal (gate/filter-independent). Holds, per candidate,
    the cached fav_cum/adv_cum/close/time 5m-forward arrays plus everything needed for
    gates and structural filters, computed ONCE."""

    def __init__(self, d15, d5, gates):
        h, l, c, o = (d15["high"].values, d15["low"].values,
                      d15["close"].values, d15["open"].values)
        atr = ta.atr(d15["high"], d15["low"], d15["close"], length=14).values
        dh = d15["high"].resample("1D").max()
        dl = d15["low"].resample("1D").min()
        pdh = dh.shift(1).reindex(d15.index, method="ffill").values
        pdl = dl.shift(1).reindex(d15.index, method="ffill").values

        n = len(d15)
        sig = (h > pdh) & (c < pdh) & ~np.isnan(pdh) & ~np.isnan(atr) & (atr > 0)
        raw_idx = np.where(sig)[0]
        raw_idx = raw_idx[raw_idx + 1 < n]

        d5idx = d5.index
        d5low, d5high, d5close = d5["low"].values, d5["high"].values, d5["close"].values
        d5time = d5.index.values

        rows = []
        n_boundary_drop = 0
        for i in raw_idx:
            entry_time = d15.index[i + 1]
            entry_price = o[i + 1]
            eb = d5idx.searchsorted(entry_time, side="left")
            cutoff = entry_time + pd.Timedelta(days=MAX_HOLD_DAYS)
            cend = d5idx.searchsorted(cutoff, side="right")
            if eb >= cend or eb >= len(d5idx):
                n_boundary_drop += 1
                continue
            lo, hi, cl, tt = d5low[eb:cend], d5high[eb:cend], d5close[eb:cend], d5time[eb:cend]
            fav = np.clip(entry_price - lo, 0.0, None)
            adv = np.clip(hi - entry_price, 0.0, None)
            fav_cum = np.maximum.accumulate(fav)
            adv_cum = np.maximum.accumulate(adv)

            atr_i = atr[i]
            s_struct = (h[i] + 0.25 * atr_i) - entry_price
            excl_struct = (s_struct < 0.5 * atr_i) or (s_struct <= 0)

            rng = h[i] - l[i]
            f3 = ((h[i] - max(o[i], c[i])) / rng) if rng > 0 else np.nan
            f2 = (h[i] - pdh[i]) / atr_i

            rows.append(dict(
                i=i, entry_time=entry_time, entry_price=entry_price,
                atr_i=atr_i, s_struct=s_struct, excl_struct=excl_struct,
                f1a_low_lt_pdl=bool(l[i] < pdl[i]), f1b_close_lt_pdl=bool(c[i] < pdl[i]),
                f2_sweep_atr=f2, f3_wick_ratio=f3,
                fav_cum=fav_cum, adv_cum=adv_cum, cl=cl, tt=tt,
                G1=bool(gates["G1_dKAMA14dn"][i]), G2=bool(gates["G2_4hKAMA14dn"][i]),
                G3=bool(gates["G3_dSMA150dn"][i]), G4=bool(gates["G4_wClose_lt_30wMA"][i]),
            ))

        self.df = pd.DataFrame(rows)
        self.n_raw = len(raw_idx)
        self.n_boundary_drop = n_boundary_drop
        print(f"  [candidates] raw signals={self.n_raw}  usable(with exit data)={len(self.df)}  "
              f"boundary_drop={n_boundary_drop}", file=sys.stderr)


def resolve_one(fav_cum, adv_cum, cl, entry_price, S, RR):
    T = RR * S
    nwin = len(adv_cum)
    idx_stop = np.searchsorted(adv_cum, S, side="left")
    idx_tgt = np.searchsorted(fav_cum, T, side="left")
    if idx_stop < nwin and (idx_tgt >= nwin or idx_stop <= idx_tgt):
        outcome, exit_pos, R_gross = "loss", idx_stop, -1.0
    elif idx_tgt < nwin:
        outcome, exit_pos, R_gross = "win", idx_tgt, RR
    else:
        outcome, exit_pos = "timeout", nwin - 1
        R_gross = (entry_price - cl[exit_pos]) / S
    return outcome, exit_pos, R_gross


def compute_stop_rr(cand: Candidates, stopcfg: str, RR: float) -> pd.DataFrame:
    """Return a DataFrame aligned with cand.df giving S, outcome, exit_time, R_gross,
    R_net, hold_days, exclude (bool) for this (stopcfg, RR)."""
    df = cand.df
    if stopcfg == "struct":
        S_arr = df["s_struct"].values
        excl = df["excl_struct"].values
    else:
        S_arr = np.full(len(df), float(stopcfg[1:]))
        excl = np.zeros(len(df), dtype=bool)

    outcomes, exit_times, R_gross, hold_days = [], [], [], []
    for row, S in zip(df.itertuples(index=False), S_arr):
        outcome, exit_pos, rg = resolve_one(row.fav_cum, row.adv_cum, row.cl,
                                             row.entry_price, S, RR)
        et = pd.Timestamp(row.tt[exit_pos]).tz_localize("UTC")
        outcomes.append(outcome); exit_times.append(et); R_gross.append(rg)
        hold_days.append((et - row.entry_time) / pd.Timedelta(days=1))

    R_gross = np.array(R_gross)
    R_net = R_gross - COST / S_arr
    out = pd.DataFrame({
        "i": df["i"].values, "entry_time": df["entry_time"].values,
        "exit_time": exit_times, "S": S_arr, "outcome": outcomes,
        "R_gross": R_gross, "R_net": R_net, "hold_days": hold_days, "exclude": excl,
        "G1": df["G1"].values, "G2": df["G2"].values, "G3": df["G3"].values, "G4": df["G4"].values,
        "f1a_low_lt_pdl": df["f1a_low_lt_pdl"].values, "f1b_close_lt_pdl": df["f1b_close_lt_pdl"].values,
        "f2_sweep_atr": df["f2_sweep_atr"].values, "f3_wick_ratio": df["f3_wick_ratio"].values,
    })
    return out


# --------------------------------------------------------------------------- busy_until + metrics
def sequential_accept(sub: pd.DataFrame) -> pd.DataFrame:
    """sub already time-sorted & already filtered (gate/struct-exclude/structural-filter).
    Apply 'no new entry while holding' sequentially and return the accepted rows."""
    sub = sub.sort_values("entry_time")
    accepted_mask = np.zeros(len(sub), dtype=bool)
    busy_until = None
    et = sub["entry_time"].values
    xt = sub["exit_time"].values
    for j in range(len(sub)):
        if busy_until is not None and et[j] < busy_until:
            continue
        accepted_mask[j] = True
        busy_until = xt[j]
    return sub.iloc[accepted_mask]


def summarize(acc: pd.DataFrame, full_start: pd.Timestamp, full_end: pd.Timestamp) -> dict:
    n = len(acc)
    total_years = (full_end - full_start) / pd.Timedelta(days=365.25)
    if n == 0:
        return dict(n=0, per_year=0.0, win_pct=np.nan, breakeven_pct=np.nan, gap=np.nan,
                    pf_gross=np.nan, pf_net=np.nan, mean_r_net=np.nan,
                    is_mean_r=np.nan, oos_mean_r=np.nan, maxdd_r=np.nan, unresolved_pct=np.nan)
    Rg = acc["R_gross"].values
    Rn = acc["R_net"].values
    win = Rg > 0
    mean_win = Rg[win].mean() if win.any() else np.nan
    mean_loss = Rg[~win].mean() if (~win).any() else np.nan
    eff_rr = mean_win / abs(mean_loss) if (win.any() and (~win).any() and mean_loss != 0) else np.nan
    breakeven = 100.0 / (1.0 + eff_rr) if np.isfinite(eff_rr) else np.nan
    win_pct = win.mean() * 100.0
    gap = win_pct - breakeven if np.isfinite(breakeven) else np.nan

    pos_g, neg_g = Rg > 0, Rg < 0
    pf_gross = (Rg[pos_g].sum() / abs(Rg[neg_g].sum())) if neg_g.any() and Rg[neg_g].sum() != 0 else np.nan
    pos_n, neg_n = Rn > 0, Rn < 0
    pf_net = (Rn[pos_n].sum() / abs(Rn[neg_n].sum())) if neg_n.any() and Rn[neg_n].sum() != 0 else np.nan

    mid = full_start + (full_end - full_start) / 2
    mid_naive = pd.Timestamp(mid).tz_localize(None)  # acc["entry_time"] loses tz via .values
    is_mask = pd.DatetimeIndex(acc["entry_time"]) < mid_naive
    is_mean = Rn[is_mask].mean() if is_mask.any() else np.nan
    oos_mean = Rn[~is_mask].mean() if (~is_mask).any() else np.nan

    eq = np.concatenate([[0.0], np.cumsum(Rn)])
    peak = np.maximum.accumulate(eq)
    maxdd_r = (peak - eq).max()

    unresolved_pct = (acc["outcome"] == "timeout").mean() * 100.0

    return dict(n=n, per_year=n / total_years, win_pct=win_pct, breakeven_pct=breakeven,
                gap=gap, pf_gross=pf_gross, pf_net=pf_net, mean_r_net=Rn.mean(),
                is_mean_r=is_mean, oos_mean_r=oos_mean, maxdd_r=maxdd_r,
                unresolved_pct=unresolved_pct, acc=acc)


def run_combo(precomp: dict, stopcfg: str, RR: float, gate_mask: np.ndarray,
              extra_filter: np.ndarray, full_start, full_end) -> dict:
    df = precomp[(stopcfg, RR)]
    m = gate_mask & ~df["exclude"].values
    if extra_filter is not None:
        m = m & extra_filter
    sub = df[m]
    acc = sequential_accept(sub)
    return summarize(acc, full_start, full_end)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    d15, d5 = load_data(args.smoke)
    full_start, full_end = d15.index[0], d15.index[-1]
    print(f"d15: {len(d15)} bars {full_start}..{full_end}   d5: {len(d5)} bars "
          f"{d5.index[0]}..{d5.index[-1]}", file=sys.stderr)

    gates = build_gates(d15)
    cand = Candidates(d15, d5, gates)
    df = cand.df
    n_cand = len(df)

    stop_cfgs = ["S500", "S750", "S1000", "struct"]
    print(f"\nprecomputing {len(stop_cfgs)} stops x {len(RR_GRID)} RR = "
          f"{len(stop_cfgs)*len(RR_GRID)} cells over {n_cand} candidates...", file=sys.stderr)
    precomp = {}
    for sc in stop_cfgs:
        for rr in RR_GRID:
            precomp[(sc, rr)] = compute_stop_rr(cand, sc, rr)
    print(f"  precompute done, {time.time()-t0:.0f}s elapsed", file=sys.stderr)

    none_mask = np.ones(n_cand, dtype=bool)
    single_gate_masks = {
        "NONE": none_mask,
        "G1": df["G1"].values, "G2": df["G2"].values,
        "G3": df["G3"].values, "G4": df["G4"].values,
    }

    # ---- pick best single gate at the anchor cell (S750, RR1.5) by net PF ----
    anchor_scores = {}
    for name, gm in single_gate_masks.items():
        if name == "NONE":
            continue
        s = run_combo(precomp, ANCHOR_STOP, ANCHOR_RR, gm, None, full_start, full_end)
        anchor_scores[name] = (s["pf_net"] if np.isfinite(s["pf_net"]) else -1, s["n"])
    best_gate_name = max(anchor_scores, key=lambda k: anchor_scores[k][0])
    print(f"\nanchor cell ({ANCHOR_STOP}, RR={ANCHOR_RR}) single-gate net PF: {anchor_scores}")
    print(f"-> best single gate = {best_gate_name}")

    other_gates = [g for g in ["G1", "G2", "G3", "G4"] if g != best_gate_name]
    combo_masks = dict(single_gate_masks)
    for og in other_gates:
        combo_masks[f"{best_gate_name}&{og}"] = single_gate_masks[best_gate_name] & single_gate_masks[og]

    # ================================================================ TABLE 1
    rows1 = []
    for stopcfg in stop_cfgs:
        for rr in RR_GRID:
            for gname, gm in combo_masks.items():
                s = run_combo(precomp, stopcfg, rr, gm, None, full_start, full_end)
                rows1.append(dict(stop=stopcfg, RR=rr, gate=gname, **{k: v for k, v in s.items() if k != "acc"}))
    t1 = pd.DataFrame(rows1)
    t1.to_csv("scratchpad/sweep_short_gates_table1.csv", index=False)
    t1.to_csv("scratchpad/sweep_short_gates.csv", index=False)  # spec-named primary CSV (= table 1)
    print("\n=== TABLE 1: stop x RR x gate ===")
    print(t1.round(4).to_string(index=False))

    # ---- structural stop exclusion rate (reported once, RR-independent) ----
    excl_rate = precomp[("struct", RR_GRID[0])]["exclude"].mean() * 100.0
    print(f"\n[structural stop] exclusion rate (S < 0.5*ATR14): {excl_rate:.2f}% of candidates")

    # ================================================================ TABLE 2
    # fix (stopcfg, RR) = the best net-PF cell in table1 restricted to the best single gate,
    # then apply F1/F2/F3 on top of it.
    t1_bestgate = t1[t1["gate"] == best_gate_name].copy()
    t1_bestgate = t1_bestgate[t1_bestgate["n"] >= 10]
    ref_row = t1_bestgate.loc[t1_bestgate["pf_net"].idxmax()]
    ref_stop, ref_rr = ref_row["stop"], ref_row["RR"]
    print(f"\n[table 2 reference] best_gate={best_gate_name}  stop={ref_stop}  RR={ref_rr}  "
          f"(pf_net={ref_row['pf_net']:.3f}, n={ref_row['n']:.0f})")

    ref_df = precomp[(ref_stop, ref_rr)]
    ref_gate_mask = combo_masks[best_gate_name]

    rows2 = []

    def add_t2(label, thr, extra_filter):
        s = run_combo(precomp, ref_stop, ref_rr, ref_gate_mask, extra_filter, full_start, full_end)
        rows2.append(dict(filter=label, threshold=thr, n=s["n"], per_year=s["per_year"],
                           pf_net=s["pf_net"], mean_r_net=s["mean_r_net"],
                           totR_per_year=(s["mean_r_net"] * s["per_year"] if s["n"] else np.nan),
                           is_mean_r=s["is_mean_r"], oos_mean_r=s["oos_mean_r"]))
        return s

    add_t2("baseline(best_gate only)", "-", None)
    add_t2("F1_low<PDL", "on", ref_df["f1a_low_lt_pdl"].values)
    add_t2("F1_close<PDL", "on", ref_df["f1b_close_lt_pdl"].values)
    for th in F2_THRESH:
        add_t2("F2_sweep_depth_atr", f">={th}", ref_df["f2_sweep_atr"].values >= th)
    for th in F3_THRESH:
        add_t2("F3_wick_ratio", f">={th}", ref_df["f3_wick_ratio"].values >= th)

    t2 = pd.DataFrame(rows2)
    t2.to_csv("scratchpad/sweep_short_gates_table2.csv", index=False)
    print("\n=== TABLE 2: structural filters on top of best-gate reference "
          f"(stop={ref_stop}, RR={ref_rr}) ===")
    print(t2.round(4).to_string(index=False))

    # ================================================================ TABLE 3
    # scan table1 (all stop x RR x gate) + table2 (best_gate + each filter) for the pass bar
    cand3 = []
    for _, r in t1.iterrows():
        cand3.append(dict(source="table1", stop=r["stop"], RR=r["RR"], gate=r["gate"],
                           filter="-", n=r["n"], per_year=r["per_year"], pf_net=r["pf_net"],
                           is_mean_r=r["is_mean_r"], oos_mean_r=r["oos_mean_r"]))
    for _, r in t2.iterrows():
        cand3.append(dict(source="table2", stop=ref_stop, RR=ref_rr, gate=best_gate_name,
                           filter=f"{r['filter']}{r['threshold']}", n=r["n"], per_year=r["per_year"],
                           pf_net=r["pf_net"], is_mean_r=r["is_mean_r"], oos_mean_r=r["oos_mean_r"]))
    c3 = pd.DataFrame(cand3)
    pass_mask = ((c3["pf_net"] >= 1.8) & (c3["per_year"] >= 30) &
                 (c3["is_mean_r"] > 0) & (c3["oos_mean_r"] > 0))
    t3 = c3[pass_mask].copy()
    t3.to_csv("scratchpad/sweep_short_gates_table3.csv", index=False)
    print(f"\n=== TABLE 3: configs passing net PF>=1.8, n/yr>=30, IS&OOS both positive ===")
    if len(t3) == 0:
        print("0件")
    else:
        print(t3.round(4).to_string(index=False))

    # ================================================================ TABLE 4
    print("\n=== TABLE 4: random-drop null for configs passing table 3 ===")
    if len(t3) == 0:
        print("該当なし（表3が0件のため）")
    else:
        rng = np.random.default_rng(20260714)
        rows4 = []
        for _, r in t3.iterrows():
            stopcfg, rr, gname, filt = r["stop"], r["RR"], r["gate"], r["filter"]
            base_df = precomp[(stopcfg, rr)]
            gate_mask = combo_masks[gname]
            if r["source"] == "table1":
                extra = None
            else:
                # reconstruct the same filter array used in table 2 for this label
                lbl, thr = filt.split(">=") if ">=" in filt else (filt, None)
                if filt.startswith("F1_low<PDL"):
                    extra = ref_df["f1a_low_lt_pdl"].values
                elif filt.startswith("F1_close<PDL"):
                    extra = ref_df["f1b_close_lt_pdl"].values
                elif filt.startswith("F2_sweep_depth_atr"):
                    extra = ref_df["f2_sweep_atr"].values >= float(thr)
                elif filt.startswith("F3_wick_ratio"):
                    extra = ref_df["f3_wick_ratio"].values >= float(thr)
                else:
                    extra = None
            actual = run_combo(precomp, stopcfg, rr, gate_mask, extra, full_start, full_end)
            actual_acc = actual["acc"]
            n_actual = len(actual_acc)
            s_years = (full_end - full_start) / pd.Timedelta(days=365.25)
            actual_totR_yr = actual_acc["R_net"].sum() / s_years
            ts_actual = pd.Series(actual_acc["R_net"].values * RISK_PCT,
                                   index=pd.DatetimeIndex(actual_acc["entry_time"]))
            _, _, actual_ratio = cagr_dd_trades(ts_actual)

            pool_mask = gate_mask & ~base_df["exclude"].values
            pool_acc = sequential_accept(base_df[pool_mask])
            if len(pool_acc) < n_actual:
                print(f"  [{stopcfg} RR{rr} {gname} {filt}] pool too small for null "
                      f"({len(pool_acc)}<{n_actual}), skipping")
                continue
            tot_r_null, cagrdd_null = [], []
            for _ in range(2000):
                pick = rng.choice(len(pool_acc), size=n_actual, replace=False)
                sub = pool_acc.iloc[np.sort(pick)]
                tot_r_null.append(sub["R_net"].sum() / s_years)
                ts = pd.Series(sub["R_net"].values * RISK_PCT, index=pd.DatetimeIndex(sub["entry_time"]))
                _, _, ratio = cagr_dd_trades(ts)
                cagrdd_null.append(ratio)
            tot_r_null = np.array(tot_r_null); cagrdd_null = np.array(cagrdd_null)
            pct_totr = (tot_r_null < actual_totR_yr).mean() * 100.0
            pct_cagrdd = (cagrdd_null < actual_ratio).mean() * 100.0
            rows4.append(dict(stop=stopcfg, RR=rr, gate=gname, filter=filt, n=n_actual,
                               actual_totR_yr=actual_totR_yr, actual_cagr_dd=actual_ratio,
                               null_totR_pctile=pct_totr, null_cagrdd_pctile=pct_cagrdd))
        t4 = pd.DataFrame(rows4)
        t4.to_csv("scratchpad/sweep_short_gates_table4.csv", index=False)
        print(t4.round(3).to_string(index=False))

    # ================================================================ TABLE 5
    print("\n=== TABLE 5: per-year (base=no gate, best single gate, and any table-3 passers) ===")
    year_configs = [("base_NONE", ANCHOR_STOP, ANCHOR_RR, none_mask, None),
                     (f"best_gate_{best_gate_name}", ref_stop, ref_rr, ref_gate_mask, None)]
    if len(t3) > 0:
        seen = set()
        for _, r in t3.iterrows():
            key = (r["stop"], r["RR"], r["gate"], r["filter"])
            if key in seen:
                continue
            seen.add(key)
            gate_mask = combo_masks[r["gate"]]
            if r["source"] == "table1":
                extra = None
            else:
                filt = r["filter"]
                if filt.startswith("F1_low<PDL"):
                    extra = ref_df["f1a_low_lt_pdl"].values
                elif filt.startswith("F1_close<PDL"):
                    extra = ref_df["f1b_close_lt_pdl"].values
                elif filt.startswith("F2_sweep_depth_atr"):
                    extra = ref_df["f2_sweep_atr"].values >= float(filt.split(">=")[1])
                elif filt.startswith("F3_wick_ratio"):
                    extra = ref_df["f3_wick_ratio"].values >= float(filt.split(">=")[1])
                else:
                    extra = None
            year_configs.append((f"PASS_{r['stop']}_RR{r['RR']}_{r['gate']}_{r['filter']}",
                                  r["stop"], r["RR"], gate_mask, extra))

    rows5 = []
    for label, stopcfg, rr, gate_mask, extra in year_configs:
        s = run_combo(precomp, stopcfg, rr, gate_mask, extra, full_start, full_end)
        acc = s["acc"]
        if len(acc) == 0:
            continue
        yrs = pd.DatetimeIndex(acc["entry_time"]).year
        for y in sorted(set(yrs)):
            m = yrs == y
            Rg, Rn = acc["R_gross"].values[m], acc["R_net"].values[m]
            pos, neg = Rg[Rg > 0].sum(), abs(Rg[Rg < 0].sum())
            pf_y = pos / neg if neg > 0 else np.nan
            rows5.append(dict(config=label, year=int(y), n=int(m.sum()), pf_gross=pf_y,
                               mean_r_net=Rn.mean()))
    t5 = pd.DataFrame(rows5)
    t5.to_csv("scratchpad/sweep_short_gates_table5.csv", index=False)
    print(t5.round(3).to_string(index=False))

    print(f"\ntotal elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
