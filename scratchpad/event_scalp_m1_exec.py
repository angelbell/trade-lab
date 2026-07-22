"""STEP A/B/C for spec card: FOMC statement gold 1min scalp -- execution realism / stop-loss /
sequential threshold.

Reuses (does not rewrite):
  fomc_event_study.atr14 / price_before / candidate_dates
  event_scalp.build_scalp_table / null_scalp_table (used only for the F0-equivalence self-check)
  event_scalp_cond.threshold_subset (conditioning on confirm_move_atr)

New machinery in THIS file is limited to what the spec explicitly asks for and cannot be done
with the existing tools: the four fill-price models (F0/F1/F2/F3) and the k*ATR / fixed-$ stop
overlay. The event selection, ATR/no-lookahead price primitives, and threshold-conditioning are
all imported unchanged.

Fill models (both entry and exit; d = +1 long / -1 short from the confirm-bar direction):
  F0 = current baseline: fill AT the confirm-bar close (both entry decision bar and exit decision
       bar) -- i.e. price_before(t_entry)/price_before(t_exit). Unrealistic (you cannot also fill
       at the same close you used to decide).
  F1 = fill at the OPEN of the next 1-minute bar after the decision instant (bar_after()).
  F2 = F1 + adverse slippage s ($/oz): entry costs more/receives less by d*s at entry,
       -d*s at exit (so slippage always works against you, never in your favor).
  F3 = pathological worst case: entry at the next bar's high (d=+1) / low (d=-1); exit at the
       next bar's low (d=+1) / high (d=-1).

No lookahead: the decision to enter/exit is always made from price_before() (a strictly-prior
confirmed close); the FILL price for F1/F2/F3 is read off the very next bar that opens at/after
that decision instant -- never a bar the decision could have seen in advance, and never a bar
that closes before the decision.

Execution: .venv/bin/python scratchpad/event_scalp_m1_exec.py [--smoke]
"""
import argparse
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from scratchpad.fomc_event_study import atr14, price_before, candidate_dates
from scratchpad.event_scalp import build_scalp_table, null_scalp_table, COST_ROUNDTRIP, SEED
from scratchpad.event_scalp_cond import threshold_subset, pctile_of_real_in_pool

COST_BASE = COST_ROUNDTRIP["GOLD"]["base"]   # 0.30 $/oz RT
COST_ALT = COST_ROUNDTRIP["GOLD"]["alt"][0]  # 0.60 $/oz RT
W_C = 1                                       # frozen per background (current reported config)
HSET = [5, 10, 15]
FRACS = [1.00, 0.50, 0.33, 0.25]
SLIPS = [0.10, 0.25, 0.50, 1.00]
DRAWS = 3000
B_BOOT = 2000

EVENTS_CSV = "scratchpad/fomc_stmt_2019.csv"
GOLD_M1 = "data/vantage_xauusd_m1.csv"


# ----------------------------------------------------------------------------
# NEW: execution-fill variants. Only the fill price differs from scalp_metrics; the
# no-lookahead decision primitives (P0, confirm direction, ATR) are identical.
# ----------------------------------------------------------------------------
def bar_after(df, ts):
    """First bar with index >= ts (the bar that opens at/after a decision instant). None if
    past the end of data."""
    pos = df.index.searchsorted(ts, side="left")
    if pos >= len(df):
        return None
    return df.iloc[pos]


def scalp_metrics_exec(df, atr, t0, w_c, horizons, fill, slip=0.0):
    P0 = price_before(df, t0)
    if P0 is None or not np.isfinite(P0):
        return None
    pos0 = df.index.searchsorted(t0, side="left")
    if pos0 == 0:
        return None
    atr_val = atr.iloc[pos0 - 1]
    if not np.isfinite(atr_val) or atr_val <= 0:
        return None

    t_entry = t0 + pd.Timedelta(minutes=w_c)
    P_entry_confirm = price_before(df, t_entry)
    if P_entry_confirm is None or not np.isfinite(P_entry_confirm):
        return None
    diff_c = P_entry_confirm - P0
    if diff_c == 0:
        return None
    d = 1.0 if diff_c > 0 else -1.0

    max_h = max(horizons)
    if df.index.max() < t_entry + pd.Timedelta(minutes=max_h):
        return None

    entry_bar = bar_after(df, t_entry)
    if entry_bar is None:
        return None

    if fill == "F0":
        P_fill_entry = P_entry_confirm
    elif fill in ("F1", "F2"):
        P_fill_entry = entry_bar["open"] + (d * slip if fill == "F2" else 0.0)
    elif fill == "F3":
        P_fill_entry = entry_bar["high"] if d > 0 else entry_bar["low"]
    else:
        raise ValueError(fill)

    out = {"t0": t0, "P0": P0, "ATR": atr_val, "d": d,
           "confirm_move": abs(diff_c), "confirm_move_atr": abs(diff_c) / atr_val,
           "P_entry_confirm": P_entry_confirm, "P_fill_entry": P_fill_entry,
           "t_entry": t_entry, "entry_bar_idx": entry_bar.name}

    for h in horizons:
        t_exit = t_entry + pd.Timedelta(minutes=h)
        exit_bar = bar_after(df, t_exit)
        if exit_bar is None:
            out[f"g_{h}"] = np.nan
            out[f"P_fill_exit_{h}"] = np.nan
            continue
        if fill == "F0":
            P_fill_exit = price_before(df, t_exit)
            if P_fill_exit is None or not np.isfinite(P_fill_exit):
                out[f"g_{h}"] = np.nan
                out[f"P_fill_exit_{h}"] = np.nan
                continue
        elif fill in ("F1", "F2"):
            P_fill_exit = exit_bar["open"] - (d * slip if fill == "F2" else 0.0)
        elif fill == "F3":
            P_fill_exit = exit_bar["low"] if d > 0 else exit_bar["high"]
        g = d * (P_fill_exit - P_fill_entry)
        out[f"g_{h}"] = g
        out[f"P_fill_exit_{h}"] = P_fill_exit
    return out


def build_scalp_table_exec(df, events, w_c, horizons, fill, slip, label):
    atr = atr14(df)
    rows = []
    for t0 in events:
        r = scalp_metrics_exec(df, atr, t0, w_c, horizons, fill, slip)
        if r is not None:
            rows.append(r)
    tbl = pd.DataFrame(rows)
    print(f"  [{label}] usable events: {len(tbl)} / {len(events)} (fill={fill}, slip={slip})", file=sys.stderr)
    return tbl


def null_scalp_table_exec(df, events, w_c, horizons, fill, slip, label, draws_target=DRAWS, seed=SEED):
    atr = atr14(df)
    cand = candidate_dates(df, events)
    if not cand:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    per_event = max(1, int(np.ceil(draws_target / max(1, len(events)))))
    rows = []
    for e in events:
        hod, mod = e.hour, e.minute
        picks = rng.choice(len(cand), size=min(per_event * 3, len(cand)), replace=False)
        n_ok = 0
        for pi in picks:
            if n_ok >= per_event:
                break
            day = cand[pi]
            t0 = pd.Timestamp(day.date(), tz="UTC") + pd.Timedelta(hours=hod, minutes=mod)
            r = scalp_metrics_exec(df, atr, t0, w_c, horizons, fill, slip)
            if r is not None:
                rows.append(r)
                n_ok += 1
    tbl = pd.DataFrame(rows)
    print(f"  [{label} NULL] draws collected: {len(tbl)} (target ~{draws_target})", file=sys.stderr)
    return tbl


def profit_factor(net):
    pos = net[net > 0].sum()
    neg = -net[net < 0].sum()
    if neg == 0:
        return np.inf if pos > 0 else np.nan
    return pos / neg


def cell_stats(real_tbl, null_tbl, frac, h, cost, span_years):
    sub, thr = threshold_subset(real_tbl, "confirm_move_atr", frac)
    nsub = null_tbl if frac >= 1.0 else null_tbl[null_tbl["confirm_move_atr"] >= thr]
    gcol = f"g_{h}"
    g = sub[gcol].dropna()
    gn = nsub[gcol].dropna()
    if len(g) == 0:
        return None
    net = g - cost
    epy = len(g) / span_years if span_years > 0 else np.nan
    pctile, n_pool = pctile_of_real_in_pool(g, gn, cost, B=B_BOOT)
    return {
        "frac": frac, "H_min": h, "thr": thr, "n": len(g), "n_per_year": epy,
        "net_mean": net.mean(), "win_pct": (g > 0).mean() * 100,
        "PF": profit_factor(net), "null_pctile": pctile, "n_null_pool": n_pool,
    }


def is_oos_split(sub, gcol, cost):
    t = sub.dropna(subset=[gcol]).sort_values("t0").reset_index(drop=True)
    half = len(t) // 2
    is_t, oos_t = t.iloc[:half], t.iloc[half:]
    out = {}
    for name, seg in [("IS", is_t), ("OOS", oos_t)]:
        if len(seg) == 0:
            out[name] = None
            continue
        net = seg[gcol] - cost
        out[name] = {"n": len(seg), "net_mean": net.mean(), "win_pct": (seg[gcol] > 0).mean() * 100,
                      "span": f"{seg['t0'].min().date()}..{seg['t0'].max().date()}"}
    return out


def annual_breakdown(sub, gcol, cost):
    t = sub.dropna(subset=[gcol]).copy()
    t["year"] = t["t0"].dt.year
    net = t[gcol] - cost
    t = t.assign(net=net)
    g = t.groupby("year").agg(n=("net", "size"), net_mean=("net", "mean"), net_sum=("net", "sum"),
                               win_pct=(gcol, lambda s: (s > 0).mean() * 100))
    return g.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="only F0/F1 at frac=1.00,0.50, H=5 -- fast sanity pass")
    args = ap.parse_args()

    ev = pd.read_csv(EVENTS_CSV, parse_dates=["dt_utc", "dt_broker"])
    ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
    events = list(ev["dt_broker"].sort_values())
    print(f"total candidate events: {len(events)}")

    df = load_mt5_csv(GOLD_M1)
    print(f"GOLD m1: {len(df)} bars, span {df.index.min()} .. {df.index.max()}")

    # -------------------------------------------------------------------
    # Self-check: F0 built via THIS file's scalp_metrics_exec must equal the existing
    # event_scalp.build_scalp_table (both are "confirm-bar-close fill") -- if they don't match
    # byte-for-byte, the new machinery has a bug and everything downstream is suspect.
    # -------------------------------------------------------------------
    ref = build_scalp_table(df, events, W_C, HSET, "GOLD-ref-F0")
    f0_check = build_scalp_table_exec(df, events, W_C, HSET, "F0", 0.0, "GOLD-selfcheck-F0")
    ref_s = ref.set_index("t0")[[f"g_{h}" for h in HSET]]
    f0_s = f0_check.set_index("t0")[[f"g_{h}" for h in HSET]]
    same = ref_s.reindex(f0_s.index).equals(f0_s)
    max_diff = (ref_s.reindex(f0_s.index) - f0_s).abs().max().max()
    print(f"\n[SELF-CHECK] F0 (this file) vs build_scalp_table (existing, tie-back-verified): "
          f"identical={same}, max_abs_diff={max_diff}")
    if not same and max_diff > 1e-9:
        print("[SELF-CHECK] MISMATCH beyond float tolerance -- STOPPING, do not trust downstream numbers.")
        return

    if args.smoke:
        fill_specs = [("F0", 0.0), ("F1", 0.0)]
        fracs = [1.00, 0.50]
        hset = [5]
    else:
        fill_specs = [("F0", 0.0), ("F1", 0.0)] + [("F2", s) for s in SLIPS] + [("F3", 0.0)]
        fracs = FRACS
        hset = HSET

    real_tables = {}
    null_tables = {}
    for fill, slip in fill_specs:
        key = f"{fill}_s{slip:g}" if fill == "F2" else fill
        t0 = time.time()
        real_tables[key] = build_scalp_table_exec(df, events, W_C, hset, fill, slip, f"GOLD-{key}")
        null_tables[key] = null_scalp_table_exec(df, events, W_C, hset, fill, slip, f"GOLD-{key}",
                                                   draws_target=DRAWS)
        print(f"  [{key}] build time: {time.time()-t0:.1f}s", file=sys.stderr)

    span_years = (real_tables["F0"]["t0"].max() - real_tables["F0"]["t0"].min()).days / 365.25
    print(f"\nspan_years (from usable F0 events) = {span_years:.3f}")

    # -------------------------------------------------------------------
    # STEP A main grid: fill x frac x H, cost_base
    # -------------------------------------------------------------------
    print(f"\n{'='*110}\nSTEP A -- fill-model x frac x H grid (cost=${COST_BASE}/oz RT, w_c={W_C}min)\n{'='*110}")
    rows = []
    for key in real_tables:
        for frac in fracs:
            for h in hset:
                r = cell_stats(real_tables[key], null_tables[key], frac, h, COST_BASE, span_years)
                if r is not None:
                    r["fill"] = key
                    rows.append(r)
    grid = pd.DataFrame(rows)[["fill", "frac", "H_min", "n", "n_per_year", "net_mean", "win_pct", "PF",
                                "null_pctile", "n_null_pool", "thr"]]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(grid.round(4).to_string(index=False))

    print(f"\n--- same grid, cost_alt=${COST_ALT}/oz (conservative) -- net_mean only ---")
    rows_alt = []
    for key in real_tables:
        for frac in fracs:
            for h in hset:
                r = cell_stats(real_tables[key], null_tables[key], frac, h, COST_ALT, span_years)
                if r is not None:
                    rows_alt.append({"fill": key, "frac": frac, "H_min": h, "n": r["n"],
                                      "net_mean_cost_alt": r["net_mean"], "win_pct": r["win_pct"]})
    print(pd.DataFrame(rows_alt).round(4).to_string(index=False))

    # -------------------------------------------------------------------
    # Judgment per spec: F1 (slip=0) net<=0, or F2 s=0.25 net<=0 -> dead on execution.
    # Judged at the CURRENTLY-CLAIMED live operating point: frac=0.50, H=5 (the cell the
    # background reports as +1.97), plus frac=1.00 (all-signals base) for context.
    # -------------------------------------------------------------------
    print(f"\n{'='*110}\nJUDGMENT (frac=0.50 / H=5, the reported live cell; frac=1.00 shown for context)\n{'='*110}")
    judged = grid[(grid["H_min"] == 5) & (grid["frac"].isin([1.00, 0.50]))]
    print(judged.round(4).to_string(index=False))

    f1_net_050 = grid[(grid["fill"] == "F1") & (grid["frac"] == 0.50) & (grid["H_min"] == 5)]["net_mean"]
    f1_net_100 = grid[(grid["fill"] == "F1") & (grid["frac"] == 1.00) & (grid["H_min"] == 5)]["net_mean"]
    s025_net_050 = grid[(grid["fill"] == "F2_s0.25") & (grid["frac"] == 0.50) & (grid["H_min"] == 5)]["net_mean"]

    f1_dead = len(f1_net_050) and f1_net_050.iloc[0] <= 0
    s025_dead = len(s025_net_050) and s025_net_050.iloc[0] <= 0
    verdict_dead = bool(f1_dead or s025_dead)
    print(f"\nF1 net_mean (frac=0.50,H=5) = {f1_net_050.iloc[0] if len(f1_net_050) else np.nan:+.4f}")
    print(f"F2 s=0.25 net_mean (frac=0.50,H=5) = {s025_net_050.iloc[0] if len(s025_net_050) else np.nan:+.4f}")
    print(f"F1 net_mean (frac=1.00,H=5, all-signals) = {f1_net_100.iloc[0] if len(f1_net_100) else np.nan:+.4f}")
    print(f"\n>>> VERDICT: {'DEAD ON EXECUTION' if verdict_dead else 'SURVIVES STEP A -- proceeding to STEP B/C'} <<<")

    # -------------------------------------------------------------------
    # Deep dive (spec: "F1 と、net が正で残る最大の s" -- annual + IS/OOS), frac=0.50, H=5
    # -------------------------------------------------------------------
    print(f"\n{'='*110}\nDEEP DIVE: annual + IS/OOS for F1, and largest surviving-net s, at frac=0.50/H=5\n{'='*110}")
    max_surviving_s = None
    for s in SLIPS:
        key = f"F2_s{s:g}"
        row = grid[(grid["fill"] == key) & (grid["frac"] == 0.50) & (grid["H_min"] == 5)]
        if len(row) and row["net_mean"].iloc[0] > 0:
            max_surviving_s = s
    print(f"largest s with net_mean>0 at frac=0.50,H=5: {max_surviving_s}")

    dive_keys = ["F1"] + ([f"F2_s{max_surviving_s:g}"] if max_surviving_s is not None else [])
    for key in dive_keys:
        sub, thr = threshold_subset(real_tables[key], "confirm_move_atr", 0.50)
        gcol = "g_5"
        print(f"\n--- {key}, frac=0.50 (thr={thr:.4f}), H=5 ---")
        ann = annual_breakdown(sub, gcol, COST_BASE)
        print(ann.round(4).to_string(index=False))
        iso = is_oos_split(sub, gcol, COST_BASE)
        for seg, v in iso.items():
            if v is None:
                print(f"  {seg}: (empty)")
            else:
                print(f"  {seg}: n={v['n']} net_mean={v['net_mean']:+.4f} win%={v['win_pct']:.1f} span={v['span']}")

    if verdict_dead:
        print(f"\n{'='*110}\nSTEP A judgment = DEAD ON EXECUTION -- STOPPING per spec. STEP B/C NOT run.\n{'='*110}")
        return

    print(f"\n{'='*110}\nSTEP A SURVIVES -- proceeding is handled by a follow-up script (event_scalp_m1_stop.py "
          f"/ event_scalp_m1_seq.py) per spec STEP B/C.\n{'='*110}")


if __name__ == "__main__":
    main()
