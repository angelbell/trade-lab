"""event_scalp_cond.py -- spec_event_scalp_cond.md (仕様カード18)

カード17（event_scalp.py）で gold は follow-through が本物（Spearman +0.28, null 98.7%ile）
だが全事象平均はコスト死、という結果を受けて: 「確認5分の動きが大きいイベントだけ取れば net が
コストを抜けるか」を、確認サイズ C=|P_entry-P0|/ATR14 の閾値スイープ + 同条件null で検定する。

流用（車輪の再発明禁止）: event_scalp.py の build_scalp_table/null_scalp_table/is_oos_table/
annual_table と、その内部で使われている fomc_event_study.py の atr14/price_before/candidate_dates/
M15_START をそのまま使う。新規に書くのは「確認サイズ(C_atr, 生$)での層別・スイープ・その専用の
bootstrap/block-bootstrap/random-drop null」だけ。自前のイベント抽出・価格取得・出口計算は一切書かない。

機構はカード17と同一・先読み厳禁（すべて scalp_metrics 内で強制済み）:
  t0=リリース, P0=直前確定足終値, w_c=5分固定, P_entry=close_{t0+5min}, d=sign(P_entry-P0)(d=0はskip)
  C = |P_entry-P0| / ATR14(m5, t0直前)  (確認サイズ、ATR正規化。生$も併記)
  建て=P_entry成行, 決済H分後の終値, g=d*(P_exit-P_entry), net=g-cost

実行:
  .venv/bin/python scratchpad/event_scalp_cond.py --smoke   # 直近2年
  .venv/bin/python scratchpad/event_scalp_cond.py           # フル
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv  # noqa: E402
from event_scalp import (  # noqa: E402
    build_scalp_table, null_scalp_table, is_oos_table, annual_table,
    GOLD_M5_START, COST_ROUNDTRIP, SEED, NULL_DRAWS_TARGET,
)
from fomc_event_study import atr14, price_before, candidate_dates, M15_START  # noqa: E402  (re-exported for the tie-back / spec-card reference; used indirectly via event_scalp's helpers)

HSET = [5, 10, 15, 20, 30]           # ユーザー指示: 5分だけでなく10/15/20/30分も見る
W_C = 5                              # 確認窓は5分固定（カード17と同一）
FRACS = [1.00, 0.70, 0.50, 0.33, 0.25, 0.20]   # C_atr 上位 take%
COST_BASE = COST_ROUNDTRIP["GOLD"]["base"]     # $0.30/oz
COST_ALT = COST_ROUNDTRIP["GOLD"]["alt"][0]    # $0.60/oz (保守)
B_BOOT = 2000
BLOCK_MONTHS = [1, 3, 6, 12]


# ----------------------------------------------------------------------------
# NEW: threshold layering on top of the scalp table event_scalp.py already builds.
# ----------------------------------------------------------------------------
def threshold_subset(tbl, col, frac):
    """Keep the top `frac` fraction of events by `col` (descending). frac=1.0 -> all."""
    if frac >= 1.0:
        return tbl, -np.inf
    thr = tbl[col].quantile(1 - frac)
    return tbl[tbl[col] >= thr], thr


def boot_stats(g, cost, B=B_BOOT, seed=SEED, events_per_year=np.nan):
    """Resample g (with replacement) B times; return median/p25/p75/std of net_mean, win%,
    P(net>0), annual_equiv across the bootstrap distribution (feedback: report median+std for
    any probability/distribution, not just the point estimate)."""
    rng = np.random.default_rng(seed)
    n = len(g)
    if n == 0:
        return {}
    g_arr = g.to_numpy()
    idx = rng.integers(0, n, size=(B, n))
    gb = g_arr[idx]
    net_b = gb - cost
    net_mean_b = net_b.mean(axis=1)
    win_b = (gb > 0).mean(axis=1) * 100
    pnetpos_b = (net_b > 0).mean(axis=1) * 100
    annual_b = net_mean_b * events_per_year
    out = {}
    for name, arr in [("net_mean", net_mean_b), ("win_pct", win_b),
                       ("P_net_pos", pnetpos_b), ("annual_equiv", annual_b)]:
        out[f"{name}_bmed"] = np.median(arr)
        out[f"{name}_bp25"] = np.quantile(arr, 0.25)
        out[f"{name}_bp75"] = np.quantile(arr, 0.75)
        out[f"{name}_bstd"] = np.std(arr)
    return out


def pctile_of_real_in_pool(real_g, pool_g, cost, B=B_BOOT, seed=SEED):
    """Resample pool_g at n=len(real_g) B times (with replacement), compute net_mean each time,
    return where real_g's net_mean falls in that sampling distribution (percentile)."""
    rng = np.random.default_rng(seed)
    n_real = len(real_g)
    pool = pool_g.to_numpy()
    n_pool = len(pool)
    if n_real == 0 or n_pool < 5:
        return np.nan, n_pool
    real_net_mean = (real_g - cost).mean()
    idx = rng.integers(0, n_pool, size=(B, n_real))
    boot_net_mean = (pool[idx] - cost).mean(axis=1)
    pctile = (boot_net_mean <= real_net_mean).mean() * 100
    return pctile, n_pool


def pctile_vs_random_drop(real_g, full_g, cost, B=B_BOOT, seed=SEED):
    """Random-drop null: draw n=len(real_g) events WITHOUT replacement from the full
    (unconditioned) real event pool, B times; where does the C-selected subset's net_mean fall
    vs. 'just having fewer, randomly chosen, events'?"""
    rng = np.random.default_rng(seed)
    n_real = len(real_g)
    full = full_g.to_numpy()
    n_full = len(full)
    if n_real == 0 or n_real >= n_full:
        return np.nan
    real_net_mean = (real_g - cost).mean()
    boot = np.empty(B)
    for b in range(B):
        pick = rng.choice(n_full, size=n_real, replace=False)
        boot[b] = (full[pick] - cost).mean()
    pctile = (boot <= real_net_mean).mean() * 100
    return pctile


def block_bootstrap_ci(subset, gcol, cost, block_months, B=B_BOOT, seed=SEED):
    """Non-overlapping calendar-month block bootstrap (resample blocks WITH replacement).
    Events are sparse (FOMC ~8/yr) so a dense daily circular-block scheme doesn't apply;
    instead we bucket events into `block_months`-wide calendar blocks (anchored at the
    subset's earliest t0) and resample blocks -- this is the event-level analogue of the
    circular block bootstrap: it tests whether the result depends on a few clustered months
    rather than being spread across the sample. Flagged as an interpretation choice."""
    t = subset.dropna(subset=[gcol]).copy()
    if len(t) < 4:
        return None
    t0min = t["t0"].min()
    months_since = (t["t0"].dt.year - t0min.year) * 12 + (t["t0"].dt.month - t0min.month)
    block_id = (months_since // block_months).astype(int)
    groups = {k: v[gcol].to_numpy() for k, v in t.groupby(block_id)}
    keys = list(groups.keys())
    n_blocks = len(keys)
    if n_blocks < 3:
        return {"n_blocks": n_blocks, "note": "too few blocks (<3), skipped"}
    rng = np.random.default_rng(seed)
    net_means = np.empty(B)
    for b in range(B):
        picks = rng.integers(0, n_blocks, size=n_blocks)
        vals = np.concatenate([groups[keys[p]] for p in picks])
        net_means[b] = (vals - cost).mean()
    return {
        "n_blocks": n_blocks, "n_events": len(t),
        "median": np.median(net_means), "p5": np.quantile(net_means, 0.05),
        "p95": np.quantile(net_means, 0.95), "std": np.std(net_means),
        "P_le_0": (net_means <= 0).mean() * 100,
    }


def sweep_table(real_full, null_full, col, fracs, hset, cost, span_years, deep=True):
    """col = 'confirm_move_atr' (C, ATR-normalized) or 'confirm_move' (raw $)."""
    rows = []
    subsets = {}
    for frac in fracs:
        sub_real, thr = threshold_subset(real_full, col, frac)
        sub_null = null_full[null_full[col] >= thr]   # SAME absolute threshold applied to the null pool
        subsets[frac] = (sub_real, sub_null, thr)
        for h in hset:
            gcol, gacol = f"g_{h}", f"gatr_{h}"
            g = sub_real[gcol].dropna()
            ga = sub_real[gacol].dropna()
            if len(g) == 0:
                continue
            n_real = len(g)
            epy = n_real / span_years if span_years > 0 else np.nan
            net = g - cost
            row = {
                "frac": frac, "thr": thr, "H_min": h, "n": n_real,
                "gross_median": g.median(), "gross_p25": g.quantile(0.25), "gross_p75": g.quantile(0.75),
                "gross_std": g.std(), "gatr_median": ga.median(),
                "win_pct": (g > 0).mean() * 100,
                "net_mean": net.mean(), "net_median": net.median(),
                "P_net_pos": (net > 0).mean() * 100,
                "annual_equiv": net.mean() * epy, "events_per_year": epy,
            }
            if deep:
                row.update(boot_stats(g, cost, events_per_year=epy))
                g_null = sub_null[gcol].dropna()
                row["n_null_conditioned"] = len(g_null)
                if len(g_null) >= 5:
                    row["net_mean_null_conditioned"] = (g_null - cost).mean()
                    row["win_pct_null_conditioned"] = (g_null > 0).mean() * 100
                    pctile_nc, _ = pctile_of_real_in_pool(g, g_null, cost)
                    row["pctile_net_mean_vs_null_conditioned"] = pctile_nc
                else:
                    row["net_mean_null_conditioned"] = np.nan
                    row["win_pct_null_conditioned"] = np.nan
                    row["pctile_net_mean_vs_null_conditioned"] = np.nan
                full_g = real_full[gcol].dropna()
                row["pctile_net_mean_vs_random_drop"] = pctile_vs_random_drop(g, full_g, cost)
            rows.append(row)
    return pd.DataFrame(rows), subsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/ext_fomc_dates.csv")
    ap.add_argument("--smoke", action="store_true", help="last 2 years only")
    ap.add_argument("--draws", type=int, default=NULL_DRAWS_TARGET)
    args = ap.parse_args()

    ev = pd.read_csv(args.events, parse_dates=["dt_utc", "dt_broker"])
    ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
    events_all = list(ev["dt_broker"].sort_values())

    if args.smoke:
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=730)
        cutoff = pd.Timestamp(cutoff, tz="UTC")
        events_all = [e for e in events_all if e >= cutoff]
        print(f"[SMOKE] using {len(events_all)} events since {cutoff.date()}")

    print(f"Total events in candidate list ({args.events}): {len(events_all)}")

    df = load_mt5_csv("data/vantage_xauusd_m5.csv")
    df = df.loc[GOLD_M5_START:]
    print(f"\nGOLD m5 data: {len(df)} bars, span {df.index.min()} .. {df.index.max()}  "
          f"(density boundary GOLD_M5_START={GOLD_M5_START})")

    real_full = build_scalp_table(df, events_all, W_C, HSET, "GOLD-cond")
    if real_full.empty or len(real_full) < 5:
        print(f"too few usable events (n={len(real_full)}) -- aborting")
        return
    span_years = (real_full["t0"].max() - real_full["t0"].min()).days / 365.25
    print(f"usable events n={len(real_full)}, span {span_years:.2f}y "
          f"({len(real_full)/span_years:.2f} events/year)")

    null_full = null_scalp_table(df, events_all, W_C, HSET, "GOLD-cond", draws_target=args.draws)
    print(f"null pool: {len(null_full)} draws")

    # ---------------- MAIN SWEEP: C_atr (ATR-normalized confirm size) ----------------
    print(f"\n{'='*100}\nMAIN SWEEP -- threshold on confirm_move_atr (C), take-top-frac x H, cost_base=${COST_BASE}/oz\n{'='*100}")
    sw_atr, subsets_atr = sweep_table(real_full, null_full, "confirm_move_atr", FRACS, HSET, COST_BASE, span_years, deep=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(sw_atr.round(4).to_string(index=False))

    print(f"\n--- same sweep, cost_alt=${COST_ALT}/oz (conservative) -- point net_mean/median only ---")
    for frac in FRACS:
        sub_real, _, thr = subsets_atr[frac]
        for h in HSET:
            gcol = f"g_{h}"
            g = sub_real[gcol].dropna()
            if len(g) == 0:
                continue
            net = g - COST_ALT
            print(f"  frac={frac:.2f} thr={thr:.4f} H={h:>2} n={len(g):>3}  "
                  f"net_mean={net.mean():+.4f}  net_median={net.median():+.4f}  P(net>0)={((net>0).mean()*100):.1f}%")

    # ---------------- ABSOLUTE $ THRESHOLD (single pass, per spec) ----------------
    print(f"\n{'='*100}\nSINGLE PASS -- threshold on raw confirm_move ($/oz) instead of C_atr\n{'='*100}")
    sw_dollar, subsets_dollar = sweep_table(real_full, null_full, "confirm_move", FRACS, HSET, COST_BASE, span_years, deep=False)
    print(sw_dollar.round(4).to_string(index=False))

    # ---------------- RANDOM-DROP null already embedded per-cell above; report best cell ----------------
    print(f"\n{'='*100}\nBEST-CELL SELECTION (by bootstrap-median net_mean, base cost) + caveats\n{'='*100}")
    cand = sw_atr[(sw_atr["frac"] < 1.0) & (sw_atr["n"] >= 8)].copy()
    if cand.empty:
        cand = sw_atr.copy()
    cand = cand.sort_values("net_mean_bmed", ascending=False)
    print(cand[["frac", "thr", "H_min", "n", "net_mean", "net_mean_bmed", "net_mean_bp25", "net_mean_bp75",
                "P_net_pos", "P_net_pos_bmed", "pctile_net_mean_vs_null_conditioned",
                "pctile_net_mean_vs_random_drop"]].head(10).round(4).to_string(index=False))
    print(f"\nNOTE: grid searched = {len(FRACS)} fracs x {len(HSET)} H = {len(FRACS)*len(HSET)} cells "
          f"(C_atr sweep) -- Bonferroni-style caution: best-cell p-values/percentiles above should be "
          f"discounted by ~{len(FRACS)*len(HSET)}x versus a single pre-registered test.")

    # Two deep-dives, reported side by side (this script does NOT pick a winner):
    #  (a) naive top-1 by bootstrap-median net_mean -- may sit at a threshold so extreme that
    #      almost no non-FOMC day ever reaches it (null-conditioned pool too thin to trust).
    #  (b) top-1 restricted to n_null_conditioned>=20 -- the best cell where the "same
    #      condition, non-FOMC" null comparison is actually statistically meaningful.
    naive_best = cand.iloc[0]
    reliable_cand = cand[cand["n_null_conditioned"] >= 20]
    reliable_best = reliable_cand.iloc[0] if not reliable_cand.empty else None

    picks = [("NAIVE top-1 (highest bootstrap-median net_mean, any n_null_conditioned)", naive_best)]
    if reliable_best is not None and not (reliable_best[["frac", "H_min"]] == naive_best[["frac", "H_min"]]).all():
        picks.append(("RELIABLE-NULL top-1 (n_null_conditioned>=20, so the null% is trustworthy)", reliable_best))
    elif reliable_best is not None:
        print("\n(naive top-1 and reliable-null top-1 coincide -- single deep-dive below)")

    for label, best in picks:
        best_frac, best_h = best["frac"], int(best["H_min"])
        sub_real_best, sub_null_best, thr_best = subsets_atr[best_frac]
        print(f"\n{'#'*100}\nDEEP-DIVE: {label}\n"
              f"  frac={best_frac:.2f} (thr C_atr>={thr_best:.4f}), H={best_h}min "
              f"(n={int(best['n'])}, net_mean_bmed={best['net_mean_bmed']:.4f}, "
              f"n_null_conditioned={int(best['n_null_conditioned'])}, "
              f"pctile_vs_null_conditioned={best['pctile_net_mean_vs_null_conditioned']:.2f}, "
              f"pctile_vs_random_drop={best['pctile_net_mean_vs_random_drop']:.2f})\n{'#'*100}")

        print(f"\n--- block bootstrap (calendar-month blocks, resample-with-replacement, B={B_BOOT}) ---")
        for bm in BLOCK_MONTHS:
            res = block_bootstrap_ci(sub_real_best, f"g_{best_h}", COST_BASE, bm)
            if res is None:
                print(f"  block={bm:>2}mo: n<4 events, skipped")
            elif "note" in res:
                print(f"  block={bm:>2}mo: {res['note']} (n_blocks={res['n_blocks']})")
            else:
                print(f"  block={bm:>2}mo: n_blocks={res['n_blocks']:>3} n_events={res['n_events']:>3}  "
                      f"net_median={res['median']:+.4f}  [p5={res['p5']:+.4f}, p95={res['p95']:+.4f}]  "
                      f"std={res['std']:.4f}  P(net<=0)={res['P_le_0']:.1f}%")

        print(f"\n--- IS/OOS split (front half / back half by t0) ---")
        ist, span_desc = is_oos_table(sub_real_best, HSET, COST_BASE)
        print(f"  {span_desc}")
        print(ist.round(4).to_string(index=False))

        print(f"\n--- annual breakdown @ H={best_h}min (net, cost_base=${COST_BASE}) ---")
        at = annual_table(sub_real_best, best_h, COST_BASE)
        print(at.round(4).to_string(index=False))

    print(f"\n\n{'='*100}\nDONE\n{'='*100}")


if __name__ == "__main__":
    main()
