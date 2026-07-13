"""btc_family_ext_throttle.py -- is the 4-week-extension throttle a btc15m_L QUIRK or a
BTC-SYMBOL-LEVEL lever?

Yesterday's finding (book_healing_veto.py): halving the size of btc15m_L trades entered right
after BTC has run hard over the prior 4 weeks improves the 6-leg book CAGR/DD 12.03 -> 12.70
(block-bootstrap P 63->78%). Question: is ret4w a property of the SYMBOL (in which case
applying the SAME throttle to the other 3 BTC legs -- btc_bo_kama, btc_pull, btc15m_S --
should ALSO help the book), or is it a path-fit specific to btc15m_L's own trade set (in which
case A2 should NOT beat A1, or should hurt).

Context variable (shared across ALL BTC legs, no lookahead):
    ret4w = dcl / dcl.shift(28) - 1.0   on BTC daily close (vantage_btcusd_m15 -> 1D),
    shifted 1 extra day (prior COMPLETED daily bar), then reindexed to each leg's own entry
    timestamps via ffill. NOT applied to the gold legs.
Threshold: computed on the FIRST HALF of each leg's OWN trades (by time, i.e. leg-specific IS),
applied throughout that leg's full history (no post-hoc threshold shopping).

Arms (book CAGR/DD is the sole arbiter; base = 12.03):
    A0  base (current book, btc15m_L = 4h gate / RR4.5 / PDH-soft-0.5, un-throttled)
    A1  btc15m_L only, IS-75pct, w=0.5           -- must reproduce 12.70
    A2  ALL 4 BTC legs, IS-75pct, w=0.5           -- the symbol-level-lever test
    A3  each OTHER leg alone (btc_bo_kama-only / btc_pull-only / btc15m_S-only), IS-75pct,w=0.5
    A4  A2's plateau: quantile in {70,75,80,85,90} x weight in {0.25,0.5,0.75}, 15 cells

Judged by: (1) single-path book CAGR/DD all arms, (2) paired circular block bootstrap of the
book's monthly returns (block 1/3/6/12mo, 2000 draws, SAME resampled months across arms) ->
median CAGR/DD + P(arm beats A0), (3) per-BTC-leg flagged-vs-rest n/meanR/PF (is the throttle
cutting dead weight or just diversifying away winners?), (4) how much the 4 legs' flagged
months OVERLAP (if they don't overlap, "BTC-level regime" is not the right description even if
A2 happens to beat A0).

Run: .venv/bin/python scratchpad/btc_family_ext_throttle.py
Run (smoke, fewer bootstrap draws): .venv/bin/python scratchpad/btc_family_ext_throttle.py --smoke
"""
import os, sys, io, contextlib, warnings, argparse
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from research.portfolio_kama import get_legs
from radar_gate_race import BASE
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from short_mirror_15m import invert
from book_hh4h_weight_sweep import book as book_orig, leg_stats, ROOT, NEW
from book_bootstrap_arbiter import cdd

BTC_LEGS = ["btc_bo_kama", "btc_pull", "btc15m_S", "btc15m_L"]


# ---------------------------------------------------------------------------
# context variable: BTC 4-week return, prior completed daily bar, no lookahead
# ---------------------------------------------------------------------------
def ret4w_daily():
    full = load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv")
    dcl = full["close"].resample("1D").last()
    return (dcl / dcl.shift(28) - 1.0).shift(1)


def flag_hot(idx, ctx_daily, q):
    """Per-leg IS-quantile threshold: quantile taken on the FIRST HALF of this leg's OWN
    trades (by time), applied to the full history. Returns (hot bool array, threshold)."""
    idx = pd.DatetimeIndex(idx)
    ctx = ctx_daily.reindex(idx, method="ffill").values
    half = idx[len(idx) // 2]
    thr = np.nanquantile(ctx[idx < half], q)
    hot = np.isfinite(ctx) & (ctx >= thr)
    return hot, thr


# ---------------------------------------------------------------------------
# build the 6 canonical (adopted-weight) leg series -- identical construction to
# book_hh4h_weight_sweep.py / book_healing_veto.py (tie-back target: book CAGR/DD 12.03)
# ---------------------------------------------------------------------------
def build_base():
    with contextlib.redirect_stderr(io.StringIO()):
        legs = {k: pd.Series(t.R.values, index=pd.DatetimeIndex(t.time)) for k, t in get_legs().items()}

        g = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                      "ext_cap": 8.0, "pullback_frac": 0.25}))
        legs["gold15m"] = pd.Series(t["R"].values - 0.3 / t["risk"].values,
                                    index=pd.DatetimeIndex(t["time"]))

        full = load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv")
        d15 = resample(full.loc["2018-10-01":], "15min")
        inv = invert(d15); C = 2 * d15["high"].max()
        ts_ = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3}))
        Rs = ts_["R"].values - 15.0 / ts_["risk"].values
        pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
        mS = (C - ts_["e_px"].values) < pdl[d15.index.get_indexer(ts_["time"])]
        legs["btc15m_S"] = pd.Series(Rs[mS], index=pd.DatetimeIndex(ts_["time"])[mS])

        tL = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                         "pullback_frac": 0.3, "rr": 4.5}))
        Rn = tL["R"].values - 15.0 / tL["risk"].values
        ei = d15.index.get_indexer(tL["time"])
        pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
        base_w = np.where(tL["e_px"].values > pdh[ei], 1.0, 0.5)
        legs["btc15m_L"] = pd.Series(Rn * base_w, index=pd.DatetimeIndex(tL["time"]))
    return legs


# ---------------------------------------------------------------------------
# generalized book(): SAME inv-vol / total-risk-3% / NEW-basket / monthly logic as
# book_hh4h_weight_sweep.book, but can override ANY subset of leg keys (not just btc15m_L)
# ---------------------------------------------------------------------------
def book_gen(legs, overrides=None):
    L = dict(legs)
    if overrides:
        L.update(overrides)
    mon = {k: s.groupby(s.index.to_period("M")).sum() for k, s in L.items()}
    st = max(s.index.min() for s in mon.values()); en = min(s.index.max() for s in mon.values())
    midx = pd.period_range(st, en, freq="M")
    M = pd.DataFrame({k: v.reindex(midx, fill_value=0.0) for k, v in mon.items()})
    sig = M.std(); w = (1.0 / sig[NEW]); w = w / w.sum() * 0.03
    port = (M[NEW] * w).sum(axis=1)
    eq = np.cumprod(1 + port.values)
    dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
    cagr = (eq[-1] ** (12 / len(port)) - 1) * 100
    return cagr, dd, cagr / dd, port


def throttle(legs, which, q, w):
    """Return {leg_name: throttled series} for every leg name in `which`, IS-q quantile,
    weight w on the flagged ('hot') trades. Independent threshold per leg."""
    ctx = ret4w_daily()
    out = {}
    for name in which:
        s = legs[name]
        hot, thr = flag_hot(s.index, ctx, q)
        out[name] = pd.Series(s.values * np.where(hot, w, 1.0), index=s.index)
    return out


def bootstrap_table(arm_ports, months, ndraw, seed=20260713):
    rng = np.random.default_rng(seed)
    names = list(arm_ports.keys())
    base_name = names[0]
    out = {}
    for blk in (1, 3, 6, 12):
        nb = int(np.ceil(months / blk))
        D = {k: [] for k in names}
        for _ in range(ndraw):
            st = rng.integers(0, months, nb)
            k_ = np.concatenate([(np.arange(s, s + blk) % months) for s in st])[:months]
            for k in names:
                D[k].append(cdd(arm_ports[k][k_], months))
        base_arr = np.array(D[base_name])
        row = {}
        for k in names:
            a = np.array(D[k])
            row[k] = (np.nanmedian(a), np.nanmean(a > base_arr) * 100)
        out[blk] = row
    return out


def leg_flag_report(legs, q=0.75):
    ctx = ret4w_daily()
    reports = {}
    for name in BTC_LEGS:
        s = legs[name]
        hot, thr = flag_hot(s.index, ctx, q)
        R = s.values
        hotR, restR = R[hot], R[~hot]
        def pf(x):
            pos, neg = x[x > 0].sum(), abs(x[x <= 0].sum())
            return pos / neg if neg > 0 else np.nan
        reports[name] = dict(thr=thr, n_hot=hot.sum(), n_rest=(~hot).sum(),
                              meanR_hot=hotR.mean() if hot.sum() else np.nan,
                              meanR_rest=restR.mean() if (~hot).sum() else np.nan,
                              pf_hot=pf(hotR), pf_rest=pf(restR),
                              hot_months=set(s.index[hot].to_period("M")))
    return reports


def main(smoke=False):
    ndraw = 200 if smoke else 2000
    legs = build_base()

    # --- tie-back checks ---
    c0, d0, cd0, k1, k2 = book_orig(legs, legs["btc15m_L"])
    print(f"[tie-back 1] book_hh4h_weight_sweep.book() (ORIGINAL fn) on freshly-built legs: "
          f"book CAGR/DD={cd0:.2f}  book DD={d0:.1f}%   (target 12.03)")

    cagrG, ddG, cdG, portG = book_gen(legs)
    print(f"[tie-back 2] book_gen (NEW general fn), no override: "
          f"book CAGR/DD={cdG:.2f}  book DD={ddG:.1f}%   (target 12.03, must match tie-back 1)")

    thr_over = throttle(legs, ["btc15m_L"], 0.75, 0.5)
    _, _, cd1chk, _ = book_gen(legs, thr_over)
    print(f"[tie-back 3] A1 via book_gen (btc15m_L-only, IS75, w=0.5): "
          f"book CAGR/DD={cd1chk:.2f}   (target 12.70, from book_healing_veto.py 'plain 4-week return' row)")

    if abs(cd0 - 12.03) > 0.05 or abs(cdG - 12.03) > 0.05 or abs(cd1chk - 12.70) > 0.05:
        print("\n*** TIE-BACK MISMATCH -- stopping before proceeding further. Report this. ***")
        return

    print()

    # --- arms ---
    arms = {}
    arm_overrides = {}
    arm_overrides["A0 base"] = {}
    arm_overrides["A1 L-only (IS75,w=0.5)"] = throttle(legs, ["btc15m_L"], 0.75, 0.5)
    arm_overrides["A2 ALL 4 BTC legs (IS75,w=0.5)"] = throttle(legs, BTC_LEGS, 0.75, 0.5)
    arm_overrides["A3 btc_bo_kama-only (IS75,w=0.5)"] = throttle(legs, ["btc_bo_kama"], 0.75, 0.5)
    arm_overrides["A3 btc_pull-only (IS75,w=0.5)"] = throttle(legs, ["btc_pull"], 0.75, 0.5)
    arm_overrides["A3 btc15m_S-only (IS75,w=0.5)"] = throttle(legs, ["btc15m_S"], 0.75, 0.5)

    results = {}
    for name, ov in arm_overrides.items():
        cagr, dd, cd, port = book_gen(legs, ov)
        results[name] = dict(cagr=cagr, dd=dd, cd=cd, port=port, ov=ov)

    print("=" * 100)
    print("TABLE 1 -- single-path book CAGR/DD, and per-leg PF/CAGR-DD of whichever BTC leg(s) changed")
    print("=" * 100)
    hdr = f"{'arm':<34}{'changed leg(s)':<40}{'leg PF':>8}{'leg C/DD':>10}{'BOOK C/DD':>11}{'book DD':>9}"
    print(hdr)
    for name, ov in arm_overrides.items():
        r = results[name]
        if not ov:
            print(f"{name:<34}{'(none)':<40}{'':>8}{'':>10}{r['cd']:>11.2f}{r['dd']:>8.1f}%")
            continue
        for leg_name, s in ov.items():
            pf, ld, lcd = leg_stats(s)
            tag = leg_name if len(ov) > 1 else ""
            print(f"{name if leg_name == list(ov)[0] else '':<34}{leg_name:<40}{pf:>8.2f}{lcd:>10.2f}"
                  f"{r['cd']:>11.2f}{r['dd']:>8.1f}%")

    # --- A4 plateau grid ---
    print()
    print("=" * 100)
    print("TABLE 1b -- A4 plateau grid (ALL 4 BTC legs throttled together): book CAGR/DD by (quantile,weight)")
    print("=" * 100)
    qs = [0.70, 0.75, 0.80, 0.85, 0.90]
    ws = [0.25, 0.5, 0.75]
    print(f"{'quantile':<10}" + "".join(f"w={w:<9}" for w in ws))
    a4_overrides = {}
    a4_results = {}
    for q in qs:
        row = []
        for w in ws:
            ov = throttle(legs, BTC_LEGS, q, w)
            cagr, dd, cd, port = book_gen(legs, ov)
            key = f"A4 q{int(q*100)}_w{w}"
            a4_overrides[key] = ov
            a4_results[key] = dict(cagr=cagr, dd=dd, cd=cd, port=port)
            row.append(f"{cd:.2f}")
        print(f"{q:<10.2f}" + "".join(f"{v:<12}" for v in row))

    # --- bootstrap: A0 vs A1 vs A2 vs A3(x3) vs A4(15 cells) ---
    print()
    print("=" * 100)
    print("TABLE 2 -- paired circular block bootstrap of the BOOK's monthly returns "
          f"({ndraw} draws/block-length, block 1/3/6/12mo)")
    print("=" * 100)
    all_names = list(arm_overrides.keys()) + list(a4_overrides.keys())
    all_results = {**results, **a4_results}
    # align all arms' monthly port series onto a common month index (should already match since
    # only magnitudes change, never the underlying date range)
    common_idx = all_results["A0 base"]["port"].index
    for nm in all_names:
        assert all_results[nm]["port"].index.equals(common_idx), f"month index mismatch in {nm}"
    months = len(common_idx)
    arm_ports = {nm: all_results[nm]["port"].values for nm in all_names}
    bt = bootstrap_table(arm_ports, months, ndraw)
    print(f"book months = {months}\n")
    print("median CAGR/DD (P beats A0 base) -- key arms:")
    key_names = list(arm_overrides.keys())
    print(f"{'block':<8}" + "".join(f"{nm[:26]:>28}" for nm in key_names))
    for blk in (1, 3, 6, 12):
        row = bt[blk]
        print(f"{f'{blk}mo':<8}" + "".join(f"{row[nm][0]:.2f} (P{row[nm][1]:.0f}%)".rjust(28) for nm in key_names))
    print("\nmedian CAGR/DD (P beats A0 base) -- A4 plateau grid:")
    a4_names = list(a4_overrides.keys())
    print(f"{'block':<8}" + "".join(f"{nm[3:]:>16}" for nm in a4_names))
    for blk in (1, 3, 6, 12):
        row = bt[blk]
        print(f"{f'{blk}mo':<8}" + "".join(f"{row[nm][0]:.2f}/P{row[nm][1]:.0f}".rjust(16) for nm in a4_names))
    print("\n  P = P(this arm's book CAGR/DD > A0 base's) on the SAME resampled months (paired). "
          "~50% = indistinguishable. A real change is consistent (P rises) as block length grows.")

    # --- per-leg flagged vs rest, and flagged-month overlap across the 4 BTC legs ---
    print()
    print("=" * 100)
    print("TABLE 3 -- per-BTC-leg flagged (IS75-hot) vs rest: n / meanR / PF")
    print("=" * 100)
    rep = leg_flag_report(legs, q=0.75)
    print(f"{'leg':<16}{'IS75 thr':>10}{'n_hot':>8}{'n_rest':>8}{'meanR_hot':>11}{'meanR_rest':>12}"
          f"{'PF_hot':>9}{'PF_rest':>9}")
    for name in BTC_LEGS:
        r = rep[name]
        print(f"{name:<16}{r['thr']:>+10.3f}{r['n_hot']:>8}{r['n_rest']:>8}{r['meanR_hot']:>+11.3f}"
              f"{r['meanR_rest']:>+12.3f}{r['pf_hot']:>9.2f}{r['pf_rest']:>9.2f}")

    print()
    print("=" * 100)
    print("TABLE 4 -- flagged-month overlap across the 4 BTC legs (Jaccard of 'has >=1 hot trade this "
          "month' sets)")
    print("=" * 100)
    months_sets = {name: rep[name]["hot_months"] for name in BTC_LEGS}
    for a in BTC_LEGS:
        for b in BTC_LEGS:
            if a >= b:
                continue
            inter = len(months_sets[a] & months_sets[b])
            union = len(months_sets[a] | months_sets[b])
            jac = inter / union if union else np.nan
            print(f"  {a:<14} vs {b:<14}  hot-months A={len(months_sets[a]):<4} B={len(months_sets[b]):<4} "
                  f"intersect={inter:<4} union={union:<4} Jaccard={jac:.2f}")
    all_inter = set.intersection(*months_sets.values())
    all_union = set.union(*months_sets.values())
    print(f"\n  all-4-legs simultaneous hot months = {len(all_inter)} / union {len(all_union)} "
          f"({100*len(all_inter)/len(all_union):.0f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    main(smoke=args.smoke)
