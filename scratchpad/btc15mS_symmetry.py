"""btc15mS_symmetry.py -- does btc15m_S deserve the same 3 treatments that just improved
btc15m_L (gate TF, RR, soft-vs-hard prior-day-level sizing), or were those never tried on
the short leg?  Long side today: (a) daily->4h gate won on meanR/PF/book AND broke regime
concentration, (b) RR4.0->4.5 passed the book arbiter, (c) soft PDH sizing beat hard PDL-style
filtering.  Short side has never been swept on any of the three.  This script sweeps all
three axes on btc15m_S ALONE, holding the other 5 legs of the 6-leg book fixed, and reports
whether the long-side wins repeat on the short side or not.

Machinery reused, not reinvented:
  - breakout_wave.run/resample (the actual trade engine)
  - short_mirror_15m.invert (the SHORT = LONG-on-inverted-price mechanism)
  - radar_gate_race.BASE (the frozen entry/exit parameter set)
  - book_spec_fix.build/book/w_trade (the 6-leg book + trade-resolution DD arbiter)
  - book_leave_one_out.cdd (CAGR/maxDD/CAGR-DD from an R series)

Tie-back gate: book_spec_fix.build("2018-01-01", False) -> book(L, NEW) must equal 8.27
(the number the spec card was frozen against). Script aborts loudly if it doesn't.

Run:
  .venv/bin/python scratchpad/btc15mS_symmetry.py --smoke   (fast sanity pass, coarse only)
  .venv/bin/python scratchpad/btc15mS_symmetry.py           (full spec: S1-S4 + bootstrap)
"""
import argparse
import contextlib
import io
import sys
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from src.data_loader import load_mt5_csv
    from breakout_wave import run, resample
    from radar_gate_race import BASE
    from short_mirror_15m import invert
    from book_spec_fix import build, book, w_trade
    from book_leave_one_out import cdd

ROOT = "/home/angelbell/dev/auto-trade"
NEW = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]
NDRAW = 2000


# ---------------------------------------------------------------- machine ---
def make_S(d15, gate_tf="1D", rr=4.0, pdl_w=None):
    """Rebuild btc15m_S with (gate TF, RR, PDL treatment) swapped in.
    pdl_w=None -> hard filter (current: drop rows not below PDL, like build()).
    pdl_w=float -> soft: full weight if below PDL, pdl_w weight otherwise (all rows kept).
    Everything else (gate_kama=14, pullback_frac=0.3, PDL definition itself) is untouched.
    """
    inv = invert(d15)
    C = 2 * d15["high"].max()
    with contextlib.redirect_stdout(io.StringIO()):
        ts = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": gate_tf,
                                         "pullback_frac": 0.3, "rr": rr}))
    if ts is None or len(ts) < 5:
        return pd.Series(dtype=float)
    Rs = ts["R"].values - 15.0 / ts["risk"].values
    idx = pd.DatetimeIndex(ts["time"])
    pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
    ei = d15.index.get_indexer(ts["time"])
    below = (C - ts["e_px"].values) < pdl[ei]
    if pdl_w is None:
        return pd.Series(Rs[below], index=idx[below])
    w = np.where(below, 1.0, pdl_w)
    return pd.Series(Rs * w, index=idx)


def leg_metrics(s, span):
    if len(s) < 5:
        return dict(n=len(s), npyr=np.nan, win=np.nan, pf=np.nan, meanR=np.nan,
                    isoos="n/a", cagr_dd=np.nan)
    yr = s.index.year.values
    half = np.median(yr)
    pos, neg = s[s > 0].sum(), abs(s[s <= 0].sum())
    pf = pos / neg if neg > 0 else np.inf
    eq = np.cumprod(1 + 0.01 * s.values)
    pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    days = (s.index[-1] - s.index[0]).days
    cagr = (eq[-1] ** (365.25 / max(days, 1)) - 1) * 100
    cagr_dd = cagr / dd if dd > 0 else np.nan
    return dict(n=len(s), npyr=len(s) / span, win=(s > 0).mean() * 100, pf=pf,
                meanR=s.mean(), isoos=f"{s[yr < half].mean():+.3f}/{s[yr >= half].mean():+.3f}",
                cagr_dd=cagr_dd)


def fmt_leg(tag, m):
    return (f"  {tag:<28}n={m['n']:>4} N/yr={m['npyr']:>5.1f} win={m['win']:>4.1f}% "
            f"PF={m['pf']:>4.2f} meanR={m['meanR']:>+.3f} IS/OOS={m['isoos']:<15} "
            f"CAGR/DD(leg)={m['cagr_dd']:>5.2f}")


def yearly_row(s):
    yr = s.groupby(s.index.year).sum()
    n_yr = s.groupby(s.index.year).size()
    return yr, n_yr


def print_yearly(tag, s):
    yr, n_yr = yearly_row(s)
    parts = " ".join(f"{y}:{yr[y]:+.1f}(n{n_yr[y]})" for y in yr.index)
    print(f"    {tag:<20} {parts}")
    for y in (2022, 2025):
        if y in yr.index:
            print(f"      -> {y}年: totR={yr[y]:+.2f}  n={n_yr[y]}")
        else:
            print(f"      -> {y}年: データ無し")


def book_with(L, s):
    LL = dict(L)
    LL["btc15m_S"] = s
    return book(LL, NEW)   # (cagr, dd, cagr/dd, n)


def book_series(legs, basket, budget=0.03):
    """Reproduce book()'s internal weighted series (book() itself only returns the
    cdd summary tuple). Tie-back checked against book()'s own cagr/dd/x below."""
    w = w_trade(legs, basket)
    st = max(legs[k].index.min() for k in basket)
    en = min(legs[k].index.max() for k in basket)
    parts = []
    for k in basket:
        s = legs[k][(legs[k].index >= st) & (legs[k].index <= en)]
        parts.append(pd.Series(s.values * w[k], index=s.index))
    return pd.concat(parts).sort_index()


def boot_pair(sA, sB, ndraw=NDRAW, seed=20260713):
    months = sorted(set(sA.index.to_period("M")) | set(sB.index.to_period("M")))
    m = len(months)
    GA = {p: g.values for p, g in sA.groupby(sA.index.to_period("M"))}
    GB = {p: g.values for p, g in sB.groupby(sB.index.to_period("M"))}
    rng = np.random.default_rng(seed)
    out = {}
    for blk in (1, 3, 6, 12):
        nb = int(np.ceil(m / blk))
        DA, DB = [], []
        for _ in range(ndraw):
            st = rng.integers(0, m, nb)
            order = [months[(s + j) % m] for s in st for j in range(blk)][:m]
            vA = np.concatenate([GA[p] for p in order if p in GA])
            vB = np.concatenate([GB[p] for p in order if p in GB])
            DA.append(cdd(vA, 365.25 * m / 12)[2])
            DB.append(cdd(vB, 365.25 * m / 12)[2])
        DA, DB = np.array(DA), np.array(DB)
        out[blk] = (np.nanmedian(DA), np.nanmedian(DB), np.nanmean(DB > DA) * 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fast partial sweep to sanity-check the machine")
    args = ap.parse_args()

    with contextlib.redirect_stderr(io.StringIO()):
        L0 = build("2018-01-01", False)
    span_full = (L0["btc15m_S"].index[-1] - L0["btc15m_S"].index[0]).days / 365.25

    # ---- tie-back --------------------------------------------------------
    c0 = book(L0, NEW)
    print(f"tie-back: build('2018-01-01', False) -> book(NEW) CAGR/DD = {c0[2]:.2f}  "
          f"(spec card says 8.27){'  OK' if abs(c0[2] - 8.27) < 0.01 else '  *** MISMATCH ***'}")
    if abs(c0[2] - 8.27) >= 0.01:
        print("ABORTING: tie-back failed, machine does not match the frozen spec card.")
        return

    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    span = (d15.index[-1] - d15.index[0]).days / 365.25

    if args.smoke:
        d15 = d15.loc["2023-01-01":]
        span = (d15.index[-1] - d15.index[0]).days / 365.25
        print(f"[SMOKE] d15 truncated to {d15.index[0].date()} -> {d15.index[-1].date()} ({span:.1f}yr)")

    cur_s = make_S(d15, "1D", 4.0, None)          # current implementation, reproduced
    cur_leg = leg_metrics(cur_s, span)
    print(f"\n現行 btc15m_S 再現チェック: n={cur_leg['n']} (build() 内の n={len(L0['btc15m_S'])})")

    # ============================================================ S1 =====
    print("\n" + "=" * 100)
    print("S1  ゲートTF   (RR=4.0現行, PDLハード現行のまま; ゲートTFだけ差し替え)")
    gate_tfs = ["1D", "240min"] if args.smoke else ["1D", "240min", "480min"]
    gate_results = {}
    for gtf in gate_tfs:
        s = make_S(d15, gtf, 4.0, None)
        m = leg_metrics(s, span)
        gate_results[gtf] = (s, m)
        tag = f"gate={gtf}" + ("(現行)" if gtf == "1D" else "")
        print(fmt_leg(tag, m))
        if len(s) >= 5:
            b = book_with(L0, s)
            print(f"    -> 6レッグブック CAGR/DD={b[2]:.2f}  (DD={b[1]:.2f}%, n={b[3]})")
            print_yearly(tag, s)
        else:
            print("    -> n不足でブック計算スキップ")

    valid_gates = {g: r for g, r in gate_results.items() if len(r[0]) >= 5}
    best_gate = max(valid_gates, key=lambda g: book_with(L0, valid_gates[g][0])[2])
    print(f"\nS1 最良ゲートTF (6レッグブックCAGR/DD基準): {best_gate}")

    # ============================================================ S2 =====
    print("\n" + "=" * 100)
    print("S2  RR スイープ (0.25刻み粗さがし -> 最良点周り0.5を0.1刻み; PDLハード現行のまま)")

    def rr_sweep(gtf, rrs, label):
        rows = []
        for rr in rrs:
            s = make_S(d15, gtf, rr, None)
            m = leg_metrics(s, span)
            b = book_with(L0, s) if len(s) >= 5 else (np.nan, np.nan, np.nan, 0)
            rows.append((rr, m, b, s))
            print(f"  [{label}] RR={rr:>4.2f}  n={m['n']:>4} N/yr={m['npyr']:>5.1f} "
                  f"win={m['win']:>4.1f}% PF={m['pf']:>4.2f} meanR={m['meanR']:>+.3f}  "
                  f"leg CAGR/DD={m['cagr_dd']:>5.2f}  6-leg book CAGR/DD={b[2]:>5.2f}")
        return rows

    def coarse_then_fine(gtf, label):
        coarse_rrs = [round(3.0 + 0.25 * i, 2) for i in range(13)]  # 3.00..6.00
        if args.smoke:
            coarse_rrs = coarse_rrs[::3]
        rows = rr_sweep(gtf, coarse_rrs, label + " 粗")
        valid = [r for r in rows if r[2][3] > 0 and not np.isnan(r[2][2])]
        if not valid:
            print("  (有効な点なし)")
            return rows, None
        best = max(valid, key=lambda r: r[2][2])
        best_rr = best[0]
        print(f"  -> [{label}] 粗探索の最良点: RR={best_rr}  book CAGR/DD={best[2][2]:.2f}")
        lo, hi = round(best_rr - 0.5, 2), round(best_rr + 0.5, 2)
        fine_rrs = sorted(set(round(lo + 0.1 * i, 2) for i in range(int(round((hi - lo) / 0.1)) + 1)))
        if args.smoke:
            fine_rrs = fine_rrs[::3]
        rows2 = rr_sweep(gtf, fine_rrs, label + " 細")
        valid2 = [r for r in rows2 if r[2][3] > 0 and not np.isnan(r[2][2])]
        best2 = max(valid2, key=lambda r: r[2][2]) if valid2 else best
        print(f"  -> [{label}] 細探索の最良点: RR={best2[0]}  book CAGR/DD={best2[2][2]:.2f}")
        return rows + rows2, best2

    print("\n-- gate=1D (現行) 上のRRスイープ --")
    rows_1d, best_1d = coarse_then_fine("1D", "gate=1D")

    rows_best_gate, best_bg = (None, None)
    if best_gate != "1D":
        print(f"\n-- gate={best_gate} (S1最良) 上のRRスイープ (交互作用チェック) --")
        rows_best_gate, best_bg = coarse_then_fine(best_gate, f"gate={best_gate}")
    else:
        print("\n(S1最良ゲート=1D=現行のため、gate側のRR再スイープは省略=同一)")
        best_bg = best_1d

    overall_best = best_bg if (best_bg is not None and (best_1d is None or best_bg[2][2] >= best_1d[2][2])) else best_1d
    best_rr = overall_best[0] if overall_best else 4.0
    best_rr_gate = best_gate if (rows_best_gate is not None and overall_best is best_bg) else "1D"
    print(f"\nS2 総合最良: gate={best_rr_gate}  RR={best_rr}  book CAGR/DD={overall_best[2][2]:.2f}"
          if overall_best else "\nS2: 有効点なし、RR=4.0(現行)を維持")

    # ============================================================ S3 =====
    print("\n" + "=" * 100)
    print("S3  PDL の扱い (gate=1D 現行, RR=4.0 現行のまま; PDLだけ差し替え)")
    pdl_modes = [("ハード・フィルタ(現行)", None), ("ソフト0.5", 0.5),
                 ("ソフト0.75", 0.75), ("無し(全部フル)", 1.0)]
    pdl_results = {}
    for tag, w in pdl_modes:
        s = make_S(d15, "1D", 4.0, w)
        m = leg_metrics(s, span)
        pdl_results[tag] = (s, m)
        print(fmt_leg(tag, m))
        if len(s) >= 5:
            b = book_with(L0, s)
            print(f"    -> 6レッグブック CAGR/DD={b[2]:.2f}  (DD={b[1]:.2f}%, n={b[3]})")

    best_pdl_tag = max(pdl_results, key=lambda t: book_with(L0, pdl_results[t][0])[2]
                        if len(pdl_results[t][0]) >= 5 else -999)
    best_pdl_w = dict(pdl_modes)[best_pdl_tag]
    print(f"\nS3 最良PDL扱い: {best_pdl_tag}")

    # ============================================================ S4 =====
    print("\n" + "=" * 100)
    print("S4  最良の組み合わせ")
    print(f"  gate={best_rr_gate}  RR={best_rr}  PDL={best_pdl_tag}")
    s_best = make_S(d15, best_rr_gate, best_rr, best_pdl_w)
    m_best = leg_metrics(s_best, span)
    print(fmt_leg("最良の組み合わせ", m_best))
    print(fmt_leg("現行 (1D/4.0/ハード)", cur_leg))
    b_best = book_with(L0, s_best)
    b_cur = book_with(L0, cur_s)
    print(f"  6レッグブック CAGR/DD: 現行={b_cur[2]:.2f}  最良={b_best[2]:.2f}  (差={b_best[2]-b_cur[2]:+.2f})")

    print("\n年別 totR (現行 vs 最良):")
    print_yearly("現行", cur_s)
    print_yearly("最良", s_best)

    print("\nブロック・ブートストラップ (最良の6レッグブック series vs 現行の6レッグブック series)")
    sA = book_series(dict(L0, btc15m_S=cur_s), NEW)
    sB = book_series(dict(L0, btc15m_S=s_best), NEW)
    # tie-back: cdd(sA) point estimate should equal book_with(L0, cur_s)
    ptA = cdd(sA.values, (sA.index[-1] - sA.index[0]).days)
    print(f"  tie-back series点推定: 現行 series経由 CAGR/DD={ptA[2]:.2f}  "
          f"(book()経由={b_cur[2]:.2f}){'  OK' if abs(ptA[2]-b_cur[2])<0.01 else '  *** MISMATCH ***'}")
    res = boot_pair(sA, sB, ndraw=(300 if args.smoke else NDRAW))
    print(f"  {'block':<8}{'現行 median':>16}{'最良 median':>16}{'P(最良>現行)':>16}")
    for blk, (da, db, p) in res.items():
        print(f"  {str(blk)+'mo':<8}{da:>16.2f}{db:>16.2f}{p:>15.1f}%")

    print("\n多重比較の注記: このS4は 3ゲート x 約24RR点 x 4PDL扱い の探索から選ばれた最良点であり、"
          "N点の中の最大値を報告している。上のブートストラップPはその選択バイアスを割り引いていない"
          "（現行 vs 最良を独立仮説として比べているだけ）。")


if __name__ == "__main__":
    main()
