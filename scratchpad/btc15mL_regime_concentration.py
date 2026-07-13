"""btc15mL_regime_concentration.py -- is btc15m_L (the book's life-line leg, without-it book
CAGR/DD 8.19 -> 3.62) simply "a machine that makes money only in years BTC rallied"?  The
adoption caveat flags ~90% of its totR coming from 2020/2023/2024.  Concentration alone is not
damning -- ALL trend-followers concentrate in trend years -- so every concentration measure here
is computed side-by-side against controls: the 3 incumbent legs (gold_bo, btc_bo_kama, btc_pull),
the 2 other candidate legs (gold15m, btc15m_S), and BTC buy-and-hold itself (as a concentration
REFERENCE, not a performance benchmark: if BTC's own annual return is this concentrated, a
BTC-long strategy concentrating similarly is unsurprising).

Reuses (imports, does not reimplement):
  - scratchpad/btc_family_ext_throttle.py: build_base()      -- the 6 canonical leg R-series
  - scratchpad/book_leave_one_out.py: cdd(), weights(), stat(), NEW, OLD
  - scratchpad/book_weighting_scheme.py: monthly_matrix(), scheme_invvol_trade(),
      series_from_weights()                                   -- the specified book weighting
  - scratchpad/book_hh4h_weight_sweep.py: leg_stats()          -- standardized single-leg
      CAGR/DD at a fixed 1%-risk-per-trade compounding (existing convention, not reinvented)
  - src/data_loader.load_mt5_csv                                -- BTC daily close for the
      buy-and-hold concentration reference (section C3)

Tie-back (mandatory, printed and checked before proceeding):
  book_cdd(legs, NEW) must reproduce 6-leg CAGR/DD = 8.19
  book_cdd(legs, OLD) must reproduce 3-leg CAGR/DD = 2.71
  (both via scheme_invvol_trade weighting + trade-resolution cdd(), per the spec card --
  NOT the monthly-DD arbiter's 12.03, which is a different judge not used in this card.)

No lookahead introduced: this script only re-slices and re-aggregates already-priced,
already-timestamped trade series from build_base() (next-bar-open fill / intrabar SL-TP-priority
/ confirmed-close HTF gates already enforced upstream). Year/month bucketing uses each trade's
ENTRY time (the "time" field from breakout_wave.run(), i.e. d.index[e_bar] -- the fill bar, not a
signal bar), so no trade is bucketed by information not yet known at entry.

Run (full):  .venv/bin/python scratchpad/btc15mL_regime_concentration.py 2>/dev/null | tee scratchpad/out_btc15mL_regime_concentration.txt
Run (smoke): .venv/bin/python scratchpad/btc15mL_regime_concentration.py --smoke 2>/dev/null | tee scratchpad/out_btc15mL_regime_concentration_smoke.txt
"""
import sys, io, contextlib, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")

from btc_family_ext_throttle import build_base
from book_leave_one_out import cdd, NEW, OLD
from book_weighting_scheme import monthly_matrix, scheme_invvol_trade, series_from_weights
from book_hh4h_weight_sweep import leg_stats
from src.data_loader import load_mt5_csv

ROOT = "/home/angelbell/dev/auto-trade"
TREND_YEARS = [2020, 2023, 2024]


# ---------------------------------------------------------------------------
# book-level CAGR/DD under the spec's arbiter: scheme_invvol_trade weighting + trade-res cdd()
# ---------------------------------------------------------------------------
def book_cdd(legs_dict, basket):
    M, midx = monthly_matrix(legs_dict, basket)
    w = scheme_invvol_trade(legs_dict, basket, M, midx)
    s = series_from_weights(legs_dict, basket, w, midx)
    days = (s.index[-1] - s.index[0]).days
    c, d, x = cdd(s.values, days)
    return c, d, x, s


# ---------------------------------------------------------------------------
# per-year raw stats (n, win%, PF, meanR, totR) on a leg's own R series
# ---------------------------------------------------------------------------
def year_table(s):
    yrs = sorted(s.index.year.unique())
    rows = []
    for y in yrs:
        R = s[s.index.year == y].values
        n = len(R)
        win = (R > 0).mean() * 100
        pos, neg = R[R > 0].sum(), abs(R[R <= 0].sum())
        pf = pos / neg if neg > 0 else np.nan
        meanR = R.mean()
        totR = R.sum()
        rows.append(dict(year=y, n=n, win=win, pf=pf, meanR=meanR, totR=totR))
    return pd.DataFrame(rows)


def leg_pf_meanR_n_totR(R):
    n = len(R)
    win = (R > 0).mean() * 100 if n else np.nan
    pos, neg = R[R > 0].sum(), abs(R[R <= 0].sum())
    pf = pos / neg if neg > 0 else np.nan
    meanR = R.mean() if n else np.nan
    totR = R.sum() if n else 0.0
    return n, win, pf, meanR, totR


# ---------------------------------------------------------------------------
# concentration measures (year / month / trade) -- shared by btc15m_L and every control leg
# ---------------------------------------------------------------------------
def gini(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 or np.mean(x) == 0:
        return np.nan
    diffs = np.abs(x.reshape(-1, 1) - x.reshape(1, -1)).sum()
    return diffs / (2 * n * n * np.mean(x))


def concentration(s, name):
    R = s.values
    n = len(R)
    totR = R.sum()

    # --- year ---
    yr = s.groupby(s.index.year).sum().sort_values(ascending=False)
    yr_share = yr / totR * 100
    top3_year_share = yr_share.iloc[:3].sum()
    cum_year_curve = yr_share.cumsum()
    gini_year = gini(yr.values)

    # --- month (continuous month grid incl. zero-trade months, matches book's own month grid) ---
    mon_sparse = s.groupby(s.index.to_period("M")).sum()
    full_midx = pd.period_range(mon_sparse.index.min(), mon_sparse.index.max(), freq="M")
    mon = mon_sparse.reindex(full_midx, fill_value=0.0).sort_values(ascending=False)
    n_months = len(mon)
    top10pct_n = max(1, int(np.ceil(0.10 * n_months)))
    top10pct_share = mon.iloc[:top10pct_n].sum() / totR * 100

    # --- trade ---
    Rsort = np.sort(R)[::-1]
    top10_share = Rsort[:min(10, n)].sum() / totR * 100
    top1pct_n = max(1, int(round(0.01 * n)))
    top1pct_share = Rsort[:top1pct_n].sum() / totR * 100

    return dict(name=name, n=n, totR=totR, n_years=len(yr), top3_year_share=top3_year_share,
                top_years=list(yr.index[:3]), gini_year=gini_year, cum_year_curve=cum_year_curve,
                n_months=n_months, top10pct_n=top10pct_n, top10pct_month_share=top10pct_share,
                top10_trade_share=top10_share, top1pct_n=top1pct_n, top1pct_trade_share=top1pct_share)


def print_concentration_row(c):
    print(f"{c['name']:<16}{c['n']:>6}{c['n_years']:>7}{c['top3_year_share']:>10.1f}%"
          f"{str(c['top_years']):>18}{c['gini_year']:>8.2f}{c['top10pct_n']:>10}"
          f"{c['top10pct_month_share']:>10.1f}%{c['top10_trade_share']:>10.1f}%"
          f"{c['top1pct_n']:>9}{c['top1pct_trade_share']:>10.1f}%")


# ---------------------------------------------------------------------------
# BTC buy-and-hold: annual LOG-return decomposition (additive across years, unlike simple %)
# ---------------------------------------------------------------------------
def btc_buyhold_annual():
    full = load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv")
    dcl = full["close"].resample("1D").last().dropna()
    logret = np.log(dcl).diff().dropna()
    yr_log = logret.groupby(logret.index.year).sum()
    yr_simple = dcl.groupby(dcl.index.year).apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100)
    return dcl, yr_log, yr_simple


def main(smoke=False):
    with contextlib.redirect_stderr(io.StringIO()):
        legs = build_base()

    # =========================================================================
    print("=" * 100)
    print("TIE-BACK (mandatory)")
    print("=" * 100)
    c6, d6, x6, _ = book_cdd(legs, NEW)
    c3, d3, x3, _ = book_cdd(legs, OLD)
    print(f"6-leg book (scheme_invvol_trade + cdd): CAGR/DD={x6:.2f}  DD={d6:.2f}%   (target 8.19)")
    print(f"3-leg book (scheme_invvol_trade + cdd): CAGR/DD={x3:.2f}  DD={d3:.2f}%   (target 2.71)")
    if abs(x6 - 8.19) > 0.02 or abs(x3 - 2.71) > 0.02:
        print("\n*** TIE-BACK MISMATCH -- stopping before proceeding further. Report this. ***")
        return
    print()

    dcl, btc_yr_log, btc_yr_simple = btc_buyhold_annual()

    # =========================================================================
    print("=" * 100)
    print("TABLE A -- btc15m_L per-year raw stats + BTC's own annual return that year")
    print("=" * 100)
    L = legs["btc15m_L"]
    ta = year_table(L)
    print(f"{'year':<6}{'n':>5}{'win%':>7}{'PF':>7}{'meanR':>8}{'totR':>9}{'BTC yr ret%':>13}")
    for _, r in ta.iterrows():
        btc_r = btc_yr_simple.get(r['year'], np.nan)
        flag = "  <- partial year" if r['year'] in (2018, 2026) else ""
        print(f"{int(r['year']):<6}{int(r['n']):>5}{r['win']:>6.1f}%{r['pf']:>7.2f}{r['meanR']:>+8.3f}"
              f"{r['totR']:>+9.1f}{btc_r:>+12.1f}%{flag}")
    print("\n(2018 = Oct-Dec stub, first live quarter of the leg's history; 2026 = partial thru "
          f"{L.index.max().date()}. BTC yr ret% = close-to-close simple return over the SAME "
          "partial window in those two years, for apples-to-apples.)")

    # =========================================================================
    print()
    print("=" * 100)
    print("TABLE B -- btc15m_L concentration (year / month / trade)")
    print("=" * 100)
    cL = concentration(L, "btc15m_L")
    hdr = (f"{'leg':<16}{'n':>6}{'nYrs':>7}{'top3yr%':>10}{'which yrs':>18}{'giniYr':>8}"
           f"{'nTop10%mo':>10}{'top10%mo%':>10}{'top10trd%':>10}{'nTop1%':>9}{'top1%trd%':>10}")
    print(hdr)
    print_concentration_row(cL)
    print(f"\ncumulative year-share curve (years sorted by totR desc), btc15m_L:")
    print("  " + ", ".join(f"{y}:{v:.0f}%" for y, v in cL['cum_year_curve'].items()))

    # =========================================================================
    print()
    print("=" * 100)
    print("TABLE C -- SAME 3 measures, controls side-by-side (this is the load-bearing table)")
    print("=" * 100)
    print(hdr)
    print_concentration_row(cL)
    print("-" * len(hdr))
    controls = {}
    for name in ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_S"]:
        c = concentration(legs[name], name)
        controls[name] = c
        print_concentration_row(c)

    print("\nC3 -- BTC buy-and-hold, YEAR measure only (concentration REFERENCE, not a performance "
          "benchmark):")
    btc_yr_sorted = btc_yr_log.sort_values(ascending=False)
    btc_total_log = btc_yr_log.sum()
    btc_yr_share = btc_yr_sorted / btc_total_log * 100
    btc_top3 = btc_yr_share.iloc[:3].sum()
    btc_gini = gini(btc_yr_log.values)
    print(f"  years with data: {sorted(btc_yr_log.index.tolist())}")
    print(f"  top-3 years by annual LOG-return share of total: {btc_top3:.1f}%  "
          f"(years: {list(btc_yr_sorted.index[:3])})")
    print(f"  Gini (annual log-return shares): {btc_gini:.2f}")
    print("  cumulative year-share curve (BTC buy-hold, sorted desc):")
    print("  " + ", ".join(f"{y}:{v:.0f}%" for y, v in btc_yr_share.cumsum().items()))
    print(f"\n  btc15m_L top-3-year share = {cL['top3_year_share']:.1f}%  vs  BTC buy-hold top-3-year "
          f"share = {btc_top3:.1f}%   (delta = {cL['top3_year_share'] - btc_top3:+.1f}pp)")

    # =========================================================================
    print()
    print("=" * 100)
    print("TABLE D1 -- leave-one-year-out on btc15m_L (2019-2026, 8 years; 2018 stub excluded -- "
          "only 3 months of the leg's very first quarter, too thin to be a meaningful 'drop' cell)")
    print("=" * 100)
    d1_years = [y for y in sorted(L.index.year.unique()) if y not in (2018,)]
    print(f"{'drop yr':<9}{'n left':>7}{'leg PF':>8}{'leg CAGR/DD':>13}{'6-leg book CAGR/DD':>20}"
          f"{'book DD%':>10}{'delta vs 8.19':>14}")
    for y in d1_years:
        Ld = L[L.index.year != y]
        pf_d, dd_d, cd_d = leg_stats(Ld)
        legs2 = dict(legs); legs2["btc15m_L"] = Ld
        c_b, d_b, x_b, _ = book_cdd(legs2, NEW)
        print(f"{y:<9}{len(Ld):>7}{pf_d:>8.2f}{cd_d:>13.2f}{x_b:>20.2f}{d_b:>9.2f}%{x_b - x6:>+14.2f}")

    # =========================================================================
    print()
    print("=" * 100)
    print(f"TABLE D2 -- btc15m_L with {TREND_YEARS} ALL removed (remaining: "
          f"{[y for y in d1_years if y not in TREND_YEARS]})")
    print("=" * 100)
    L_notrend = L[~L.index.year.isin(TREND_YEARS)]
    n2, win2, pf2, meanR2, totR2 = leg_pf_meanR_n_totR(L_notrend.values)
    pf2b, dd2b, cd2b = leg_stats(L_notrend)
    print(f"n={n2}  win%={win2:.1f}  PF={pf2:.2f}  meanR={meanR2:+.3f}  totR={totR2:+.1f}  "
          f"leg CAGR/DD={cd2b:.2f}  (leg_stats PF check={pf2b:.2f}, should match PF above)")
    verdict_L = "PF < 1.0 -> 上げ年専用の機械" if pf2 < 1.0 else "PF > 1.0 -> 集中はしているが土台はある"
    print(f"  -> {verdict_L}")

    # =========================================================================
    print()
    print("=" * 100)
    print("TABLE D3 -- SAME year-drop test on the 3 incumbent legs, using EACH LEG'S OWN top-3 "
          "totR years (not literally 2020/2023/2024 -- gold_bo's trend years need not be BTC's)")
    print("=" * 100)
    for name in ["gold_bo", "btc_bo_kama", "btc_pull"]:
        s = legs[name]
        own_top3 = controls[name]['top_years']
        s_rest = s[~s.index.year.isin(own_top3)]
        n_, win_, pf_, meanR_, totR_ = leg_pf_meanR_n_totR(s_rest.values)
        pf_b, dd_b, cd_b = leg_stats(s_rest)
        remaining_years = sorted(set(s.index.year.unique()) - set(own_top3))
        print(f"\n{name}: own top-3 totR years = {own_top3}  (dropped); remaining years = "
              f"{remaining_years}")
        print(f"  n={n_}  win%={win_:.1f}  PF={pf_:.2f}  meanR={meanR_:+.3f}  totR={totR_:+.1f}  "
              f"leg CAGR/DD={cd_b:.2f}")
        v = "PF < 1.0 -> このレッグも上げ(強)年専用" if pf_ < 1.0 else "PF > 1.0 -> 土台はある"
        print(f"  -> {v}")

    # =========================================================================
    print()
    print("=" * 100)
    print("TABLE E -- btc15m_L + btc15m_S combined, per-year totR (raw R units, equal risk-scale)")
    print("=" * 100)
    S = legs["btc15m_S"]
    all_years = sorted(set(L.index.year.unique()) | set(S.index.year.unique()))
    print(f"{'year':<6}{'L totR':>9}{'S totR':>9}{'L+S totR':>10}{'BTC yr ret%':>13}")
    for y in all_years:
        lr = L[L.index.year == y].values.sum()
        sr = S[S.index.year == y].values.sum()
        btc_r = btc_yr_simple.get(y, np.nan)
        bear_flag = "  <- BTC down that (partial) year" if (not np.isnan(btc_r) and btc_r < 0) else ""
        print(f"{y:<6}{lr:>+9.1f}{sr:>+9.1f}{lr+sr:>+10.1f}{btc_r:>+12.1f}%{bear_flag}")

    print()
    print("=" * 100)
    print("done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    main(smoke=args.smoke)
