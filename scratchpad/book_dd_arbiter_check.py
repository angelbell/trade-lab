"""IS THE BOOK ARBITER ITSELF BROKEN?

Every book verdict made today (RR4.5 adopted, HH4H sizing rejected, the 4-week-extension throttle
"improving" 12.03 -> 12.70) was read off `book()`, which:
   1. aggregates each leg's R into MONTHLY sums,
   2. inv-vol weights the 6 legs to 3% total risk,
   3. compounds the MONTHLY series and takes maxDD of the MONTHLY equity.

Step 3 is the problem. The autopsy shows the resulting maxDD is a SINGLE MONTH (2019-06 -> 2019-07,
depth 3.62%), producing CAGR 43.6% / DD 3.6% = a Calmar of 12 for a 6-leg book at 3% risk. That is
not a believable drawdown -- it is the artifact of never looking inside a month. And it means every
"CAGR/DD 12.03 vs 12.70" comparison is a comparison of ONE month's composition.

This script rebuilds the same book but measures the drawdown on a DAILY equity curve (each trade's
weighted R booked on its own day, compounded), and re-ranks the arms.  If the ranking survives, the
verdicts stand and only the DD magnitude was wrong. If it flips, today's book verdicts must be redone.

NOTE on the daily curve: leg R series are indexed by ENTRY time, so a trade's P&L is booked at entry.
That is the same convention the monthly aggregation already uses -- this changes only the resolution
of the drawdown measurement, nothing else, so the comparison is apples-to-apples.

Run: .venv/bin/python scratchpad/book_dd_arbiter_check.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from btc_family_ext_throttle import build_base, ret4w_daily
from breakout_wave import run, resample, swings_zigzag
from trend_leg_aging import atr as atr_fn
from src.data_loader import load_mt5_csv

NEW = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]
ROOT = "/home/angelbell/dev/auto-trade"


def weights(L):
    """the book's inv-vol weights (identical to book(): monthly sigma, 3% total risk)."""
    mon = {k: s.groupby(s.index.to_period("M")).sum() for k, s in L.items()}
    st = max(s.index.min() for s in mon.values()); en = min(s.index.max() for s in mon.values())
    midx = pd.period_range(st, en, freq="M")
    M = pd.DataFrame({k: v.reindex(midx, fill_value=0.0) for k, v in mon.items()})
    sig = M.std(); w = (1.0 / sig[NEW]); w = w / w.sum() * 0.03
    return w, M, midx


def stats(L, freq):
    """book CAGR / maxDD / CAGR-DD at the given aggregation frequency."""
    w, M, midx = weights(L)
    st = midx[0].to_timestamp().tz_localize("UTC")
    en = midx[-1].to_timestamp(how="end").tz_localize("UTC")
    if freq == "M":
        port = (M[NEW] * w).sum(axis=1).values
        n_per_yr = 12
    else:
        acc = {}
        for k in NEW:
            s = L[k]
            s = s[(s.index >= st) & (s.index <= en)]
            acc[k] = s.groupby(s.index.floor("D")).sum() * w[k]
        D = pd.DataFrame(acc)
        didx = pd.date_range(st.floor("D"), en.floor("D"), freq="D")
        D = D.reindex(didx, fill_value=0.0).fillna(0.0)
        port = D.sum(axis=1).values
        n_per_yr = 365.25
    eq = np.cumprod(1 + port); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    cagr = (eq[-1] ** (n_per_yr / len(port)) - 1) * 100
    return cagr, dd, cagr / dd


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        legs = build_base()
    L0 = legs["btc15m_L"]
    v = ret4w_daily().reindex(L0.index, method="ffill")
    half = L0.index[len(L0) // 2]

    def hot_at(q):
        thr = np.nanquantile(v[L0.index < half], q)
        return (v >= thr).values & np.isfinite(v.values)

    arms = {"A0 base (adopted)": L0}
    for q, w in ((0.75, 0.5), (0.75, 0.0), (0.90, 0.0), (0.60, 0.0)):
        arms[f"ext-throttle q{q} w={w}"] = pd.Series(
            L0.values * np.where(hot_at(q), w, 1.0), index=L0.index)

    # the arm today's lab REJECTED, for a sanity check of the re-ranking
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        h4 = d15.resample("4h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        a4 = atr_fn(h4["high"].values, h4["low"].values, h4["close"].values)
        sh = pd.Series(np.nan, index=h4.index)
        for (ci, pi, px, kind) in swings_zigzag(h4["high"].values, h4["low"].values, a4, 2.0):
            if kind == +1:
                sh.iloc[ci] = px

    print(f"{'arm':<26}{'MONTHLY-DD arbiter (today)':>34}{'DAILY-DD arbiter':>30}")
    print(f"{'':<26}{'CAGR':>10}{'maxDD':>9}{'C/DD':>8}{'   |':>4}{'CAGR':>10}{'maxDD':>9}{'C/DD':>8}")
    rank_m, rank_d = {}, {}
    for k, s in arms.items():
        L = dict(legs); L["btc15m_L"] = s[s != 0] if (s == 0).any() else s
        cm, dm, xm = stats(L, "M")
        cd, dd, xd = stats(L, "D")
        rank_m[k], rank_d[k] = xm, xd
        print(f"{k:<26}{cm:>9.1f}%{dm:>8.2f}%{xm:>8.2f}{'   |':>4}{cd:>9.1f}%{dd:>8.2f}%{xd:>8.2f}")

    print("\nranking under the MONTHLY-DD arbiter (today's judge):")
    for i, (k, x) in enumerate(sorted(rank_m.items(), key=lambda t: -t[1]), 1):
        print(f"  {i}. {k:<28}{x:.2f}")
    print("ranking under the DAILY-DD arbiter:")
    for i, (k, x) in enumerate(sorted(rank_d.items(), key=lambda t: -t[1]), 1):
        print(f"  {i}. {k:<28}{x:.2f}")

    # how concentrated is the monthly DD?  (peak->trough length, and the worst months)
    w_, M, midx = weights(legs)
    port = (M[NEW] * w_).sum(axis=1)
    eq = np.cumprod(1 + port.values); pk = np.maximum.accumulate(eq)
    ddser = (pk - eq) / pk
    tr = int(np.argmax(ddser)); pkm = int(np.argmax(eq[:tr + 1]))
    print(f"\nMONTHLY equity: maxDD {ddser.max()*100:.2f}%  peak {midx[pkm]} -> trough {midx[tr]}  "
          f"= {tr - pkm} month(s) long")
    worst = port.sort_values().head(5)
    print("worst 5 months of the book:  " + "  ".join(f"{p}:{100*r:+.2f}%" for p, r in worst.items()))
    print(f"negative months: {(port < 0).sum()} / {len(port)}")


if __name__ == "__main__":
    main()
