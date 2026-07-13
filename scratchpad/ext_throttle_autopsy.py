"""AUTOPSY of the 4-week-extension throttle: is the book gain a MECHANISM or ONE DRAWDOWN?

Red flag from the verification grid: flagging 21 / 45 / 77 / 118 / 148 / 179 trades all move the
book to ~13.4, and the book's DD only moves 3.6% -> 3.4% while CAGR barely changes. If the mechanism
("BTC longs entered after a big 4-week run are weaker") were real, the gain should ACCUMULATE with
the number of trades down-weighted. It does not. That is the signature of a single drawdown episode
being deleted -- CAGR/DD with a 3.6% denominator is hypersensitive to one episode.

Three tests, in order:
  T1  Decompose: book CAGR and DD separately for each arm. If CAGR is flat and only DD moves, the
      "gain" is a denominator effect.
  T2  Locate the book's max-DD window under the base, then ask: what fraction of the flagged trades'
      NEGATIVE R lands inside it?  And does the throttle simply erase that window?
  T3  The honest control: instead of the ret4w rule, drop a RANDOM set of the SAME SIZE from
      btc15m_L's trades that fall inside the base's max-DD window. If a random cut of the drawdown
      window reproduces ~13.4, the ret4w variable is doing NO work -- any rule that happens to touch
      that episode would "win".
Run: .venv/bin/python scratchpad/ext_throttle_autopsy.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from btc_family_ext_throttle import build_base, book_gen, ret4w_daily

NEW = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]


def monthly(legs, override=None):
    L = dict(legs)
    if override is not None:
        L["btc15m_L"] = override
    mon = {k: s.groupby(s.index.to_period("M")).sum() for k, s in L.items()}
    st = max(s.index.min() for s in mon.values()); en = min(s.index.max() for s in mon.values())
    midx = pd.period_range(st, en, freq="M")
    M = pd.DataFrame({k: v.reindex(midx, fill_value=0.0) for k, v in mon.items()})
    sig = M.std(); w = (1.0 / sig[NEW]); w = w / w.sum() * 0.03
    return (M[NEW] * w).sum(axis=1)


def cagr_dd(port):
    eq = np.cumprod(1 + port.values); pk = np.maximum.accumulate(eq)
    ddser = (pk - eq) / pk
    dd = ddser.max()
    cagr = eq[-1] ** (12 / len(port)) - 1
    trough = int(np.argmax(ddser))
    peak = int(np.argmax(eq[:trough + 1])) if trough > 0 else 0
    return cagr * 100, dd * 100, (cagr * 100) / (dd * 100), port.index[peak], port.index[trough]


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        legs = build_base()
    L = legs["btc15m_L"]                       # already PDH-weighted R series
    r4 = ret4w_daily()
    v = r4.reindex(L.index, method="ffill")
    half = L.index[len(L) // 2]

    base = monthly(legs)
    c0, d0, cd0, pk0, tr0 = cagr_dd(base)
    print("T1 -- decomposition (does CAGR move, or only DD?)")
    print(f"    {'arm':<26}{'n_hot':>7}{'book CAGR':>11}{'book DD':>9}{'CAGR/DD':>9}")
    print(f"    {'A0 base':<26}{0:>7}{c0:>10.2f}%{d0:>8.2f}%{cd0:>9.2f}")
    rows = {}
    for q in (0.60, 0.75, 0.90, 0.95):
        thr = np.nanquantile(v[L.index < half], q)
        hot = (v >= thr).values & np.isfinite(v.values)
        for w in (0.5, 0.0):
            s = pd.Series(L.values * np.where(hot, w, 1.0), index=L.index)
            c, d, cd, _, _ = cagr_dd(monthly(legs, s))
            rows[(q, w)] = (hot.sum(), c, d, cd)
            print(f"    {f'q{q} w={w}':<26}{hot.sum():>7}{c:>10.2f}%{d:>8.2f}%{cd:>9.2f}")

    print(f"\nT2 -- the base book's max drawdown: peak {pk0} -> trough {tr0}"
          f"  (depth {d0:.2f}%)")
    win = (L.index.to_period("M") >= pk0) & (L.index.to_period("M") <= tr0)
    thr75 = np.nanquantile(v[L.index < half], 0.75)
    hot75 = (v >= thr75).values & np.isfinite(v.values)
    ddwin_R = L.values[win]
    print(f"    btc15m_L trades inside that window : n={win.sum()}  sumR {ddwin_R.sum():+.1f}")
    print(f"    of which flagged hot (q0.75)       : n={int((win & hot75).sum())}  "
          f"sumR {L.values[win & hot75].sum():+.1f}")
    print(f"    flagged hot OUTSIDE the window     : n={int((~win & hot75).sum())}  "
          f"sumR {L.values[~win & hot75].sum():+.1f}")
    tot_neg = L.values[hot75][L.values[hot75] < 0].sum()
    win_neg = L.values[win & hot75][L.values[win & hot75] < 0].sum()
    print(f"    share of the flagged set's LOSSES that sit in the DD window: "
          f"{100 * win_neg / tot_neg:.0f}%")

    print("\nT3 -- control: drop a RANDOM equal-sized set of btc15m_L trades from INSIDE the")
    print("     base's max-DD window.  If this reproduces the 'gain', ret4w does no work.")
    rng = np.random.default_rng(7)
    idx_win = np.where(win)[0]
    for n_drop in (21, 45, 148):
        if n_drop > len(idx_win):
            print(f"    drop {n_drop:>3} random in-window trades : only {len(idx_win)} exist -- skip")
            continue
        vals = []
        for _ in range(400):
            pick = rng.choice(idx_win, size=n_drop, replace=False)
            w_ = np.ones(len(L)); w_[pick] = 0.0
            s = pd.Series(L.values * w_, index=L.index)
            vals.append(cagr_dd(monthly(legs, s))[2])
        vals = np.array(vals)
        print(f"    drop {n_drop:>3} random in-window trades : book CAGR/DD median {np.median(vals):.2f}"
              f"   [p10 {np.percentile(vals,10):.2f}, p90 {np.percentile(vals,90):.2f}]")
    print(f"\n    for reference: the ret4w rule at q0.75 w=0.0 gives {rows[(0.75, 0.0)][3]:.2f}, "
          f"q0.95 w=0.0 gives {rows[(0.95, 0.0)][3]:.2f}, base = {cd0:.2f}")


if __name__ == "__main__":
    main()
