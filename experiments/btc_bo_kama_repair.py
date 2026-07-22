"""btc_bo_kama_repair.py -- is the btc_bo_kama leg (BTC 4H breakout + daily-KAMA gate, RR2.0)
repairable, or is it a single-RR spike sitting on n=70?

Frozen spec card (2026-07-13). Reuses, does NOT reimplement:
  - breakout_wave.run / resample                      (research/portfolio_kama.py's own imports)
  - research.regime_gate_lab.CFG / at                 (the base breakout config + causal-align helper)
  - research.regime_adaptive.kama                     (the KAMA formula used by kama_gate_btc)
  - experiments.book_spec_fix.build/book/w_trade/OLD/NEW  (the corrected 6-leg machine, tie-back gate)
  - experiments.book_leave_one_out.cdd                  (trade-resolution CAGR/DD)
  - experiments.short_mirror_15m.invert                 (price inversion for the short mirror, E4)

E1: RR 1.5->3.0 step 0.1 (16 pts) on the leg itself, holding zz_k/trend_ema/bo_window/tf at
    current values. Reports win%, median R, PF, meanR, CAGR/DD (leg), plus 3-leg & 6-leg book
    CAGR/DD with that RR substituted for btc_bo_kama. Block bootstrap on RR2.0 vs the best point.
E2: one-axis-at-a-time frequency sweep (zz_k / trend_ema / bo_window / tf), RR held at 2.0.
E3: shrink btc_bo_kama's inv-vol weight to {1,.75,.5,.35,.25}x, reallocate the freed budget to
    the other 5 legs pro-rata to their OWN inv-vol weights, total risk budget fixed at 3%.
E4: short mirror -- invert BTC 4H bars, KAMA(14)-falling gate (== KAMA-rising on the inverted
    series, by construction), RR2, otherwise identical machine. 7-leg book (6-leg + short).

Run: .venv/bin/python experiments/btc_bo_kama_repair.py [--smoke]
"""
import sys, os, argparse, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd

ROOT = "/home/angelbell/dev/auto-trade"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/experiments")

from src.data_loader import load_mt5_csv
from breakout_wave import run as run_bo, resample
from research.regime_gate_lab import CFG, at
from research.regime_adaptive import kama
from book_spec_fix import build, book, w_trade, OLD, NEW
from book_leave_one_out import cdd
from short_mirror_15m import invert

BTC_H1 = f"{ROOT}/data/vantage_btcusd_h1.csv"
NDRAW = 2000
SEED = 20260713


# ---------------------------------------------------------------- leg construction ----------

def kama_gate(t, d, n=14):
    """Same gate as research/portfolio_kama.kama_gate_btc, generalized to any leg-TF frame d:
    daily-KAMA(n) rising (confirmed prior day, ffilled) -> keep. shift(1)+ffill = no lookahead."""
    dc = d["close"].resample("1D").last().dropna()
    km = kama(dc, n)
    return t[at((km > km.shift(1)).shift(1), t.time)]


def make_leg(rr=2.0, zz_k=2.0, trend_ema=80, bo_window=20, tf="4h"):
    """btc_bo_kama with one or more params overridden; everything else = CFG (research/
    regime_gate_lab.CFG), matching research/portfolio_kama.get_legs()'s btc_bo_kama exactly
    at the default point (tf=4h, rr=2.0, zz_k=2.0, trend_ema=80, bo_window=20)."""
    d = resample(load_mt5_csv(BTC_H1), tf)
    args = SimpleNamespace(**{**CFG, "csv": "x", "tf": tf, "rr": rr, "fwd": 300,
                               "zz_k": zz_k, "trend_ema": trend_ema, "bo_window": bo_window})
    t = run_bo(d, args)
    if t is None or len(t) == 0:
        return pd.DataFrame(columns=["time", "R"])
    t = t[["time", "R"]]
    return kama_gate(t, d)


def to_series(t):
    return pd.Series(t.R.values, index=pd.DatetimeIndex(t.time))


def leg_stats(t):
    if len(t) < 3:
        return None
    s = t.sort_values("time")
    span = max((s.time.iloc[-1] - s.time.iloc[0]).days / 365.25, 0.1)
    win = (s.R > 0).mean() * 100
    med = s.R.median()
    pos, neg = s.R[s.R > 0].sum(), s.R[s.R <= 0].sum()
    pf = pos / abs(neg) if neg < 0 else np.nan
    meanR = s.R.mean()
    sd = s.R.std()
    eq = (1 + 0.01 * s.R).cumprod()
    dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100
    cagr = (eq.iloc[-1] ** (1 / span) - 1) * 100
    cddv = cagr / max(dd, 1e-9)
    return dict(n=len(s), npy=len(s) / span, win=win, med=med, sd=sd, pf=pf,
                meanR=meanR, cagr=cagr, dd=dd, cdd=cddv, span=span)


# ---------------------------------------------------------------- book plumbing -------------

def book_series(legs, basket, budget=0.03):
    """Same construction as book_spec_fix.book(), but returns the weighted trade series
    itself (book() only returns the summary tuple) -- needed for the block bootstrap.
    w_trade is imported, not reimplemented."""
    w = w_trade(legs, basket, budget)
    st = max(legs[k].index.min() for k in basket)
    en = min(legs[k].index.max() for k in basket)
    parts = []
    for k in basket:
        s = legs[k][(legs[k].index >= st) & (legs[k].index <= en)]
        parts.append(pd.Series(s.values * w[k], index=s.index))
    return pd.concat(parts).sort_index()


def w_trade_scaled(legs, basket, scale, target="btc_bo_kama", budget=0.03):
    """w_trade's inv-vol weights, but target's weight *= scale; the freed (or added) budget
    is re-split among the OTHER legs in proportion to THEIR OWN baseline inv-vol weights
    (their relative mix is unchanged), total budget held fixed."""
    sig = pd.Series({k: legs[k].std() for k in basket})
    w0 = 1.0 / sig
    w0 = w0 / w0.sum() * budget
    w = w0.copy()
    w[target] = w0[target] * scale
    others = [k for k in basket if k != target]
    leftover = budget - w[target]
    base_sum = w0[others].sum()
    for k in others:
        w[k] = w0[k] / base_sum * leftover
    return w


def book_series_scaled(legs, basket, scale, target="btc_bo_kama", budget=0.03):
    w = w_trade_scaled(legs, basket, scale, target, budget)
    st = max(legs[k].index.min() for k in basket)
    en = min(legs[k].index.max() for k in basket)
    parts = []
    for k in basket:
        s = legs[k][(legs[k].index >= st) & (legs[k].index <= en)]
        parts.append(pd.Series(s.values * w[k], index=s.index))
    return pd.concat(parts).sort_index()


def cdd_series(s):
    return cdd(s.values, (s.index[-1] - s.index[0]).days)  # (cagr, dd, cdd)


def block_boot(named_series, base_key, blocks=(1, 3, 6, 12), ndraw=NDRAW, seed=SEED):
    """Paired circular block bootstrap over calendar months (same algorithm as
    book_leave_one_out.main() / book_final_decisions.boot(), generalized to take
    pre-built named series instead of building them from a single legs+basket)."""
    months = sorted(set(named_series[base_key].index.to_period("M")))
    m = len(months)
    G = {k: {p: g.values for p, g in s.groupby(s.index.to_period("M"))} for k, s in named_series.items()}
    rng = np.random.default_rng(seed)
    names = list(named_series)
    print(f"  {'block':<7}" + "".join(f"{k:>26}" for k in names))
    for blk in blocks:
        nb = int(np.ceil(m / blk))
        D = {k: [] for k in names}
        for _ in range(ndraw):
            st_ = rng.integers(0, m, nb)
            order = [months[(s + j) % m] for s in st_ for j in range(blk)][:m]
            for k in names:
                v = np.concatenate([G[k][p] for p in order if p in G[k]])
                D[k].append(cdd(v, 365.25 * m / 12)[2])
        b = np.array(D[base_key])
        row = [f"{np.nanmedian(np.array(D[k])):.2f}(P{np.nanmean(np.array(D[k]) > b)*100:.0f}%)" for k in names]
        print(f"  {f'{blk}mo':<7}" + "".join(f"{r:>26}" for r in row))


def fmt_leg(tag, st):
    if st is None:
        print(f"  {tag:<28} too few trades")
        return
    print(f"  {tag:<28} n={st['n']:>4}  N/yr={st['npy']:>5.1f}  win={st['win']:>5.1f}%  "
          f"medR={st['med']:>+6.2f}  PF={st['pf']:>5.2f}  meanR={st['meanR']:>+6.3f}  "
          f"sd={st['sd']:>5.2f}  legCAGR/DD={st['cdd']:>5.2f}")


# ================================================================== main =====================

def main(smoke=False):
    print("=" * 100)
    print("STEP 0 -- tie-back: book_spec_fix.build('2018-01-01', False), 3-leg & 6-leg CAGR/DD")
    print("=" * 100)
    BASE_LEGS = build("2018-01-01", False)
    c3 = book(BASE_LEGS, OLD)
    c6 = book(BASE_LEGS, NEW)
    print(f"  3-leg (OLD)  CAGR/DD = {c3[2]:.2f}   (spec card says 3.03)")
    print(f"  6-leg (NEW)  CAGR/DD = {c6[2]:.2f}   (spec card says 8.27)")
    tie_ok = abs(c3[2] - 3.03) < 0.05 and abs(c6[2] - 8.27) < 0.05
    if not tie_ok:
        print("  *** TIE-BACK MISMATCH -- stopping per spec card instruction. ***")
        return
    print("  tie-back OK.\n")

    base_leg_df = None  # will hold the make_leg(rr=2.0) trades for the leg-level tie-back
    print("STEP 0b -- tie-back: make_leg(rr=2.0) vs research/portfolio_kama.get_legs()['btc_bo_kama']")
    base_leg_df = make_leg(rr=2.0)
    bstat = leg_stats(base_leg_df)
    fmt_leg("btc_bo_kama (rebuilt, RR2.0)", bstat)
    print("  spec card diagnosed: n=70, win=54.3%, PF=2.31, meanR=+0.602 (8.8yr, ~8/yr)\n")

    rr_grid = [1.5] if smoke else [round(1.5 + 0.1 * i, 1) for i in range(16)]

    print("=" * 100)
    print("E1 -- RR sweep 1.5 -> 3.0, step 0.1 (others held at current: zz_k=2.0, trend_ema=80,")
    print("      bo_window=20, tf=4h). win%/中央値R/PF/meanR/CAGR/DD, + 3-leg & 6-leg book CAGR/DD.")
    print("=" * 100)
    e1_rows = []
    print(f"  {'RR':>4}  {'n':>4}  {'N/yr':>5}  {'win%':>6}  {'medR':>7}  {'PF':>6}  {'meanR':>7}  "
          f"{'legC/DD':>8}  {'3legC/DD':>9}  {'6legC/DD':>9}")
    for rr in rr_grid:
        t = make_leg(rr=rr)
        st = leg_stats(t)
        if st is None:
            print(f"  {rr:>4.1f}  too few trades"); continue
        s = to_series(t)
        L = dict(BASE_LEGS); L["btc_bo_kama"] = s
        b3 = book(L, OLD); b6 = book(L, NEW)
        e1_rows.append(dict(rr=rr, s=s, **st, c3=b3[2], c6=b6[2]))
        print(f"  {rr:>4.1f}  {st['n']:>4}  {st['npy']:>5.1f}  {st['win']:>5.1f}%  {st['med']:>+7.2f}  "
              f"{st['pf']:>6.2f}  {st['meanR']:>+7.3f}  {st['cdd']:>8.2f}  {b3[2]:>9.2f}  {b6[2]:>9.2f}")

    if not smoke and e1_rows:
        below50 = [r["rr"] for r in e1_rows if r["win"] < 50.0]
        first_below = min(below50) if below50 else None
        print(f"\n  勝率が50%を割る最初のRR: {first_below}")
        best = max(e1_rows, key=lambda r: r["c6"])
        print(f"  6レッグ・ブックCAGR/DD最良点: RR={best['rr']} (6leg={best['c6']:.2f}, "
              f"vs RR2.0の6leg={[r for r in e1_rows if r['rr']==2.0][0]['c6']:.2f})")
        cur = [r for r in e1_rows if r["rr"] == 2.0][0]
        print(f"\n  ブロック・ブートストラップ: RR2.0 (現行) vs RR={best['rr']} (最良点) -- 6レッグ・ブック")
        named = {
            "6leg_RR2.0(現行)": book_series(dict(BASE_LEGS, btc_bo_kama=cur["s"]), NEW),
            f"6leg_RR{best['rr']}(最良)": book_series(dict(BASE_LEGS, btc_bo_kama=best["s"]), NEW),
        }
        block_boot(named, "6leg_RR2.0(現行)")
        print(f"\n  同、3レッグ・ブック")
        named3 = {
            "3leg_RR2.0(現行)": book_series(dict(BASE_LEGS, btc_bo_kama=cur["s"]), OLD),
            f"3leg_RR{best['rr']}(最良)": book_series(dict(BASE_LEGS, btc_bo_kama=best["s"]), OLD),
        }
        block_boot(named3, "3leg_RR2.0(現行)")

    print("\n" + "=" * 100)
    print("E2 -- frequency axes (RR held at 2.0), one axis at a time. n=70 の病根を叩けるか。")
    print("=" * 100)
    axes = {
        "zz_k": [1.25, 1.5, 1.75, 2.0, 2.5] if not smoke else [2.0],
        "trend_ema": [0, 50, 80, 120] if not smoke else [80],
        "bo_window": [10, 20, 40, 60] if not smoke else [20],
        "tf": ["1h", "2h", "4h", "6h", "8h"] if not smoke else ["4h"],
    }
    for axis, vals in axes.items():
        print(f"\n  --- axis: {axis} ---")
        print(f"  {'val':>6}  {'n':>4}  {'N/yr':>5}  {'win%':>6}  {'PF':>6}  {'meanR':>7}  "
              f"{'legC/DD':>8}  {'6legC/DD':>9}")
        for v in vals:
            kw = dict(zz_k=2.0, trend_ema=80, bo_window=20, tf="4h", rr=2.0)
            kw[axis] = v
            t = make_leg(**kw)
            st = leg_stats(t)
            if st is None:
                print(f"  {str(v):>6}  too few trades"); continue
            s = to_series(t)
            L = dict(BASE_LEGS); L["btc_bo_kama"] = s
            b6 = book(L, NEW)
            print(f"  {str(v):>6}  {st['n']:>4}  {st['npy']:>5.1f}  {st['win']:>5.1f}%  "
                  f"{st['pf']:>6.2f}  {st['meanR']:>+7.3f}  {st['cdd']:>8.2f}  {b6[2]:>9.2f}")

    print("\n" + "=" * 100)
    print("E3 -- btc_bo_kama の重みを縮小、余りは他5レッグへ現行比率で再配分（総リスク3%固定）")
    print("=" * 100)
    scales = [1.0, 0.75, 0.5, 0.35, 0.25]
    e3_series = {}
    print(f"  {'scale':>6}  {'6legCAGR/DD':>12}  {'6leg maxDD%':>12}")
    for sc in scales:
        s = book_series_scaled(BASE_LEGS, NEW, sc)
        c = cdd_series(s)
        e3_series[f"w={sc}"] = s
        print(f"  {sc:>6.2f}  {c[2]:>12.2f}  {c[1]:>12.2f}")
    print(f"\n  ブロック・ブートストラップ（w=1.0 が基準）")
    block_boot(e3_series, "w=1.0")

    print("\n" + "=" * 100)
    print("E4 -- ショート鏡像（BTC 4H, KAMA(14)下向き, RR2, invert()使用）")
    print("=" * 100)
    d4 = resample(load_mt5_csv(BTC_H1), "4h")
    inv = invert(d4)
    args = SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0, "fwd": 300, "gate_kama": 14})
    ts = run_bo(inv, args)
    if ts is None or len(ts) == 0:
        print("  no entries -- short mirror leg is empty.")
    else:
        ts = ts[["time", "R"]]
        st = leg_stats(ts)
        fmt_leg("btc_bo_kama SHORT (KAMA14下向き)", st)
        s_short = to_series(ts)
        yr = s_short.groupby(s_short.index.year).sum()
        print("  年別 totR: " + " ".join(f"{y}:{v:+.1f}" for y, v in yr.items()))
        L7 = dict(BASE_LEGS); L7["btc_bo_kama_short"] = s_short
        basket7 = NEW + ["btc_bo_kama_short"]
        b7 = book(L7, basket7)
        b6ref = book(BASE_LEGS, NEW)
        print(f"\n  6レッグ(現行) CAGR/DD = {b6ref[2]:.2f}   7レッグ(+ショート) CAGR/DD = {b7[2]:.2f}")
        # correlation with btc15m_S (annual R)
        if "btc15m_S" in BASE_LEGS:
            sS = BASE_LEGS["btc15m_S"]
            yS = sS.groupby(sS.index.year).sum()
            al = pd.concat([yr, yS], axis=1).fillna(0); al.columns = ["short_mirror", "btc15m_S"]
            print(f"  年次R相関 (short_mirror vs btc15m_S): {al.short_mirror.corr(al.btc15m_S):+.2f}  "
                  f"(重複年={len(al)})")
        if not smoke:
            named7 = {"6leg(現行)": book_series(BASE_LEGS, NEW), "7leg(+short)": book_series(L7, basket7)}
            print(f"\n  ブロック・ブートストラップ: 6レッグ vs 7レッグ")
            block_boot(named7, "6leg(現行)")

    print("\n" + "=" * 100)
    print("DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    main(smoke=a.smoke)
