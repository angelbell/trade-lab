"""Sizing btc15m_A down when the daily is below its SMA150 lifts CAGR/DD 3.59 -> 4.25 and cuts maxDD
10.9% -> 8.5%. But shrinking any bet cuts drawdown -- that is arithmetic, not edge. Four times today
a "gain" turned out to be nothing but a smaller (or bigger) bet.

So settle it the only way that cannot be gamed: DE-LEVER EVERY ARM TO THE SAME maxDD and compare CAGR.
If the soft-size arm still earns more at 10.9% drawdown than the flat leg does, the rule is real.
Plus: sweep finely enough to see whether 0.75 is a HILL or the edge of the grid, run the reversed
dummy (size UP when the daily is down -- it must come out worst), and the block bootstrap.
Run: .venv/bin/python experiments/A_soft_equal_dd.py
"""
import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE

ROOT = "/home/angelbell/dev/auto-trade"
RNG = np.random.default_rng(20260714)
NDRAW = 2000
CFG = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3, "fill_win": 200,
       "rr": 4.5, "fwd": 500}


def cd(R, f):
    eq = np.cumprod(1 + f * R)
    if np.any(eq <= 0):
        return -99, 100.0
    pk = np.maximum.accumulate(eq)
    return eq, ((pk - eq) / pk).max() * 100


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**CFG))
    dly = d15["close"].resample("1D").last().dropna()
    up = (dly > dly.rolling(150).mean()).shift(1).reindex(d15.index, method="ffill")
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    ei = d15.index.get_indexer(t["time"])
    ab = t["e_px"].values > pdh[ei]
    R = (t["R"].values - 15.0 / t["risk"].values)[ab]
    ti = pd.DatetimeIndex(t["time"])[ab]
    U = up.values[ei][ab] == True
    yrs = (ti[-1] - ti[0]).days / 365.25
    print(f"btc15m_A  n={len(R)}   うち 日足SMA150の【上】= {U.sum()}本 / 【下】= {(~U).sum()}本 "
          f"（下の年内訳: " + " ".join(f"{y}:{v}" for y, v in
                                    pd.Series(ti[~U]).dt.year.value_counts().sort_index().items()) + "）\n")

    # 基準: フラット（日足を見ない）を賭け率1%で回したときの maxDD
    _, D0 = cd(R, 0.01)
    e0, _ = cd(R, 0.01)
    C0 = (e0[-1] ** (1 / yrs) - 1) * 100
    print(f"基準（フラット・賭け率1%）: CAGR {C0:+.1f}%  maxDD {D0:.2f}%  CAGR/DD {C0/D0:.2f}")
    print(f"→ **全ての腕を maxDD {D0:.2f}% にそろえて CAGR で比べる**（サイズを下げれば DD が下がるのは")
    print("   当たり前なので、それを完全に打ち消す。今日4件をこれで撤回した）\n")
    print(f"  {'日足↓のときのサイズ':<26}{'賭け率':>8}{'CAGR':>9}{'maxDD':>8}{'現行比 CAGR':>13}")

    def eq_dd(w):
        lo, hi = 0.0005, 0.20
        for _ in range(70):
            mid = (lo + hi) / 2
            if cd(R * w, mid)[1] > D0:
                hi = mid
            else:
                lo = mid
        eq, dd = cd(R * w, lo)
        return lo, (eq[-1] ** (1 / yrs) - 1) * 100, dd

    arms = [("1.00（現行・日足を見ない）", 1.00), ("0.90", 0.90), ("0.80", 0.80), ("0.75", 0.75),
            ("0.60", 0.60), ("0.50", 0.50), ("0.35", 0.35), ("0.25", 0.25), ("0.00（建てない）", 0.0),
            ("1.50（逆向きダミー＝下で増やす）", 1.50)]
    base_c = None
    for tag, f in arms:
        w = np.where(U, 1.0, f)
        bet, c, d = eq_dd(w)
        if f == 1.0:
            base_c = c
        print(f"  {tag:<26}{100*bet:>7.2f}%{c:>8.1f}%{d:>7.2f}%{c-base_c:>+12.1f}pt"
              + ("  ★" if base_c is not None and c > base_c + 1 else ""))

    print("\n\n巡回ブロック・ブートストラップ（同じ賭け率1%で、CAGR/DD を直接対決。2000回）")
    print("  本物なら P はブロックを伸ばすほど上がる\n")
    m, months = len(R), pd.Series(ti).dt.to_period("M")
    mo = sorted(set(months)); M = len(mo)
    idx_by = {x: np.where(months == x)[0] for x in mo}

    def cdd_of(w, order):
        v = np.concatenate([R[idx_by[mo[i]]] * w[idx_by[mo[i]]] for i in order
                            if len(idx_by[mo[i]])])
        if len(v) < 20:
            return None
        eq = np.cumprod(1 + 0.01 * v); pk = np.maximum.accumulate(eq)
        dd = ((pk - eq) / pk).max()
        return (eq[-1] ** (1 / yrs) - 1) / max(dd, 1e-9)

    wb = np.ones(m)
    print(f"  {'日足↓のサイズ':<18}{'CAGR/DD':>9}{'1か月':>8}{'3か月':>8}{'6か月':>8}{'12か月':>8}")
    for f in (0.9, 0.75, 0.5, 0.25):
        wa = np.where(U, 1.0, f)
        eq, dd = cd(R * wa, 0.01)
        r = (eq[-1] ** (1 / yrs) - 1) * 100 / dd
        row = f"  {f'×{f}':<18}{r:>9.2f}"
        for Lb in (1, 3, 6, 12):
            nb = int(np.ceil(M / Lb)); wins = 0
            for _ in range(NDRAW):
                s0 = RNG.integers(0, M, nb)
                order = np.concatenate([(np.arange(s, s + Lb) % M) for s in s0])[:M]
                a, b = cdd_of(wa, order), cdd_of(wb, order)
                if a is None or b is None:
                    continue
                wins += a > b
            row += f"{100*wins/NDRAW:>7.0f}%"
        print(row)


if __name__ == "__main__":
    main()
