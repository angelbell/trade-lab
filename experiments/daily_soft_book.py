"""The daily soft-size rule (size x0.75 when BTC's daily close is below its 150-day SMA) survives the
equal-drawdown test on btc15m_A standalone: +9.3 CAGR points at the same 10.92% maxDD, a clean hill
(both the skip end and the reversed dummy fail), block bootstrap rising.

Structural law 10 says a leg gain is not a book gain. So:
  1. does the same rule help the FULL btc15m_L (A + the PDH-soft half), which is what the book runs?
  2. does it help the 6-leg BOOK -- judged at EQUAL maxDD, because de-levering is how four of today's
     "improvements" turned out to be nothing?
  3. does it transfer to the other legs (gold15m, btc15m_S, gold_bo)? A rule that only helps one leg
     has no mechanism -- that is how the stop/price filter died this morning.
The daily state is BTC's own (close vs its 150-day SMA); for the gold legs use GOLD's daily SMA150,
which is already their entry gate -- so for them the test is "size down when the gate is OFF" (they
currently skip). Report the reversed dummy everywhere.
Run: .venv/bin/python experiments/daily_soft_book.py
"""
import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE
from wide_stop_stress import raw_legs, SIX

ROOT = "/home/angelbell/dev/auto-trade"
RNG = np.random.default_rng(20260714)
NDRAW = 2000


def main():
    L = raw_legs()
    legs0 = {k: v[0] for k, v in L.items()}
    sig = pd.Series({k: legs0[k].std() for k in SIX})
    w0 = (1 / sig) / (1 / sig).sum() * 0.03
    st = max(legs0[k].index.min() for k in SIX)
    en = min(legs0[k].index.max() for k in SIX)
    cut = lambda s: s[(s.index >= st) & (s.index <= en)]

    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3, "rr": 4.5, "fill_win": 200}))
    dly = d15["close"].resample("1D").last().dropna()
    upB = (dly > dly.rolling(150).mean()).shift(1).reindex(d15.index, method="ffill")
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    ei = d15.index.get_indexer(t["time"])
    wp = np.where(t["e_px"].values > pdh[ei], 1.0, 0.5)          # 既存の PDH サイズ
    Rn = t["R"].values - 15.0 / t["risk"].values                  # コスト込み・サイズ前
    ti = pd.DatetimeIndex(t["time"])
    U = upB.values[ei] == True                                   # 日足 SMA150 の上か

    def book_curve(fdaily, scale=1.0):
        """btc15m_L のサイズ = PDH倍率 × (日足が下なら fdaily)。他レッグは現行のまま。"""
        w = wp * np.where(U, 1.0, fdaily)
        bl = cut(pd.Series(Rn * w, index=ti))
        parts = [(cut(legs0[k]), w0[k]) for k in SIX if k != "btc15m_L"] + [(bl, w0["btc15m_L"])]
        s = pd.concat([pd.Series(x.values * ww * scale, index=x.index) for x, ww in parts]).sort_index()
        eq = np.cumprod(1 + s.values); pk = np.maximum.accumulate(eq)
        dd = ((pk - eq) / pk).max() * 100
        cagr = (eq[-1] ** (365.25 / max((s.index[-1] - s.index[0]).days, 1)) - 1) * 100
        return cagr, dd, s

    C0, D0, _ = book_curve(1.0)
    print(f"現行ブック: CAGR {C0:+.1f}%  maxDD {D0:.2f}%  CAGR/DD {C0/D0:.2f}")
    print(f"→ **maxDD {D0:.2f}% にそろえて CAGR で比べる**（レバレッジを完全排除）\n")
    print(f"  {'btc15m_L: 日足↓のサイズ':<30}{'総リスク':>9}{'CAGR':>9}{'maxDD':>8}"
          f"{'CAGR/DD':>10}{'現行比 CAGR':>13}")

    def eq_dd(f):
        lo, hi = 0.2, 3.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if book_curve(f, mid)[1] > D0:
                hi = mid
            else:
                lo = mid
        c, d, _ = book_curve(f, lo)
        return lo, c, d

    base = None
    for f in (1.0, 0.9, 0.8, 0.75, 0.6, 0.5, 0.25, 0.0, 1.5):
        sc, c, d = eq_dd(f)
        if f == 1.0:
            base = c
        tag = ("1.0（現行・日足を見ない）" if f == 1.0 else "0.0（日足↓では建てない）" if f == 0
               else "1.5（逆向きダミー）" if f == 1.5 else f"{f}")
        print(f"  {tag:<30}{3*sc:>8.2f}%{c:>8.1f}%{d:>7.2f}%{c/max(d,1e-9):>10.2f}"
              f"{c-base:>+12.1f}pt" + ("  ★" if c > base + 1 else ""))

    print("\n\n巡回ブロック・ブートストラップ（ブックの CAGR/DD で直接対決・賭け率そのまま）")
    _, _, b = book_curve(1.0)
    bm = b.index.to_period("M"); mo = sorted(set(bm)); M = len(mo)
    def cdd(v):
        eq = np.cumprod(1 + v); pk = np.maximum.accumulate(eq)
        return (eq[-1] ** (12.0 / M) - 1) / max(((pk - eq) / pk).max(), 1e-9)
    b_by = {m: b[bm == m].values for m in mo}
    print(f"  {'日足↓のサイズ':<18}{'ブックCAGR/DD':>13}{'1か月':>8}{'3か月':>8}{'6か月':>8}{'12か月':>8}")
    for f in (0.9, 0.75, 0.5):
        c, d, a = book_curve(f)
        am = a.index.to_period("M"); a_by = {m: a[am == m].values for m in mo}
        row = f"  {f'×{f}':<18}{c/d:>13.2f}"
        for Lb in (1, 3, 6, 12):
            nb = int(np.ceil(M / Lb)); wins = 0
            for _ in range(NDRAW):
                s0 = RNG.integers(0, M, nb)
                order = np.concatenate([(np.arange(s, s + Lb) % M) for s in s0])[:M]
                bb = np.concatenate([b_by[mo[i]] for i in order if len(b_by[mo[i]])])
                aa = np.concatenate([a_by[mo[i]] for i in order if len(a_by[mo[i]])])
                if len(bb) < 20 or len(aa) < 20:
                    continue
                wins += cdd(aa) > cdd(bb)
            row += f"{100*wins/NDRAW:>7.0f}%"
        print(row)


if __name__ == "__main__":
    main()
