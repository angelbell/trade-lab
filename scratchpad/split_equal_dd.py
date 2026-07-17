"""Every internal-allocation arm that "improved" the book today also quietly raised the bet. The only
comparison that cannot be gamed by leverage: DE-LEVER EACH ARM UNTIL ITS BOOK maxDD EQUALS THE
BASELINE'S, then compare CAGR. Same risk, more money = a real improvement. (This is the method that
settled btc_bo_kama: "diversifier or risk dial?")

Arms, all inside the BTC-15m-long family (A = break ABOVE the previous day's high, B = inside it):
  current   A:B exposure 1.00 : 0.50   <- today's book
  4:1       A:B          1.00 : 0.25
  A only    A:B          1.00 : 0.00
  inv-vol   A:B          1/sigma       <- what the book's own weighting rule would do
  reversed  A:B          0.50 : 1.00   <- the deliberate dummy; must come out WORST
The whole book's risk budget is scaled per arm to hit maxDD = 7.74%. Then CAGR is the answer.
Run: .venv/bin/python scratchpad/split_equal_dd.py
"""
import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE
from wide_stop_stress import raw_legs, SIX

ROOT = "/home/angelbell/dev/auto-trade"
RNG = np.random.default_rng(20260714)


def main():
    L = raw_legs()
    legs0 = {k: v[0] for k, v in L.items()}
    sig = pd.Series({k: legs0[k].std() for k in SIX})
    w0 = (1 / sig) / (1 / sig).sum() * 0.03
    WL = w0["btc15m_L"]
    st = max(legs0[k].index.min() for k in SIX)
    en = min(legs0[k].index.max() for k in SIX)
    cut = lambda s: s[(s.index >= st) & (s.index <= en)]
    others = [(cut(legs0[k]), w0[k]) for k in SIX if k != "btc15m_L"]

    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3, "rr": 4.5, "fill_win": 200}))
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    ab = t["e_px"].values > pdh[d15.index.get_indexer(t["time"])]
    Rraw = t["R"].values - 15.0 / t["risk"].values
    idx = pd.DatetimeIndex(t["time"])
    A = cut(pd.Series(Rraw[ab], index=idx[ab]))
    B = cut(pd.Series(Rraw[~ab], index=idx[~ab]))

    def stat(fa, fb, scale):
        parts = [(x, w * scale) for x, w in others] + [(A, WL * fa * scale), (B, WL * fb * scale)]
        s = pd.concat([pd.Series(x.values * w, index=x.index) for x, w in parts]).sort_index()
        eq = np.cumprod(1 + s.values); pk = np.maximum.accumulate(eq)
        dd = ((pk - eq) / pk).max() * 100
        cagr = (eq[-1] ** (365.25 / max((s.index[-1] - s.index[0]).days, 1)) - 1) * 100
        return cagr, dd

    c0, D0 = stat(1.0, 0.5, 1.0)                        # 現行
    print(f"現行のブック: CAGR {c0:+.1f}%  maxDD {D0:.2f}%  CAGR/DD {c0/D0:.2f}")
    print(f"→ **全ての腕を maxDD {D0:.2f}% にそろえて、CAGR だけで比べる**（レバレッジを完全に排除）\n")
    ia = (1 / A.std()) / (1 / A.std() + 1 / B.std())
    arms = {
        "現行（A フル / B 半分）": (1.0, 0.5),
        "A に厚く（1.00 / 0.25）": (1.0, 0.25),
        "A だけ（1.00 / 0.00）": (1.0, 0.0),
        "ブックの inv-vol（1/σ）": (ia / max(ia, 1 - ia), (1 - ia) / max(ia, 1 - ia)),
        "B に厚く（0.50 / 1.00）＝逆向きダミー": (0.5, 1.0),
        "A をさらに厚く（1.00 / 0.10）": (1.0, 0.10),
    }
    print(f"  {'内部の配分':<34}{'総リスク':>9}{'CAGR':>9}{'maxDD':>8}{'CAGR/DD':>10}"
          f"{'現行比 CAGR':>13}")
    for tag, (fa, fb) in arms.items():
        lo, hi = 0.05, 4.0                              # DD が D0 になる倍率を二分探索
        for _ in range(60):
            mid = (lo + hi) / 2
            if stat(fa, fb, mid)[1] > D0:
                hi = mid
            else:
                lo = mid
        c, d = stat(fa, fb, lo)
        tot = 0.03 * lo
        print(f"  {tag:<34}{100*tot:>8.2f}%{c:>8.1f}%{d:>7.2f}%{c/max(d,1e-9):>10.2f}"
              f"{c-c0:>+12.1f}pt" + ("  ★" if c > c0 + 1 else ""))

    print("\n\n同じ物差しで、A の内部の閾値（B をどこまで絞るか）を掃引")
    print(f"  {'B の倍率':>9}{'総リスク':>9}{'CAGR':>9}{'maxDD':>8}{'現行比':>10}")
    for fb in (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0):
        lo, hi = 0.05, 4.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if stat(1.0, fb, mid)[1] > D0:
                hi = mid
            else:
                lo = mid
        c, d = stat(1.0, fb, lo)
        print(f"  {fb:>9.2f}{100*0.03*lo:>8.2f}%{c:>8.1f}%{d:>7.2f}%{c-c0:>+9.1f}pt"
              + ("  ← 現行" if fb == 0.5 else ""))


if __name__ == "__main__":
    main()
