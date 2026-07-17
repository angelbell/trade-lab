"""btc15m_L is PF 2.01 GROSS and PF 1.76 NET. Where does the 0.25R go, and can any of it be taken back?

Cost enters as cost_R = $15 / stop_distance. So the SAME $15 is a huge tax on a trade whose stop is
$300 (0.050R) and a rounding error on one whose stop is $3,000 (0.005R). The obvious "fix" is to skip
the small-stop trades. But that is a SELECTION rule, and selection rules are luck-sorters (checklist
7): the only honest way to report one is with the N it costs and the totR/yr it leaves behind, not
the PF it buys. A PF that rises while totR/yr falls has bought nothing.
Run: .venv/bin/python scratchpad/btc15mL_cost_anatomy.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE

ROOT = "/home/angelbell/dev/auto-trade"


def pf(x):
    return x[x > 0].sum() / abs(x[x <= 0].sum())


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3, "rr": 4.5, "fill_win": 200}))
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    w = np.where(t["e_px"].values > pdh[d15.index.get_indexer(t["time"])], 1.0, 0.5)
    R = t["R"].values * w                       # gross, PDH soft-size applied
    risk = t["risk"].values / w                 # $ per 1R, after the soft-size
    cR = 15.0 / risk                            # the $15 round trip, in R
    idx = pd.DatetimeIndex(t["time"])
    yrs = (idx[-1] - idx[0]).days / 365.25
    N = R - cR

    print(f"btc15m_L   n={len(R)}   {len(R)/yrs:.0f}本/年   PF(素)={pf(R):.2f}   PF(実コスト)={pf(N):.2f}")
    print(f"  コスト(R単位): 中央値 {np.median(cR):.3f}R  平均 {cR.mean():.3f}R  "
          f"25/75%点 {np.percentile(cR,25):.3f} / {np.percentile(cR,75):.3f}R")
    print(f"  損切り幅($):  中央値 {np.median(risk):,.0f}  10%点 {np.percentile(risk,10):,.0f}  "
          f"90%点 {np.percentile(risk,90):,.0f}")
    print(f"  年間の総R: 素 {R.sum()/yrs:+.1f}R/年  →  実コスト {N.sum()/yrs:+.1f}R/年  "
          f"（コストが年 {(R.sum()-N.sum())/yrs:.1f}R を食っている）\n")

    print("  損切り幅の五分位ごと（＝コストの重さ順）")
    print(f"    {'帯':<10}{'n':>6}{'本/年':>7}{'損切り中央値':>13}{'コスト':>10}"
          f"{'PF(素)':>9}{'PF(実)':>9}{'meanR(実)':>11}{'totR/年':>10}")
    q = pd.qcut(risk, 5, labels=False)
    for i in range(5):
        m = q == i
        print(f"    Q{i+1}{'':<8}{m.sum():>6}{m.sum()/yrs:>7.0f}{np.median(risk[m]):>12,.0f}$"
              f"{np.mean(cR[m]):>9.3f}R{pf(R[m]):>9.2f}{pf(N[m]):>9.2f}{N[m].mean():>+11.3f}"
              f"{N[m].sum()/yrs:>+10.1f}")

    print("\n  「損切りが小さいトレード（＝コストが重いトレード）を捨てる」と何が起きるか")
    print("  ★ PF だけ見てはいけない。**N と totR/年** が答え。")
    print(f"    {'残す条件':<22}{'n':>6}{'本/年':>7}{'PF(実)':>9}{'meanR(実)':>11}"
          f"{'totR/年':>10}{'素のtotR/年':>12}")
    for thr in (0, 300, 500, 750, 1000, 1500, 2000):
        m = risk >= thr
        if m.sum() < 30:
            continue
        lab = "全部（現行）" if thr == 0 else f"損切り >= ${thr:,}"
        print(f"    {lab:<22}{m.sum():>6}{m.sum()/yrs:>7.0f}{pf(N[m]):>9.2f}{N[m].mean():>+11.3f}"
              f"{N[m].sum()/yrs:>+10.1f}{R[m].sum()/yrs:>+12.1f}")

    print("\n  比較: コストがゼロだったら（＝執行を完璧にしたときの上限）")
    print(f"    全部・コスト0             {len(R):>6}{len(R)/yrs:>7.0f}{pf(R):>9.2f}{R.mean():>+11.3f}"
          f"{R.sum()/yrs:>+10.1f}")
    print(f"\n  → コストを完全に消しても PF は {pf(N):.2f} → {pf(R):.2f}。**それが執行レバーの天井。**")


if __name__ == "__main__":
    main()
