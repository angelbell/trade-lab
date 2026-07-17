"""The recipe works on exactly two instruments (gold, BTC). Both are secular-uptrend, trend-forming
assets. The repo turns out to hold Vantage H1 for four more that were never tried:
NAS100 (2016-), GER40 (2015-), XAGUSD (2015-, plus M15 from 2018), USOUSD/oil (2015-).

Run gold_bo's exact recipe on each -- ZigZag(2xATR) Pattern-B breakout, 1H, confirmed-close entry,
structural stop, fixed far target -- with NO gate first (all-signals base; filters concentrate an
edge, they never create one), then with each gate.

Two falsifiers stated before running:
  - An index long-only in a secular bull is BETA, not edge (checklist 10). So print buy-and-hold's
    CAGR next to the leg's, and demand the leg beat a RANDOM-ENTRY null with the same bracket.
  - Silver sits in the metals cluster with gold (structural law 6: edge and independence trade off).
    A silver leg that works is probably gold_bo wearing a hat -- the annual-R correlation decides.
Cost is a price ratio (conservative): index 0.02%, oil 0.05%, silver 0.1% (gold's canon).
Run: .venv/bin/python scratchpad/new_instruments.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from research.regime_gate_lab import CFG

ROOT = "/home/angelbell/dev/auto-trade"
RNG = np.random.default_rng(20260714)
INST = {
    "NAS100":   ("vantage_nas100.r_h1.csv", 0.0002),
    "GER40":    ("vantage_ger40.r_h1.csv",  0.0002),
    "XAGUSD 銀": ("vantage_xagusd_h1.csv",   0.0010),
    "USOUSD 原油": ("vantage_usousd_h1.csv", 0.0005),
    "XAUUSD 金(対照)": ("vantage_xauusd_h1.csv", 0.0010),
}
GATES = {"ゲート無し(素)": {}, "日足SMA150↑": {"daily_sma": 150, "daily_slope_k": 10},
         "日足KAMA(14)↑": {"gate_kama": 14}}


def bh(d):
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    return ((d["close"].iloc[-1] / d["close"].iloc[0]) ** (1 / yrs) - 1) * 100


def stat(t, yrs):
    R = t["R"].values
    if len(R) < 20:
        return None
    pf = R[R > 0].sum() / abs(R[R <= 0].sum()) if (R <= 0).any() else np.inf
    eq = np.cumprod(1 + 0.01 * R); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100
    h = len(R) // 2
    return dict(n=len(R), per_year=len(R) / yrs, win=100 * (R > 0).mean(), pf=pf,
                meanR=R.mean(), IS=R[:h].mean(), OOS=R[h:].mean(),
                cdd=cagr / max(dd, 1e-9), cagr=cagr, dd=dd)


def main():
    print("gold_bo のレシピ（ZigZag 2×ATR · Pattern-B · 1時間足 · 確定足ブレイク · 構造ストップ · RR3）")
    print("を、当てたことのない銘柄へ。**まずゲート無しの全シグナル・ベース**（フィルタはエッジを濃縮する")
    print("だけで、作りはしない）。\n")
    per_year_R = {}
    for name, (csv, cost) in INST.items():
        with contextlib.redirect_stderr(io.StringIO()):
            d = resample(load_mt5_csv(f"{ROOT}/data/{csv}"), "1h")
        if name.startswith("XAUUSD"):
            d = d.loc["2018-01-01":]
        yrs = (d.index[-1] - d.index[0]).days / 365.25
        print(f"  === {name}   {d.index[0].date()} → {d.index[-1].date()} ({yrs:.1f}年)   "
              f"買い持ちの CAGR {bh(d):+.1f}%   コスト {100*cost:.2f}%")
        print(f"      {'':<16}{'n':>5}{'本/年':>7}{'勝率':>7}{'PF':>7}{'meanR':>9}"
              f"{'IS':>9}{'OOS':>9}{'CAGR':>8}{'maxDD':>8}{'CAGR/DD':>9}")
        for gname, g in GATES.items():
            t = run(d, SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                          "cost": cost, **g}))
            s = stat(t, yrs)
            if s is None:
                print(f"      {gname:<16}n={len(t)} 少なすぎ"); continue
            mark = "  ★" if s["pf"] > 1.4 and s["OOS"] > 0 else ""
            print(f"      {gname:<16}{s['n']:>5}{s['per_year']:>7.0f}{s['win']:>6.1f}%{s['pf']:>7.2f}"
                  f"{s['meanR']:>+9.3f}{s['IS']:>+9.3f}{s['OOS']:>+9.3f}{s['cagr']:>7.1f}%"
                  f"{s['dd']:>7.1f}%{s['cdd']:>9.2f}{mark}")
            if gname == "日足SMA150↑":
                r = pd.Series(t["R"].values, index=pd.DatetimeIndex(t["time"]))
                per_year_R[name] = r.groupby(r.index.year).sum()
        # ベータ帰無: 同じ本数・同じブラケットでランダムに買う
        t = run(d, SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                      "cost": cost, "daily_sma": 150, "daily_slope_k": 10}))
        if len(t) >= 20:
            risk = np.median(t["risk"].values)
            c, h, l = d["close"].values, d["high"].values, d["low"].values
            nulls = []
            for _ in range(300):
                idx = RNG.choice(len(c) - 501, len(t), replace=False)
                Rs = []
                for i in idx:
                    e = c[i]; sl = e - risk; tp = e + 3.0 * risk
                    r = 0.0
                    for j in range(i + 1, i + 501):
                        if l[j] <= sl: r = -1.0; break
                        if h[j] >= tp: r = 3.0; break
                    else:
                        r = (c[i + 500] - e) / risk
                    Rs.append(r - cost * e / risk)
                Rs = np.array(Rs)
                nulls.append(Rs[Rs > 0].sum() / max(abs(Rs[Rs <= 0].sum()), 1e-9))
            print(f"      → ランダムに買う帰無（同数・同ブラケット・300回）: "
                  f"PF 中央値 {np.median(nulls):.2f}  95%点 {np.percentile(nulls,95):.2f}"
                  f"   【日足SMA150↑ の PF がこれを超えて初めてエッジ】")
        print()

    if len(per_year_R) >= 2:
        print("\n  年別Rの相関（＝既存ブックと冗長でないか。金属クラスタは金と冗長になるはず）")
        M = pd.DataFrame(per_year_R).fillna(0.0)
        print(M.corr().round(2).to_string())


if __name__ == "__main__":
    main()
