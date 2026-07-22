"""Can the book's PROVEN entries be repackaged as a HIGH-WIN-RATE leg?

The user wants win% >= 80% at RR 0.5 (or >= 70% at RR 1.0). The cheapest possible test of whether
that is even reachable: take the entries we already know have edge, keep their structural stop (the
stop is what makes the entry work -- it sits at the invalidation level), and just move the TARGET in.

The arithmetic that decides it, stated before running:
    random-walk win% = breakeven win% = 1/(1+RR)      (they are the same number)
  RR 0.5 -> 66.7%   RR 0.7 -> 58.8%   RR 1.0 -> 50.0%
So the ONLY thing that matters is the gap between the observed win% and 1/(1+RR). A leg that hits
80% at RR 0.5 clears the bar by 13.3 points. A leg that hits 68% clears it by ~1 point and is noise.

Report, per leg per RR: n, win%, the breakeven win%, the GAP, PF, meanR, and the per-year win% (so a
leg that only clears the bar in bull years is visible immediately -- the user's own objection).
Cost is charged in price units (BTC $15, gold $0.30) and converted to R via the structural stop, so
shrinking the target does NOT shrink the cost -- which is exactly the tax a near target has to pay.
Run: .venv/bin/python experiments/near_target_repack.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from src.data_loader import load_mt5_csv, GOLD_H1_START
from breakout_wave import run, resample
from research.regime_gate_lab import CFG
from research.portfolio_kama import kama_gate_btc
from radar_gate_race import BASE
from short_mirror_15m import invert

ROOT = "/home/angelbell/dev/auto-trade"
RRS = [0.1, 0.2, 0.3, 0.4, 0.5]


# Charge cost ONCE, at the LIVE-measured level (2026-07-02), in price units. The canonical leg
# configs disagree: CFG carries cost=0.001 (= 0.1% of price = $2/oz on gold, $50 on 4H BTC -- 6-13x
# the real cost), BASE carries cost=0. Left as-is that would double-charge two legs and under-charge
# three. Here every leg runs GROSS (cost=0 inside run()) and the real round-trip is subtracted
# afterwards in R units: cost_R = cost_price / stop_distance.
COST = {"gold_bo (1H)": 0.30, "btc_bo_kama (4H)": 15.0, "gold15m": 0.30,
        "btc15m_L": 15.0, "btc15m_S": 15.0}


def legs_at(rr, net=True):
    """The 5 breakout legs, entry + structural stop UNCHANGED, target moved to `rr`.
    net=False -> gross (no cost at all), to separate "no edge" from "edge, killed by cost"."""
    out = {}
    with contextlib.redirect_stderr(io.StringIO()):
        g1 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv").loc[GOLD_H1_START:], "1h")
        t = run(g1, SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": rr, "fwd": 500,
                                       "cost": 0.0, "daily_sma": 150, "daily_slope_k": 10}))
        out["gold_bo (1H)"] = (t["R"].values, t["risk"].values, pd.DatetimeIndex(t["time"]))

        b4 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h")
        t = kama_gate_btc(run(b4, SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": rr,
                                                     "cost": 0.0, "fwd": 300})))
        out["btc_bo_kama (4H)"] = (t["R"].values, t["risk"].values, pd.DatetimeIndex(t["time"]))

        g15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g15, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10, "ext_cap": 8.0,
                                        "pullback_frac": 0.25, "rr": rr, "fill_win": 200}))
        out["gold15m"] = (t["R"].values, t["risk"].values, pd.DatetimeIndex(t["time"]))

        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3, "rr": rr, "fill_win": 200}))
        pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
        w = np.where(t["e_px"].values > pdh[d15.index.get_indexer(t["time"])], 1.0, 0.5)
        out["btc15m_L"] = (t["R"].values * w, t["risk"].values / w, pd.DatetimeIndex(t["time"]))

        inv = invert(d15); C = 2 * d15["high"].max()
        t = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3, "rr": rr,
                                        "fill_win": 200}))
        pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
        m = (C - t["e_px"].values) < pdl[d15.index.get_indexer(t["time"])]
        out["btc15m_S"] = (t["R"].values[m], t["risk"].values[m], pd.DatetimeIndex(t["time"])[m])

    return {k: pd.Series(R - (COST[k] / risk if net else 0.0), index=idx)
            for k, (R, risk, idx) in out.items()}


def main():
    print("入口と損切りは現行のまま。**利確だけ**を近くに寄せる。")
    print("合格線は「勝率 > 損益分岐 1/(1+RR)」。この2つの差（GAP）だけが意味を持つ。\n")
    print("コストは「エッジ判定の後段」で分離する（台帳の規律）。素(gross)で勝てないなら、そもそもエッジが無い。")
    print("素で勝てるのにコストで死ぬなら、それは別の病気＝執行の問題。\n")
    print("⚠️ 損益分岐は 1/(1+RR) では出せない。押し目指値の3レッグは約定値が e より下なので、")
    print("   『RR 0.5』と指定しても 勝ち+0.8R / 負け−0.7R ＝ **実効RR 1.14** になる（＝指値の効能そのもの）。")
    print("   なので実効RR = (勝ちの平均R)/(負けの平均|R|) を実測し、そこから損益分岐を出す。\n")
    print(f"  {'leg':<18}{'指定RR':>7}{'勝ちの平均':>10}{'負けの平均':>10}{'実効RR':>8}{'損益分岐':>9}"
          f"{'n':>6}{'本/年':>6}{'勝率':>7}{'GAP':>8}{'PF(素)':>8}{'PF(実)':>8}{'meanR':>9}")
    hist = {}
    for rr in RRS:
        G, L = legs_at(rr, net=False), legs_at(rr, net=True)
        for k in L:
            g, s = G[k], L[k]
            yrs = (s.index[-1] - s.index[0]).days / 365.25
            win = 100 * (s > 0).mean()
            W, Ls = g[g > 0].mean(), abs(g[g <= 0].mean())      # 素の payoff で実効RRを出す
            eff = W / Ls
            be = 100.0 / (1.0 + eff)                             # ドリフト無しの到達確率＝損益分岐
            pfg = g[g > 0].sum() / abs(g[g <= 0].sum())
            pf = s[s > 0].sum() / abs(s[s <= 0].sum())
            mark = "  ★" if pf >= 2.0 else ""
            print(f"  {k:<18}{rr:>7.1f}{W:>+10.2f}{-Ls:>+10.2f}{eff:>8.2f}{be:>8.1f}%"
                  f"{len(s):>6}{len(s)/yrs:>6.0f}{win:>6.1f}%{win-be:>+8.1f}"
                  f"{pfg:>8.2f}{pf:>8.2f}{s.mean():>+9.3f}{mark}")
            hist[(k, rr)] = s
        print()

    print("\n年別の勝率（RR 0.5 と RR 1.0 のみ。上げ相場でしか勝てない脚はここで露見する）")
    for rr in (0.5, 1.0):
        be = 100.0 / (1.0 + rr)
        print(f"\n  RR {rr}  （損益分岐 {be:.1f}%。この数字を下回った年は赤字）")
        ks = [k for (k, r) in hist if r == rr]
        yrs = sorted({y for k in ks for y in hist[(k, rr)].index.year})
        print("    " + " " * 18 + "".join(f"{y:>7}" for y in yrs))
        for k in ks:
            s = hist[(k, rr)]
            row = ""
            for y in yrs:
                g = s[s.index.year == y]
                row += f"{100*(g>0).mean():>6.0f}%" if len(g) >= 5 else "     ·"
            print(f"    {k:<18}{row}")


if __name__ == "__main__":
    main()
