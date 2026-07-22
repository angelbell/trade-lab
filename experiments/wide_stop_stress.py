"""The wide-stop filter (skip breakouts whose stop > X% of price) looks real on btc15m_L:
n 759->584, totR/yr +40.4 -> +41.2 (it goes UP), PF 1.76 -> 1.97, block-bootstrap P RISES 75->94%.

Three ways it could still be fake, tested here:
  S1  It is a disguised YEAR filter. 2021's median stop is 2.09% of price, so a "<= 2.0%" rule
      deletes half of 2021 -- a chop year. If the gain is 2021, the rule is a backward-looking
      regime filter wearing a volatility costume. Test: per-year n / totR before vs after, and the
      totR gain decomposed by year. A real filter improves MOST years; a year filter improves one.
  S2  It does not survive the BOOK (structural law 10 -- a leg gain that deletes the trades carrying
      the book's decorrelation is a loss). Test: the 6-leg book on the deployed spec, trade-resolution
      DD, inv-vol on trade-R sigma, 3% budget.
  S3  It is in-sample only. Test: IS/OOS halves, and the threshold plateau.
Also: does the same rule transfer to the other breakout legs (gold15m, btc15m_S, gold_bo)?
Run: .venv/bin/python experiments/wide_stop_stress.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from src.data_loader import load_mt5_csv, GOLD_H1_START
from breakout_wave import run, resample
from ema_pullback import run as run_pb
from research.regime_gate_lab import CFG
from research.portfolio_kama import kama_gate_btc, cycle_gate_pull, PB
from radar_gate_race import BASE
from short_mirror_15m import invert
from book_spec_fix import book

ROOT = "/home/angelbell/dev/auto-trade"
SIX = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]


def pf(x):
    return x[x > 0].sum() / abs(x[x <= 0].sum()) if (x <= 0).any() else np.inf


def raw_legs():
    """Every leg, plus the stop/price % of each trade (the filter variable)."""
    out = {}
    with contextlib.redirect_stderr(io.StringIO()):
        g1 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv").loc[GOLD_H1_START:], "1h")
        t = run(g1, SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                       "daily_sma": 150, "daily_slope_k": 10}))
        out["gold_bo"] = (pd.Series(t["R"].values, index=pd.DatetimeIndex(t["time"])),
                          100 * t["risk"].values / t["e_px"].values)

        b4 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h")
        t = kama_gate_btc(run(b4, SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0,
                                                     "fwd": 300})))
        out["btc_bo_kama"] = (pd.Series(t["R"].values, index=pd.DatetimeIndex(t["time"])),
                              100 * t["risk"].values / t["e_px"].values)

        t = cycle_gate_pull(run_pb(b4, "long", SimpleNamespace(**{**PB, "csv": "x", "tf": "4h"}), 0.0))
        out["btc_pull"] = (pd.Series(t["R"].values, index=pd.DatetimeIndex(t["time"])),
                           100 * t["risk"].values / t["e_px"].values)

        g15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g15, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10, "ext_cap": 8.0,
                                        "pullback_frac": 0.25, "fill_win": 200}))
        out["gold15m"] = (pd.Series(t["R"].values - 0.30 / t["risk"].values,
                                    index=pd.DatetimeIndex(t["time"])),
                          100 * t["risk"].values / t["e_px"].values)

        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3, "rr": 4.5, "fill_win": 200}))
        pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
        w = np.where(t["e_px"].values > pdh[d15.index.get_indexer(t["time"])], 1.0, 0.5)
        risk = t["risk"].values / w
        out["btc15m_L"] = (pd.Series(t["R"].values * w - 15.0 / risk,
                                     index=pd.DatetimeIndex(t["time"])),
                           100 * risk / t["e_px"].values)

        inv = invert(d15); C = 2 * d15["high"].max()
        t = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3, "rr": 4.5,
                                        "fill_win": 200}))
        pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
        m = (C - t["e_px"].values) < pdl[d15.index.get_indexer(t["time"])]
        px_s = C - t["e_px"].values                          #真の価格（鏡像を戻す）
        out["btc15m_S"] = (pd.Series(t["R"].values[m] - 15.0 / t["risk"].values[m],
                                     index=pd.DatetimeIndex(t["time"])[m]),
                           100 * t["risk"].values[m] / px_s[m])
    return out


def main():
    L = raw_legs()
    R, sp = L["btc15m_L"]
    yrs = (R.index[-1] - R.index[0]).days / 365.25

    print("S1  これは『2021年を消しているだけ』か？  年別に分解する\n")
    print(f"  {'年':<7}{'全部 n':>8}{'全部 totR':>11}{'<=2.0% n':>10}{'<=2.0% totR':>13}"
          f"{'捨てた本数':>11}{'捨てた分の totR':>16}")
    keep = sp <= 2.0
    tot_cut = 0.0
    for y in sorted(set(R.index.year)):
        m = R.index.year == y
        mk = m & keep
        mc = m & ~keep
        tot_cut += R.values[mc].sum()
        print(f"  {y:<7}{m.sum():>8}{R.values[m].sum():>+11.1f}{mk.sum():>10}{R.values[mk].sum():>+13.1f}"
              f"{mc.sum():>11}{R.values[mc].sum():>+16.1f}")
    print(f"\n  捨てたトレードの合計: {(~keep).sum()}本 / {tot_cut:+.1f}R "
          f"（1本あたり {tot_cut/max((~keep).sum(),1):+.3f}R）")
    print("  → 捨てた分が特定の1年に集中していれば年フィルタ。全年に散っていれば本物。")
    cut_by_y = pd.Series(R.values[~keep], index=R.index[~keep]).groupby(lambda x: x.year).sum()
    worst = cut_by_y.idxmin()
    print(f"     最も損していた年: {worst} ({cut_by_y[worst]:+.1f}R)  / "
          f"捨てた分がマイナスだった年: {(cut_by_y < 0).sum()}年 / {len(cut_by_y)}年")

    print("\n\nS3  IS / OOS（前半・後半で割る）と、しきい値のプラトー\n")
    half = R.index[len(R) // 2]
    print(f"  {'条件':<22}{'n':>6}{'PF':>7}{'meanR':>9}{'IS meanR':>10}{'OOS meanR':>11}{'totR/年':>10}")
    for thr in (99.0, 6.0, 4.0, 3.0, 2.5, 2.0, 1.75, 1.5):
        m = sp <= thr
        if m.sum() < 100:
            continue
        s = R[m]
        i, o = s[s.index < half], s[s.index >= half]
        lab = "全部（現行）" if thr > 50 else f"損切り <= {thr}%"
        print(f"  {lab:<22}{len(s):>6}{pf(s.values):>7.2f}{s.mean():>+9.3f}"
              f"{i.mean():>+10.3f}{o.mean():>+11.3f}{s.sum()/yrs:>+10.1f}")

    print("\n\nS2  ブックで裁定する（構造法則10: レッグの改善はブックの改善ではない）\n")
    legs0 = {k: v[0] for k, v in L.items()}
    c0, d0, r0, n0 = book(legs0, SIX)
    print(f"  {'条件':<34}{'ブックn':>8}{'CAGR':>9}{'maxDD':>8}{'CAGR/DD':>10}")
    print(f"  {'全部（現行の6レッグ）':<34}{n0:>8}{c0:>8.1f}%{d0:>7.2f}%{r0:>10.2f}")
    for thr in (3.0, 2.5, 2.0, 1.75):
        legs = dict(legs0)
        legs["btc15m_L"] = L["btc15m_L"][0][L["btc15m_L"][1] <= thr]
        c, d, r, n = book(legs, SIX)
        print(f"  {'btc15m_L に 損切り<='+str(thr)+'% を課す':<34}{n:>8}{c:>8.1f}%{d:>7.2f}%"
              f"{r:>10.2f}{'  ★' if r > r0 else ''}")

    print("\n  同じ規則を **全レッグ** に課したら（転移するか）")
    for thr in (3.0, 2.0):
        legs = {k: v[0][v[1] <= thr] for k, v in L.items()}
        c, d, r, n = book(legs, SIX)
        print(f"  {'全6レッグに 損切り<='+str(thr)+'%':<34}{n:>8}{c:>8.1f}%{d:>7.2f}%{r:>10.2f}")

    print("\n  レッグ別の効き方（損切り <= 2.0%）")
    print(f"    {'leg':<14}{'n(前)':>7}{'n(後)':>7}{'meanR(前)':>11}{'meanR(後)':>11}{'totR/年(前→後)':>20}")
    for k in SIX:
        s0, spk = L[k]
        s1 = s0[spk <= 2.0]
        yk = (s0.index[-1] - s0.index[0]).days / 365.25
        print(f"    {k:<14}{len(s0):>7}{len(s1):>7}{s0.mean():>+11.3f}{s1.mean():>+11.3f}"
              f"{s0.sum()/yk:>+11.1f} → {s1.sum()/yk:>+6.1f}")


if __name__ == "__main__":
    main()
