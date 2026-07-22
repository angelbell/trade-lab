"""Two SPEC MISMATCHES found in the machine every book number today was read off.

  M1  research/portfolio_kama.get_legs() loads data/vantage_xauusd_h1.csv with NO --start filter.
      Gold H1 is effectively DAILY data before 2018 (~250-300 bars/yr vs ~5900 after), and today's
      swings_zigzag fix made trades appear in that sparse region. CLAUDE.md now says gold h1 must be
      started at 2018-01-01. So gold_bo carries ~12 trades from garbage-sparse years (2008-2017,
      1-3 trades each) -- which also destroys its concentration stats (16 "years", 8 of them junk).

  M2  build_base()'s gold15m has NO 9-15 UTC session skip, but the documented validated candidate is
      "ext-cap 8% + RR4 + 9-15 UTC skip". breakout_wave.py has no session argument at all, so the leg
      we have been calling gold15m all day is a DIFFERENT (weaker) strategy than the one in the ledger.
      Note the ledger's candidate is RR4 with pullback_frac 0.25; build_base uses RR from BASE.

Both are corrections to the measuring machine, not new ideas. Fix them and re-read the book.
Judged on the corrected arbiter: trade-resolution DD + trade-level-sigma inv-vol at 3%.
Run: .venv/bin/python experiments/book_spec_fix.py
"""
import sys, io, contextlib, warnings
from types import SimpleNamespace
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from research.regime_gate_lab import CFG
from research.portfolio_kama import kama_gate_btc, cycle_gate_pull, PB
from ema_pullback import run as run_pb
from short_mirror_15m import invert
from radar_gate_race import BASE
from book_leave_one_out import cdd

ROOT = "/home/angelbell/dev/auto-trade"
NEW = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]
OLD = ["gold_bo", "btc_bo_kama", "btc_pull"]


def w_trade(legs, basket, budget=0.03):
    sig = pd.Series({k: legs[k].std() for k in basket})
    w = 1.0 / sig
    return w / w.sum() * budget


def book(legs, basket):
    w = w_trade(legs, basket)
    st = max(legs[k].index.min() for k in basket)
    en = min(legs[k].index.max() for k in basket)
    parts = []
    for k in basket:
        s = legs[k][(legs[k].index >= st) & (legs[k].index <= en)]
        parts.append(pd.Series(s.values * w[k], index=s.index))
    s = pd.concat(parts).sort_index()
    return cdd(s.values, (s.index[-1] - s.index[0]).days) + (len(s),)


def conc(s, tag):
    """top-3-year share + year gini + what's left after removing those 3 years."""
    yr = s.groupby(s.index.year).sum()
    yr = yr[(yr.index >= 2019) & (yr.index <= 2025)]          # complete years only
    tot = yr.sum()
    top3 = yr.sort_values(ascending=False).head(3)
    rest = s[(s.index.year >= 2019) & (s.index.year <= 2025) & (~s.index.year.isin(top3.index))]
    pf = rest[rest > 0].sum() / abs(rest[rest <= 0].sum()) if (rest <= 0).any() else np.nan
    x = np.sort(yr.values); n = len(x)
    gini = (2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum()) - (n + 1) / n) if x.sum() > 0 else np.nan
    print(f"  {tag:<34}n={len(s):>4}  上位3年={100*top3.sum()/tot:>5.1f}%  ジニ={gini:>5.2f}  "
          f"抜いた残り: n={len(rest):>4} PF={pf:.2f} meanR={rest.mean():+.3f}")


def build(gold_start, gold15m_skip):
    with contextlib.redirect_stderr(io.StringIO()):
        g1h = load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv")
        if gold_start:
            g1h = g1h.loc[gold_start:]
        gb = run(resample(g1h, "1h"), SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0,
                                                         "fwd": 500, "daily_sma": 150,
                                                         "daily_slope_k": 10}))
        legs = {"gold_bo": pd.Series(gb["R"].values, index=pd.DatetimeIndex(gb["time"]))}
        b4 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h")
        bo = run(b4, SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0, "fwd": 300}))[["time", "R"]]
        bk = kama_gate_btc(bo)
        legs["btc_bo_kama"] = pd.Series(bk.R.values, index=pd.DatetimeIndex(bk.time))
        pb = cycle_gate_pull(run_pb(b4, "long", SimpleNamespace(**{**PB, "csv": "x", "tf": "4h"}), 0.0)[["time", "R"]])
        legs["btc_pull"] = pd.Series(pb.R.values, index=pd.DatetimeIndex(pb.time))

        g15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g15, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                        "ext_cap": 8.0, "pullback_frac": 0.25}))
        R = t["R"].values - 0.3 / t["risk"].values
        idx = pd.DatetimeIndex(t["time"])
        if gold15m_skip:                       # the ledger's candidate: skip the 9-15 UTC dead window
            keep = ~idx.hour.isin(range(9, 15))
            R, idx = R[keep], idx[keep]
        legs["gold15m"] = pd.Series(R, index=idx)

        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        tL = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                         "pullback_frac": 0.3, "rr": 4.5}))
        RL = tL["R"].values - 15.0 / tL["risk"].values
        ei = d15.index.get_indexer(tL["time"])
        pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
        legs["btc15m_L"] = pd.Series(RL * np.where(tL["e_px"].values > pdh[ei], 1.0, 0.5),
                                     index=pd.DatetimeIndex(tL["time"]))
        inv = invert(d15); C = 2 * d15["high"].max()
        ts = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3}))
        Rs = ts["R"].values - 15.0 / ts["risk"].values
        pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
        mS = (C - ts["e_px"].values) < pdl[d15.index.get_indexer(ts["time"])]
        legs["btc15m_S"] = pd.Series(Rs[mS], index=pd.DatetimeIndex(ts["time"])[mS])
    return legs


def main():
    arms = {
        "現状（今日ずっと使っていた機械）": (None, False),
        "M1 のみ修正: gold h1 を 2018- に": ("2018-01-01", False),
        "M2 のみ修正: gold15m に 9-15UTC スキップ": (None, True),
        "M1+M2 両方修正（台帳どおりの仕様）": ("2018-01-01", True),
    }
    print(f"{'arm':<40}{'3-leg C/DD':>12}{'6-leg C/DD':>12}{'6-leg DD':>10}{'gold_bo n':>11}{'gold15m n':>11}")
    keep = {}
    for tag, (gs, sk) in arms.items():
        L = build(gs, sk)
        keep[tag] = L
        c3 = book(L, OLD); c6 = book(L, NEW)
        print(f"{tag:<40}{c3[2]:>12.2f}{c6[2]:>12.2f}{c6[1]:>9.2f}%"
              f"{len(L['gold_bo']):>11}{len(L['gold15m']):>11}")

    print("\n集中度（完全な年 2019-2025 のみ、修正後の機械で）")
    L = keep["M1+M2 両方修正（台帳どおりの仕様）"]
    L0 = keep["現状（今日ずっと使っていた機械）"]
    conc(L0["gold_bo"], "gold_bo  修正前（疎データ込み）")
    conc(L["gold_bo"], "gold_bo  修正後（2018-）")
    conc(L0["gold15m"], "gold15m  修正前（スキップ無し）")
    conc(L["gold15m"], "gold15m  修正後（9-15UTC スキップ）")
    conc(L["btc15m_L"], "btc15m_L （比較用）")

    print("\n修正後の機械での leave-one-out（6レッグ）")
    base = book(L, NEW)[2]
    print(f"  {'6-leg (all)':<24}{base:>8.2f}")
    for k in NEW:
        x = book(L, [j for j in NEW if j != k])[2]
        print(f"  {'  minus ' + k:<24}{x:>8.2f}{x - base:>+9.2f}")


if __name__ == "__main__":
    main()
