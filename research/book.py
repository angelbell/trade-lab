"""The canonical BOOK pipeline — build the adopted 6 legs at DEPLOYED spec and
judge them with the adopted arbiter.

Deployed spec (what the Pine files actually order): pullback-limit fill window
200 bars, btc15m_S target RR 4.5, net costs per leg (gold $0.3, BTC $15),
PDH-soft 0.5 on btc15m_L, PDL-hard mask on btc15m_S.

Arbiter (CLAUDE.md checklist 8): maxDD at TRADE resolution (never monthly),
weights = inverse of each leg's trade-R sigma, scaled to 3% total risk.

Anchor (adopted 2026-07-13, after the fill-bar bug fix): the 6-leg book prints
n/yr ≈ 206 / CAGR ≈ +61.0% / maxDD ≈ 7.74% / CAGR/DD ≈ 7.88.

Lifted verbatim from experiments/book_deployed_spec.py + book_spec_fix.py
(kept frozen as evidence); size overlays go through src/engine/size (proven
array-identical to the originals by invariants/size_tieback.py). Guarded by
invariants/book_tieback.py — legs and verdict must match the frozen scripts.

Run: .venv/bin/python research/book.py
"""
import io
import contextlib
import os
import sys
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_mt5_csv, GOLD_H1_START
from breakout_wave import run, resample
from ema_pullback import run as run_pb
from research.regime_gate_lab import CFG
from research.portfolio_kama import kama_gate_btc, cycle_gate_pull, PB
from src.engine.presets import BASE
from src.engine.mirror import invert
from src.engine.size import pdh_soft, pdl_break_mask

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIX = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]


def cdd(vals, days):
    """Trade-resolution (CAGR%, maxDD%, CAGR/DD) — never collapse to monthly."""
    eq = np.cumprod(1 + vals); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    if dd <= 0:
        return np.nan, np.nan, np.nan
    cagr = (eq[-1] ** (365.25 / days) - 1) * 100
    return cagr, dd, cagr / dd


def w_trade(legs, basket, budget=0.03):
    """Inverse trade-R sigma weights at constant total risk (the adopted scheme;
    monthly-sigma inv-vol misreads no-trade months as low vol — s07)."""
    sig = pd.Series({k: legs[k].std() for k in basket})
    w = 1.0 / sig
    return w / w.sum() * budget


def book(legs, basket):
    """(CAGR%, maxDD%, CAGR/DD, n) of the weighted book on the common span."""
    w = w_trade(legs, basket)
    st = max(legs[k].index.min() for k in basket)
    en = min(legs[k].index.max() for k in basket)
    parts = []
    for k in basket:
        s = legs[k][(legs[k].index >= st) & (legs[k].index <= en)]
        parts.append(pd.Series(s.values * w[k], index=s.index))
    s = pd.concat(parts).sort_index()
    return cdd(s.values, (s.index[-1] - s.index[0]).days) + (len(s),)


def get_book_legs(fill_win=200, rr_short=4.5):
    """The 6 adopted legs as net-R series (deployed spec by default)."""
    with contextlib.redirect_stderr(io.StringIO()):
        g1 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv").loc[GOLD_H1_START:], "1h")
        gb = run(g1, SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                        "daily_sma": 150, "daily_slope_k": 10}))
        legs = {"gold_bo": pd.Series(gb["R"].values, index=pd.DatetimeIndex(gb["time"]))}

        b4 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h")
        bk = kama_gate_btc(run(b4, SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0,
                                                      "fwd": 300}))[["time", "R"]])
        legs["btc_bo_kama"] = pd.Series(bk.R.values, index=pd.DatetimeIndex(bk.time))
        pb = cycle_gate_pull(run_pb(b4, "long", SimpleNamespace(**{**PB, "csv": "x", "tf": "4h"}),
                                    0.0)[["time", "R"]])
        legs["btc_pull"] = pd.Series(pb.R.values, index=pd.DatetimeIndex(pb.time))

        g15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g15, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10, "ext_cap": 8.0,
                                        "pullback_frac": 0.25, "fill_win": fill_win}))
        legs["gold15m"] = pd.Series(t["R"].values - 0.3 / t["risk"].values,
                                    index=pd.DatetimeIndex(t["time"]))

        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        tL = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                         "pullback_frac": 0.3, "rr": 4.5, "fill_win": fill_win}))
        WL, _ = pdh_soft(d15, tL)
        legs["btc15m_L"] = pd.Series((tL["R"].values - 15.0 / tL["risk"].values) * WL,
                                     index=pd.DatetimeIndex(tL["time"]))

        inv = invert(d15); C = 2 * d15["high"].max()
        ts = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3,
                                         "rr": rr_short, "fill_win": fill_win}))
        mS = pdl_break_mask(d15, ts, C)
        legs["btc15m_S"] = pd.Series((ts["R"].values - 15.0 / ts["risk"].values)[mS],
                                     index=pd.DatetimeIndex(ts["time"])[mS])
    return legs


def main():
    legs = get_book_legs()
    c0, d0, r0, n0 = book(legs, SIX)
    st = max(legs[k].index.min() for k in SIX)
    en = min(legs[k].index.max() for k in SIX)
    yrs = (en - st).days / 365.25

    print("運用仕様（fill_win=200 / S の RR=4.5）・審判＝トレード解像度DD × トレードRσ逆数・総リスク3%\n")
    print(f"  {'leg':<14}{'n':>5}{'本/年':>7}{'勝率':>7}{'PF':>7}{'meanR':>9}{'重み':>8}")
    w = w_trade(legs, SIX)
    for k in SIX:
        s = legs[k]
        pf = s[s > 0].sum() / abs(s[s <= 0].sum())
        y = (s.index[-1] - s.index[0]).days / 365.25
        print(f"  {k:<14}{len(s):>5}{len(s)/y:>7.0f}{100*(s>0).mean():>6.1f}%{pf:>7.2f}"
              f"{s.mean():>+9.3f}{100*w[k]:>7.3f}%")
    print(f"\n  ブック合計: n={n0}  {n0/yrs:.0f}本/年  CAGR {c0:+.1f}%  maxDD {d0:.2f}%  "
          f"CAGR/DD **{r0:.2f}**   (採用アンカー: 206本/年・+61.0%・7.74%・7.88)")

    print(f"\n  1つ抜いたら（leave-one-out）")
    print(f"  {'抜いた leg':<14}{'CAGR':>9}{'maxDD':>8}{'CAGR/DD':>10}{'差':>9}")
    print(f"  {'（6本全部）':<14}{c0:>8.1f}%{d0:>7.2f}%{r0:>10.2f}{'':>9}")
    for k in SIX:
        c, d, r, _ = book(legs, [x for x in SIX if x != k])
        print(f"  {k:<14}{c:>8.1f}%{d:>7.2f}%{r:>10.2f}{r-r0:>+9.2f}")


if __name__ == "__main__":
    main()
