"""a10_followup.py -- complete the A10 evaluation for the surviving cell: BTC 15m long
with a gate-MATCHED 4h KAMA-falling exit (exit when the series that let you in turns off).

1) $25 cost stress (spread-only instrument; observed $25 on a quiet day).
2) Single-leg f-ladder: monthly joint bootstrap 1yr multiplier at f 1/2/3% (user axis).
3) Book swap: the 6-leg new book with btc15m_L rebuilt with the exit -> 3% ladder + corr.
PDH soft-0.5 weighting applied post-hoc as in book_integration.py.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from research.portfolio_kama import get_legs
from radar_gate_race import BASE
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from short_mirror_15m import invert

rng = np.random.default_rng(7)

b = load_mt5_csv("data/vantage_btcusd_m15.csv")
cnt = b.groupby(b.index.date).size()
okd = cnt[cnt.rolling(30).median() >= 80]
d15 = resample(b[b.index.date >= okd.index[0]], "15min")
span = (d15.index[-1] - d15.index[0]).days / 365.25
pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values

CELL = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3}


def leg(extra, rt):
    t = run(d15, SimpleNamespace(**{**CELL, **extra}))
    Rn = t["R"].values - rt / t["risk"].values
    ab = t["e_px"].values > pdh[d15.index.get_indexer(t["time"])]
    return pd.Series(Rn * np.where(ab, 1.0, 0.5), index=pd.DatetimeIndex(t["time"]))


def stats(tag, s):
    yr = s.index.year
    half = np.median(yr)
    pf = s[s > 0].sum() / abs(s[s <= 0].sum())
    eq = s.cumsum().values
    dd = (np.maximum.accumulate(eq) - eq).max()
    print(f"  {tag:<26} N/yr={len(s)/span:5.1f} PF={pf:4.2f} meanR={s.mean():+.3f} "
          f"IS/OOS={s[yr<half].mean():+.2f}/{s[yr>=half].mean():+.2f} "
          f"totR/yr={s.sum()/span:+5.1f} DD={dd:5.1f}R")


def ladder(port, tag):
    mult = np.array([np.prod(1 + port[rng.integers(0, len(port), 12)]) for _ in range(4000)])
    eq = np.cumprod(1 + port)
    ddp = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
    cagr = (eq[-1] ** (12 / len(port)) - 1) * 100
    print(f"  {tag:<34} CAGR={cagr:5.1f}% maxDD={ddp:4.1f}% CAGR/DD={cagr/ddp:5.2f} | "
          f"1yr倍率 med={np.median(mult):.2f} sd={mult.std():.2f} p10={np.percentile(mult,10):.2f} "
          f"p90={np.percentile(mult,90):.2f} P2x={(mult>=2).mean()*100:2.0f}% "
          f"P半減={(mult<=0.5).mean()*100:.1f}%")


print(f"===== A10 follow-up: BTC15m long +PDHsoft ({span:.1f}yr) =====")
variants = {}
for tag, extra in [("exit OFF", {}), ("exit KAMA14 4h", {"exit_kama": 14, "exit_kama_tf": "240min"})]:
    for rt in (15.0, 25.0):
        s = leg(extra, rt)
        stats(f"{tag} ${rt:.0f}", s)
        variants[(tag, rt)] = s

print("\n-- 単レッグ f-ladder (月次ブートストラップ4000, net $15) --")
for f in (0.01, 0.02, 0.03):
    for tag in ("exit OFF", "exit KAMA14 4h"):
        s = variants[(tag, 15.0)]
        mon = (s * f).groupby(s.index.to_period("M")).sum()
        midx = pd.period_range(mon.index.min(), mon.index.max(), freq="M")
        ladder(mon.reindex(midx, fill_value=0.0).values, f"f={f*100:.0f}% {tag}")

# ---- book swap ----
legs = {}
for k, t in get_legs().items():
    legs[k] = pd.Series(t.R.values, index=pd.DatetimeIndex(t.time))
g = resample(load_mt5_csv("data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
t = run(g, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                              "ext_cap": 8.0, "pullback_frac": 0.25}))
legs["gold15m"] = pd.Series(t["R"].values - 0.3 / t["risk"].values, index=pd.DatetimeIndex(t["time"]))
inv = invert(d15)
ts_ = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3}))
Rs = ts_["R"].values - 15.0 / ts_["risk"].values
pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
C = 2 * d15["high"].max()
mS = (C - ts_["e_px"].values) < pdl[d15.index.get_indexer(ts_["time"])]
legs["btc15m_S"] = pd.Series(Rs[mS], index=pd.DatetimeIndex(ts_["time"])[mS])

print("\n-- 新6レッグブック: btc15m_L の出口 OFF vs 4h (総リスク3%, inv-vol) --")
for tag in ("exit OFF", "exit KAMA14 4h"):
    L = dict(legs)
    L["btc15m_L"] = variants[(tag, 15.0)]
    mon = {k: s.groupby(s.index.to_period("M")).sum() for k, s in L.items()}
    start = max(s.index.min() for s in mon.values())
    end = min(s.index.max() for s in mon.values())
    midx = pd.period_range(start, end, freq="M")
    M = pd.DataFrame({k: v.reindex(midx, fill_value=0.0) for k, v in mon.items()})
    w = 1.0 / M.std()
    w = w / w.sum() * 0.03
    ladder((M * w).sum(axis=1).values, f"ブック {tag}")
    if tag == "exit OFF":
        c0 = M.corr()["btc15m_L"].drop("btc15m_L").round(2)
    else:
        print(f"    btc15m_L相関の変化: OFF {c0.to_dict()} -> 4h "
              f"{M.corr()['btc15m_L'].drop('btc15m_L').round(2).to_dict()}")
