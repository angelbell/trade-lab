"""RR5がスパイクでなくプラトーか: gold15m単体でRRを細かく掃引(3.0..7.0)。
各RRで n/win/PF/meanR/IS-OOS/CAGR@1%/中央値ブートDD/資金倍率中央値。丘型か単発かを見る。"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from src.engine.presets import BASE
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
g15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")

def cdd_median(vals, times, k_months=3, nb=2000, seed=20260719):
    """中央値ブートDD(巡回ブロック)と CAGR@1%固定リスク."""
    r = np.asarray(vals)
    # CAGR at 1% risk (fixed fractional)
    eq = np.cumprod(1 + 0.01 * r)
    days = (times[-1] - times[0]).days
    cagr = (eq[-1] ** (365.25 / max(days,1)) - 1) * 100
    # median DD via cyclic block bootstrap of trade sequence
    df = pd.DataFrame({"r": r, "t": pd.DatetimeIndex(times)})
    df["mk"] = df["t"].dt.to_period("M")
    keys = df["mk"].unique(); blk = k_months
    rng = np.random.default_rng(seed); dds = []
    for _ in range(nb):
        pick = rng.integers(0, len(keys), size=int(np.ceil(len(keys)/blk))); sel=[]
        for p in pick: sel.extend(keys[p:p+blk])
        rr = df[df["mk"].isin(sel)]["r"].values
        if len(rr) < 5: continue
        e = np.cumprod(1 + 0.01*rr); pk = np.maximum.accumulate(e)
        dds.append(((pk-e)/pk).max()*100)
    return cagr, float(np.median(dds))

def mult_median(vals, nyr, seed=20260719, nb=3000):
    """1年窓(nyr本)・k=3moブロック相当の口座倍率中央値（簡易: nyr本を復元抽出）."""
    r = np.asarray(vals); rng = np.random.default_rng(seed); ms=[]
    n1 = max(5, int(round(nyr)))
    for _ in range(nb):
        s = rng.choice(r, n1, replace=True)
        ms.append(np.prod(1 + 0.01*s))
    return float(np.median(ms))

print(f"{'RR':>4} {'n':>4} {'n/yr':>5} {'win%':>5} {'PF':>5} {'meanR':>7} {'IS':>6} {'OOS':>6} "
      f"{'CAGR%':>6} {'medDD%':>6} {'CAGR/DD':>7} {'倍率':>6}")
for rr in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]:
    t = run(g15, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                    "ext_cap": 8.0, "pullback_frac": 0.25, "fill_win": 200, "rr": rr}))
    R = t["R"].values - 0.3 / t["risk"].values
    times = pd.DatetimeIndex(t["time"])
    n = len(R); nyr = n / ((times[-1]-times[0]).days/365.25)
    win = (R > 0).mean()*100
    pf = R[R>0].sum() / -R[R<0].sum()
    half = len(R)//2
    is_, oos = R[:half].mean(), R[half:].mean()
    cagr, meddd = cdd_median(R, times)
    mult = mult_median(R, nyr)
    print(f"{rr:>4.1f} {n:>4} {nyr:>5.1f} {win:>5.1f} {pf:>5.2f} {R.mean():>+7.3f} "
          f"{is_:>+6.3f} {oos:>+6.3f} {cagr:>6.1f} {meddd:>6.1f} {cagr/meddd:>7.2f} {mult:>5.2f}x")
