"""NEW BOOK integration: adopted 3 legs + today's 15m additions (gold15m, btc15m-long
PDH-soft, btc15m-short PDL). Monthly-R composite (joint-month bootstrap preserves cross-
correlation), inv-vol weights at CONSTANT total risk, f-ladder 1yr multiplier: old book
vs new book vs new book with BTC-family parity cap. Plus the monthly corr matrix."""
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

legs = {}
for k, t in get_legs().items():
    legs[k] = pd.Series(t.R.values, index=pd.DatetimeIndex(t.time))

# gold 15m canon ($0.3)
g = resample(load_mt5_csv("data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
t = run(g, SimpleNamespace(**{**BASE, "daily_sma":150, "daily_slope_k":10, "ext_cap":8.0, "pullback_frac":0.25}))
legs["gold15m"] = pd.Series(t["R"].values - 0.3/t["risk"].values, index=pd.DatetimeIndex(t["time"]))

# BTC 15m long: kama4h + PDH soft0.5 ($15)
b = load_mt5_csv("data/vantage_btcusd_m15.csv").loc["2018-10-01":]
d15 = resample(b, "15min")
t = run(d15, SimpleNamespace(**{**BASE, "gate_kama":14, "gate_kama_tf":"240min", "pullback_frac":0.3}))
Rn = t["R"].values - 15.0/t["risk"].values
pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
ab = t["e_px"].values > pdh[d15.index.get_indexer(t["time"])]
legs["btc15m_L"] = pd.Series(Rn*np.where(ab,1.0,0.5), index=pd.DatetimeIndex(t["time"]))

# BTC 15m short: inverted, kama1d, PDL hard ($15)
inv = invert(d15); C = 2*d15["high"].max()
ts_ = run(inv, SimpleNamespace(**{**BASE, "gate_kama":14, "pullback_frac":0.3}))
Rs = ts_["R"].values - 15.0/ts_["risk"].values
pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
mS = (C - ts_["e_px"].values) < pdl[d15.index.get_indexer(ts_["time"])]
legs["btc15m_S"] = pd.Series(Rs[mS], index=pd.DatetimeIndex(ts_["time"])[mS])

# ---- monthly matrix on common span ----
mon = {k: s.groupby(s.index.to_period("M")).sum() for k, s in legs.items()}
start = max(s.index.min() for s in mon.values())
end = min(s.index.max() for s in mon.values())
midx = pd.period_range(start, end, freq="M")
M = pd.DataFrame({k: v.reindex(midx, fill_value=0.0) for k, v in mon.items()})
print(f"common span: {start} -> {end} ({len(midx)} months)")
print("\nmonthly R corr matrix:")
print(M.corr().round(2).to_string())

sig = M.std()
OLD = ["gold_bo", "btc_bo_kama", "btc_pull"]
NEW = OLD + ["gold15m", "btc15m_L", "btc15m_S"]
def weights(names, total, parity=False):
    w = (1.0/sig[names]); w = w/w.sum()*total
    if parity:
        btc = [n for n in names if "btc" in n]; gld = [n for n in names if "gold" in n]
        sb, sg = w[btc].sum(), w[gld].sum()
        if sb > sg:
            w[btc] *= sg/sb
            w = w/w.sum()*total
    return w

rng = np.random.default_rng(7)
def ladder(w, tag):
    port = (M[w.index]*w).sum(axis=1).values   # monthly account return at these f-weights
    for _ in range(1):
        mult = np.array([np.prod(1+port[rng.integers(0,len(port),12)]) for _ in range(4000)])
    eq = np.cumprod(1+port)
    dd = ((np.maximum.accumulate(eq)-eq)/np.maximum.accumulate(eq)).max()*100
    yrs = len(port)/12
    cagr = (eq[-1]**(1/yrs)-1)*100
    print(f"  {tag:<28} CAGR={cagr:5.1f}% maxDD={dd:4.1f}% CAGR/DD={cagr/dd:4.2f} | "
          f"1yr倍率 med={np.median(mult):.2f} sd={mult.std():.2f} p10={np.percentile(mult,10):.2f} "
          f"p90={np.percentile(mult,90):.2f} P2x={(mult>=2).mean()*100:2.0f}% P半減={(mult<=0.5).mean()*100:.1f}%")

print("\n===== f-ladder (総リスク一定・inv-vol配分・月次ブートストラップ4000) =====")
for total in (0.02, 0.03):
    print(f"-- total risk {total*100:.0f}% --")
    ladder(weights(OLD, total), f"旧ブック3レッグ")
    ladder(weights(NEW, total), f"新ブック6レッグ")
    ladder(weights(NEW, total, parity=True), f"新6レッグ+BTC≤goldパリティ")
print("\nweights (total 3%, inv-vol):")
print((weights(NEW, 0.03)*100).round(2).to_string())
print("\nweights (total 3%, parity):")
print((weights(NEW, 0.03, parity=True)*100).round(2).to_string())
