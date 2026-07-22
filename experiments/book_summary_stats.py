import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd, pandas_ta as ta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from research.portfolio_kama import get_legs
def card(tag, R, ts):
    span = (ts.max() - ts.min()).days / 365.25
    pf = R[R>0].sum()/abs(R[R<=0].sum())
    print(f"{tag:<22} N/yr={len(R)/span:5.1f}  PF={pf:4.2f}  win={(R>0).mean()*100:4.1f}%  meanR={R.mean():+.3f}")
legs = get_legs()
for k, t in legs.items():
    card(k, t.R.values, pd.DatetimeIndex(t.time))
# BTC 15m kama4h + PDH variants
from radar_gate_race import BASE
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
d15 = resample(load_mt5_csv("data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
t = run(d15, SimpleNamespace(**{**BASE, "gate_kama":14, "gate_kama_tf":"240min", "pullback_frac":0.3}))
Rn = t["R"].values - 15.0/t["risk"].values
dh = d15["high"].resample("1D").max().dropna()
pdh = dh.shift(1).reindex(d15.index, method="ffill").values
ab = (t["e_px"].values > pdh[d15.index.get_indexer(t["time"])])
ts = pd.DatetimeIndex(t["time"])
card("btc15m kama4h full", Rn, ts)
card("btc15m PDHハード", Rn[ab], ts[ab])
card("btc15m PDHソフト0.5", Rn*np.where(ab,1.0,0.5), ts)
