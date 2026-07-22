import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0,"."); sys.path.insert(0,"experiments")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE
with contextlib.redirect_stderr(io.StringIO()):
    d15 = resample(load_mt5_csv("data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    t = run(d15, SimpleNamespace(**{**BASE,"gate_kama":14,"gate_kama_tf":"240min",
            "pullback_frac":0.3,"rr":4.5,"fill_win":200,"fwd":500}))
ei=d15.index.get_indexer(t["time"]); e=t["e_px"].values; rk=t["risk"].values
stop,tgt=e-rk,e+(4.8/0.7)*rk
hi,lo,cl=d15["high"].values,d15["low"].values,d15["close"].values
R=np.empty(len(e)); H=np.empty(len(e))
for i in range(len(e)):
    j0=ei[i]; lim=min(j0+500,len(cl)-1)
    if lo[j0]<=stop[i]: r,jj=-1.0,j0
    else:
        r=None
        for j in range(j0+1,lim+1):
            if lo[j]<=stop[i]: r,jj=-1.0,j; break
            if hi[j]>=tgt[i]: r,jj=4.8/0.7,j; break
        if r is None: jj=lim; r=(cl[jj]-e[i])/rk[i]
    R[i]=r; H[i]=(jj-j0)*0.25/24
d=np.abs(R-t["R"].values); idx=np.argsort(-d)[:6]
print("差の大きいトレード（re-walk R / 正典 R / 保有 re / 正典）")
for i in idx:
    print(f"  {t['time'].values[i]}  re={R[i]:+.4f} canon={t['R'].values[i]:+.4f} "
          f"hold re={H[i]:.3f} canon={t['hold'].values[i]:.3f}  risk={rk[i]:.2f} e={e[i]:.1f}")
print("差>1e-6 の本数:", int((d>1e-6).sum()), "/", len(e))
