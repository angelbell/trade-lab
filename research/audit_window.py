"""overfit audit for the 15M gold_bo+ext-cap leg with a DROP-TIME-WINDOW search.
config space = every contiguous drop-window I could have picked (start 5..18 x width 3..6),
plus no-drop. CSCV PBO then measures whether window-SELECTION is overfitting; DSR deflates
the chosen window by the number of windows searched. Chosen = drop 9-15 UTC."""
import os, sys, subprocess, itertools, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import numpy as np, pandas as pd
from research.overfit_audit import psr, sr0, cscv, cdd_R, block_resample

CFG=["--csv","data/vantage_xauusd_m5.csv","--tf","15min","--pattern","B","--swing","zigzag",
     "--zz-k","2","--trend-ema","80","--bo-window","20","--tp-mode","rr","--rr","4","--fwd","500",
     "--daily-sma","150","--daily-slope-k","10","--risk","0.01","--cost","0.0002","--ext-cap","8"]
out=subprocess.run([".venv/bin/python","breakout_wave.py",*CFG,"--dump-trades"],
                   capture_output=True,text=True,cwd="/home/angelbell/dev/auto-trade").stdout.splitlines()
i=next(k for k,l in enumerate(out) if l.startswith("entry_time,"))
base=pd.read_csv(pd.io.common.StringIO("\n".join(out[i:])))
base["time"]=pd.to_datetime(base["entry_time"],utc=True); base["R"]=base["R"].astype(float)
base["h"]=base["time"].dt.hour; base=base.sort_values("time")

def keep(start,width):  # drop hours [start, start+width)
    if width==0: return base
    return base[~base.h.between(start,start+width-1)]

# ---- build the WINDOW config grid (the real search space) ----
configs=[(0,0)]+[(s,w) for s in range(5,19) for w in (3,4,5,6)]
cols,srs,labels=[],[],[]
for (s,w) in configs:
    t=keep(s,w)
    if len(t)<30: continue
    m=t.set_index("time").R.groupby(pd.Grouper(freq="M")).sum()
    cols.append(m.rename(f"{s}_{w}")); srs.append(t.R.mean()/t.R.std(ddof=1)); labels.append((s,w))
M=pd.concat(cols,axis=1).fillna(0.0).values
V=float(np.var(srs))
print(f"window configs tried: {len(labels)}  (Sharpe variance V={V:.4f})")

chosen=keep(9,6)  # drop 9-15
R=chosen.R.values
yrs=max((chosen.time.max()-chosen.time.min()).days/365.25,.5)

print("\n=== A. DEFLATED SHARPE (deflated by the #windows searched) ===")
_,sr,g1,g4=psr(R,0.0); t_=sr*np.sqrt(len(R))
print(f"  drop9-15  n={len(R)}  SR/tr={sr:+.3f}  t={t_:.2f}  skew={g1:+.2f} kurt={g4:.1f}")
for N in [1,len(labels),50,100,200]:
    print(f"    DSR@{N:>3}={psr(R,sr0(N,V))[0]:.2f}", end="")
print("\n  (N=#trials; DSR>0.95 survives the haircut. N=%d = the actual window search)"%len(labels))

print("\n=== B. PBO via CSCV over the WINDOW configs (does window-SELECTION overfit?) ===")
real,noise,oosm=[],[],[]
for sd in range(24):
    pbo,om,_=cscv(M,S=10,seed=sd); real.append(pbo); oosm.append(om)
    noise.append(cscv(np.random.default_rng(sd).standard_normal(M.shape),S=10,seed=sd)[0])
real,noise=np.array(real),np.array(noise)
print(f"  grid={M.shape[1]} window-cfg x {M.shape[0]} months")
print(f"  REAL  PBO={real.mean():.2f}   IS-best mean OOS-Sharpe={np.mean(oosm):+.2f}")
print(f"  NOISE PBO={noise.mean():.2f} (must center ~0.50)   gap={real.mean()-noise.mean():+.2f}")

print("\n=== C. BLOCK-BOOTSTRAP CI on CAGR/DD + mean-removed NULL ===")
obs=cdd_R(R,yrs)[2]; B,L=4000,5; rng=np.random.default_rng(7)
boot=np.array([cdd_R(block_resample(R,L,rng),yrs)[2] for _ in range(B)])
rn=R-R.mean(); nul=np.array([cdd_R(block_resample(rn,L,rng),yrs)[2] for _ in range(B)])
p=(nul>=obs).mean(); c5,c50,c95=np.percentile(boot,[5,50,95])
print(f"  obs CAGR/DD={obs:+.2f}  CI[5/50/95]={c5:+.2f}/{c50:+.2f}/{c95:+.2f}  null p={p:.3f}")
