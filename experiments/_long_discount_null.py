import sys, io, contextlib
sys.path.insert(0,"/home/angelbell/dev/auto-trade"); sys.path.insert(0,"/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ict_killzone import load_ny, SYMS
from ict_v2_mss import prep, walk, MODEL
from ict_ablation import build, PIP, BUF
from ict_abstain import join_days, sc
from ict_pd_bias import pd_frame
RNG=np.random.default_rng(20260715); F,RR=0.25,4.0
def bb(tr,m,nrep=3000):
    s=pd.Series([t[1] for t in tr],index=pd.to_datetime([t[0] for t in tr])).sort_index()
    g=[x.values for _,x in s.groupby(s.index.to_period("M"))]; nb=max(1,len(g)//m)
    bl=[np.concatenate(g[i*m:(i+1)*m]) for i in range(nb) if len(g[i*m:(i+1)*m])]; bl=[b for b in bl if len(b)]
    if len(bl)<4: return np.nan
    return 100.0*sum(1 for _ in range(nrep) if np.concatenate([bl[i] for i in RNG.integers(0,len(bl),len(bl))]).sum()>0)/nrep
def rdd_null(base,k,nrep=2000):
    net=np.array([t[1] for t in base]); out=[]
    for _ in range(nrep):
        x=net[np.sort(RNG.choice(len(net),k,replace=False))]; cum=np.cumsum(x)
        dd=(np.maximum.accumulate(cum)-cum).max(); out.append(x.sum()/dd if dd>0 else np.inf)
    return np.array(out)
for name in ("eurusd","gbpusd"):
    for spread in (0.3,0.9):
        with contextlib.redirect_stderr(io.StringIO()): df,_=load_ny(SYMS[name])
        df,tarr,dates=prep(df); P=pd_frame(df)
        S=build(df,tarr,dates,True,True,"mss",0); sp=spread*PIP[name]; _,cost=MODEL[name]
        L={d:net for (d,net,g,risk) in walk(df,S,F,RR,BUF,sp,cost,"long")}
        J=join_days(sorted(L),P)
        base=[(d,L[d]) for d in J.index if d in L]
        for band in (0.0,0.10,0.20):
            disc=[(d,L[d]) for d,r in J.iterrows() if not pd.isna(r["pos10"]) and r["pos10"]<0.5-band and d in L]
            s=sc(disc); b=sc(base)
            nul=rdd_null(base,s["n"]); pc=100*(s["rdd"]>nul).mean()
            print(f"{name} spread{spread} band{band:.2f}: base totR/DD={b['rdd']:.2f}(n{b['n']}) -> discount {s['rdd']:.2f}(n{s['n']} PF{s['pf']:.2f} net{s['net']:+.3f}) 間引き%ile={pc:.0f}% blockBoot(1/3/6/12mo)={bb(disc,1):.0f}/{bb(disc,3):.0f}/{bb(disc,6):.0f}/{bb(disc,12):.0f}%")
    print()
