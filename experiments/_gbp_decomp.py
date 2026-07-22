import sys, io, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade"); sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ict_killzone import load_ny, SYMS
from ict_v2_mss import prep, walk, MODEL
from ict_ablation import build, PIP, BUF
from ict_abstain import join_days, sc
from ict_pd_bias import pd_frame
F, RR = 0.25, 4.0
ERAS=[(2000,2008),(2009,2016),(2017,2020),(2021,2026)]
def eras(tr): return " ".join(f"{sum(x[1] for x in tr if a<=pd.Timestamp(x[0]).year<=b):+6.0f}" for a,b in ERAS)
for name in ("gbpusd","eurusd"):
    with contextlib.redirect_stderr(io.StringIO()): df,_=load_ny(SYMS[name])
    df,tarr,dates=prep(df); P=pd_frame(df)
    S=build(df,tarr,dates,True,True,"mss",0); sp=0.3*PIP[name]; _,cost=MODEL[name]
    pool={s:{d:net for (d,net,g,risk) in walk(df,S,F,RR,BUF,sp,cost,s)} for s in ("long","short")}
    J=join_days(sorted(set(list(pool["long"])+list(pool["short"]))),P)
    print(f"\n=== {name} ===  pos10 band=0.20 を サイド分解")
    for poscol,band in (("pos10",0.20),):
        lo,hi=0.5-band,0.5+band
        # long-only (discount), short-only (premium), 各サイドの素のベースも
        longD=[(d,pool["long"][d]) for d,r in J.iterrows() if not pd.isna(r[poscol]) and r[poscol]<lo and d in pool["long"]]
        shortP=[(d,pool["short"][d]) for d,r in J.iterrows() if not pd.isna(r[poscol]) and r[poscol]>hi and d in pool["short"]]
        longAll=[(d,pool["long"][d]) for d in J.index if d in pool["long"]]
        shortAll=[(d,pool["short"][d]) for d in J.index if d in pool["short"]]
        for lab,tr in (("ロング素(全)",longAll),("ロング@discount",longD),("ショート素(全)",shortAll),("ショート@premium",shortP)):
            s=sc(tr)
            if s: print(f"  {lab:18s} n={s['n']:4d} PF={s['pf']:.2f} net={s['net']:+.3f} totR={s['tot']:+6.1f} totR/DD={s['rdd']:5.2f}  時代 {eras(tr)}")
