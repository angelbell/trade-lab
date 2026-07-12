"""STEP 1: replace the parametric frac limit with a STRUCTURAL anchor (param-free):
limit at H1 (the broken level = classic breakout retest), and at the L2->H1 midpoint.
e>H1>L2 so retracement-to-H1 is a DATA-DRIVEN depth per trade. Does a param-free structural
anchor match/beat the best fixed frac, and drop the 'which depth' problem entirely?"""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag
AGG={"open":"first","high":"max","low":"min","close":"last"}
RR,BO,FWD=4.0,20,500; SP=0.6
d=load_mt5_csv("data/vantage_xauusd_m5.csv").resample("15min").agg(AGG).dropna()
h,l,c=d["high"].values,d["low"].values,d["close"].values
a=ta.atr(d["high"],d["low"],d["close"],14).values
es=d["close"].ewm(span=80,adjust=False).mean().values
dc=d["close"].resample("1D").last().dropna(); sma=dc.rolling(150).mean()
reg=((dc>sma)&(sma>sma.shift(10))).shift(1).reindex(d.index,method="ffill").fillna(False).values
ext_arr=(((dc-sma)/sma*100.0).shift(1)).reindex(d.index,method="ffill").values
sw=swings_zigzag(h,l,a,2.0)
def first_breakout(level,after):
    for j in range(after,min(after+BO,len(c))):
        if c[j]>level: return j
    return None
entries=[]  # (i,e,stop=L2,tgt,H1)
for t in range(2,len(sw)):
    (cL2,iL2,pL2,kL2),(cH1,iH1,pH1,kH1),(cL0,iL0,pL0,kL0)=sw[t],sw[t-1],sw[t-2]
    if not(kL2==-1 and kH1==+1 and kL0==-1): continue
    if pL2<=pL0 or pH1-pL0<=0: continue
    if es is not None and not np.isnan(es[cL2]) and pH1<es[cL2]: continue
    e_i=first_breakout(pH1,cL2+1)
    if e_i is None: continue
    if not reg[e_i]: continue
    if not np.isnan(ext_arr[e_i]) and ext_arr[e_i]>8: continue
    e=c[e_i]; stop=pL2; risk=e-stop
    if risk<=0: continue
    tgt=e+RR*risk
    if tgt<=e: continue
    entries.append((e_i,e,stop,tgt,pH1))
entries.sort(key=lambda x:x[0]); seen=set(); uniq=[]
for en in entries:
    if en[0] in seen: continue
    seen.add(en[0]); uniq.append(en)
entries=uniq
def stats(tr):
    R=np.array([r for _,r in tr]); ts=[t for t,_ in tr]; yr=np.array([t.year for t in ts])
    span=(ts[-1]-ts[0]).days/365.25; pf=R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.99
    cum=np.cumsum(R); dd=(np.maximum.accumulate(cum)-cum).max(); yrs=np.unique(yr); half=yrs[len(yrs)//2]
    return dict(N=len(R),npy=len(R)/span,win=(R>0).mean()*100,pf=pf,meanR=R.mean(),
                IS=R[yr<half].mean(),OOS=R[yr>=half].mean(),retdd=R.sum()/dd if dd>0 else np.inf)
def evaluate(limfn):
    """limfn(e,stop,H1)->limit price (or None=market at e)."""
    busy=-1; tr=[]; miss=0
    for (i,e,stop,tgt,H1) in entries:
        if i<=busy: continue
        lim=limfn(e,stop,H1)
        if lim is None:   # market
            risk=e-stop; reward=tgt-e; exit_j=min(i+FWD,len(c)-1); R=None
            for j in range(i+1,min(i+1+FWD,len(c))):
                if l[j]<=stop: R=-1.0; exit_j=j; break
                if h[j]>=tgt: R=reward/risk; exit_j=j; break
            if R is None: R=(c[exit_j]-e)/risk
            R-=SP/risk; tr.append((d.index[i],R)); busy=exit_j; continue
        if lim<=stop or lim>=e: miss+=1; continue   # invalid anchor
        fill_j=None
        for j in range(i+1,min(i+1+FWD,len(c))):
            if h[j]>=tgt: break
            if l[j]<=lim: fill_j=j; break
        if fill_j is None: miss+=1; continue
        risk=lim-stop; reward=tgt-lim
        if l[fill_j]<=stop: R=-1.0; exit_j=fill_j
        else:
            exit_j=min(fill_j+FWD,len(c)-1); R=None
            for j in range(fill_j+1,min(fill_j+1+FWD,len(c))):
                if l[j]<=stop: R=-1.0; exit_j=j; break
                if h[j]>=tgt: R=reward/risk; exit_j=j; break
            if R is None: R=(c[exit_j]-lim)/risk
        R-=SP/risk; tr.append((d.index[fill_j],R)); busy=exit_j
    return tr,miss
# per-trade retracement-to-H1 depth distribution (data-driven)
depths=[(e-H1)/(e-stop) for (_,e,stop,_,H1) in entries]
print(f"retrace-to-H1 depth (frac of risk): med={np.median(depths):.2f} "
      f"[25/75={np.percentile(depths,25):.2f}/{np.percentile(depths,75):.2f}]  (param-free, per-trade)")
cfgs={"market":lambda e,s,H:None,
      "H1 retest":lambda e,s,H:H,
      "L2-H1 mid":lambda e,s,H:(s+H)/2,
      "frac0.27":lambda e,s,H:e-0.27*(e-s),
      "frac0.5":lambda e,s,H:e-0.5*(e-s)}
print(f"\n  {'anchor':<11}{'N':>5}{'miss%':>7}{'win':>6}{'PF':>7}{'meanR':>8}{'IS/OOS':>13}{'ret/DD':>8}")
res={}
for nm,fn in cfgs.items():
    tr,miss=evaluate(fn); res[nm]=tr; s=stats(tr); mp=miss/(miss+len(tr))*100 if miss+len(tr) else 0
    print(f"  {nm:<11}{s['N']:>5}{mp:>6.0f}%{s['win']:>5.0f}%{s['pf']:>7.2f}{s['meanR']:>+8.3f}{s['IS']:>+6.2f}/{s['OOS']:>+.2f}{s['retdd']:>8.2f}")
from research.edge_harness import audit
print("\n-- audit: does the PARAM-FREE H1-retest have a cleaner overfit profile? --")
audit({k:[(t,r) for t,r in res[k]] for k in ("market","H1 retest","L2-H1 mid","frac0.27")}, flagship="H1 retest")
