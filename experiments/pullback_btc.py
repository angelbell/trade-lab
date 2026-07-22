"""STEP 2: does the pullback-to-fixed-target lever TRANSFER to BTC breakout? (data=H1 only,
no 15m BTC). Same Pattern-B structural breakout + daily-KAMA-rising gate (btc_bo). Test the
LEVER (market vs H1-retest vs frac0.3) at RR2 and RR4, on BTC 1h and 4h. Cost=$15 rt (harness)."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag, kama_adaptive
AGG={"open":"first","high":"max","low":"min","close":"last"}
BO,FWD,SP=20,500,15.0
base=load_mt5_csv("data/vantage_btcusd_h1.csv")
def build(df,RR):
    h,l,c=df["high"].values,df["low"].values,df["close"].values
    a=ta.atr(df["high"],df["low"],df["close"],14).values
    es=df["close"].ewm(span=80,adjust=False).mean().values
    dck=df["close"].resample("1D").last().dropna(); kmg=kama_adaptive(dck,14)
    kreg=((kmg>kmg.shift(1)).shift(1)).reindex(df.index,method="ffill").fillna(False).values
    sw=swings_zigzag(h,l,a,2.0)
    def fb(level,after):
        for j in range(after,min(after+BO,len(c))):
            if c[j]>level: return j
        return None
    E=[]
    for t in range(2,len(sw)):
        (cL2,iL2,pL2,kL2),(cH1,iH1,pH1,kH1),(cL0,iL0,pL0,kL0)=sw[t],sw[t-1],sw[t-2]
        if not(kL2==-1 and kH1==+1 and kL0==-1): continue
        if pL2<=pL0 or pH1-pL0<=0: continue
        if es is not None and not np.isnan(es[cL2]) and pH1<es[cL2]: continue
        e_i=fb(pH1,cL2+1)
        if e_i is None: continue
        if not kreg[e_i]: continue
        e=c[e_i]; stop=pL2; risk=e-stop
        if risk<=0: continue
        tgt=e+RR*risk
        E.append((e_i,e,stop,tgt,pH1))
    E.sort(key=lambda x:x[0]); seen=set(); U=[]
    for en in E:
        if en[0] in seen: continue
        seen.add(en[0]); U.append(en)
    return df,U,h,l,c
def stats(tr):
    R=np.array([r for _,r in tr]); ts=[t for t,_ in tr]; yr=np.array([t.year for t in ts])
    span=(ts[-1]-ts[0]).days/365.25; pf=R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.99
    cum=np.cumsum(R); dd=(np.maximum.accumulate(cum)-cum).max(); yrs=np.unique(yr); half=yrs[len(yrs)//2]
    return dict(N=len(R),npy=len(R)/span,win=(R>0).mean()*100,pf=pf,meanR=R.mean(),
                IS=R[yr<half].mean(),OOS=R[yr>=half].mean(),retdd=R.sum()/dd if dd>0 else np.inf)
def evaluate(df,E,h,l,c,limfn):
    busy=-1; tr=[]; miss=0
    for (i,e,stop,tgt,H1) in E:
        if i<=busy: continue
        lim=limfn(e,stop,H1)
        if lim is None:
            risk=e-stop; reward=tgt-e; exit_j=min(i+FWD,len(c)-1); R=None
            for j in range(i+1,min(i+1+FWD,len(c))):
                if l[j]<=stop: R=-1.0; exit_j=j; break
                if h[j]>=tgt: R=reward/risk; exit_j=j; break
            if R is None: R=(c[exit_j]-e)/risk
            R-=SP/risk; tr.append((df.index[i],R)); busy=exit_j; continue
        if lim<=stop or lim>=e: miss+=1; continue
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
        R-=SP/risk; tr.append((df.index[fill_j],R)); busy=exit_j
    return tr,miss
anchors={"market":lambda e,s,H:None,"H1 retest":lambda e,s,H:H,"frac0.3":lambda e,s,H:e-0.3*(e-s)}
for tf,fr in [("1h",None),("4h","240min")]:
    df=base if fr is None else base.resample(fr).agg(AGG).dropna()
    for RR in (2.0,4.0):
        d2,E,h,l,c=build(df,RR)
        if len(E)<20: continue
        print(f"\n### BTC {tf} RR{RR:.0f}  (n_setups={len(E)}) ###")
        print(f"  {'anchor':<11}{'N':>5}{'miss%':>7}{'win':>6}{'PF':>7}{'meanR':>8}{'IS/OOS':>13}{'ret/DD':>8}")
        for nm,fn in anchors.items():
            tr,miss=evaluate(d2,E,h,l,c,fn)
            if len(tr)<12: print(f"  {nm:<11} too few"); continue
            s=stats(tr); mp=miss/(miss+len(tr))*100
            print(f"  {nm:<11}{s['N']:>5}{mp:>6.0f}%{s['win']:>5.0f}%{s['pf']:>7.2f}{s['meanR']:>+8.3f}{s['IS']:>+6.2f}/{s['OOS']:>+.2f}{s['retdd']:>8.2f}")
