"""EURUSD 2nd-leg gauntlet: cost stress + per-year + beta null + overfit audit + gold correlation
+ combined 2-leg net R/yr. Config = gold's (VWAP reclaim+RSI40+ADX25+daily-up+clpos.85, 1.5/5ATR)."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from research.edge_harness import audit
def build(df, spread=0.0, sslip=0.0, clpos_th=0.85):
    o,h,l,c=(df[k].values for k in ("open","high","low","close")); vol=df["volume"].values
    p=(h+l+c)/3; day=df.index.normalize()
    vwap=(pd.Series(p*vol,index=df.index).groupby(day).cumsum()/pd.Series(vol,index=df.index).groupby(day).cumsum().replace(0,np.nan)).values
    rsi=ta.rsi(df["close"],14).values; atr=ta.atr(df["high"],df["low"],df["close"],14).values
    adx=ta.adx(df["high"],df["low"],df["close"],14)["ADX_14"].values
    dc_=df["close"].resample("1D").last().dropna(); dsma=dc_.rolling(150).mean()
    dup=((dc_>dsma).shift(1)).reindex(df.index,method="ffill").fillna(False).values
    n=len(c); clpos=(c-l)/np.maximum(h-l,1e-9)
    cu=np.zeros(n,bool); cu[1:]=(c[:-1]>=vwap[:-1])&(c[1:]<vwap[1:])
    setup=cu&(rsi<40)&(adx<25)&dup; R=[];ts=[];busy=-1
    for k in np.where(setup)[0]:
        for j in range(1,4):
            if k+j>=n: break
            if c[k+j]>vwap[k+j]:
                rj=k+j;ei=rj+1
                if ei>=n or atr[rj]<=0 or np.isnan(atr[rj]): break
                if clpos[rj]<clpos_th: break
                if ei<=busy: break
                e=o[ei];a=atr[rj];risk=1.5*a;stop=e-risk;tgt=e+5*a;r=None;xj=min(ei+300,n-1)
                for m in range(ei,min(ei+300,n)):
                    if l[m]<=stop:over=stop-l[m];r=-1.0-sslip*over/risk;xj=m;break
                    if h[m]>=tgt:r=(tgt-e)/risk;xj=m;break
                if r is None:r=(c[xj]-e)/risk
                r-=spread/risk;R.append(r);ts.append(df.index[ei]);busy=xj; break
    return list(zip(ts,R))
eur=load_mt5_csv("data/vantage_eurusd_m15.csv")
gold=load_mt5_csv("data/vantage_xauusd_m5.csv").resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
span=(eur.index[-1]-eur.index[0]).days/365.25
def stat(tr):
    R=np.array([r for _,r in tr]); 
    if len(R)<12: return None
    y=np.array([t.year for t in [t for t,_ in tr]]); pf=R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.9; half=np.median(y)
    return len(R),len(R)/span,(R>0).mean()*100,pf,R.mean(),R[y<half].mean(),R[y>=half].mean()
print("=== EURUSD (1) COST STRESS (FX spread in price; ~1-2 pip = 0.0001-0.0002) ===")
print(f"  {'spread':>9}{'N/yr':>6}{'win':>6}{'PF':>7}{'meanR':>9}{'IS/OOS':>13}")
for sp,ss in [(0.0,0.0),(0.0001,0.5),(0.0002,0.5)]:
    s=stat(build(eur,sp,ss))
    if s: print(f"  {sp:>9.4f}{s[1]:>6.1f}{s[2]:>5.0f}%{s[3]:>7.2f}{s[4]:>+9.3f}   {s[5]:+.2f}/{s[6]:+.2f}")
tr_e=build(eur,0.00012,0.5); R=np.array([r for _,r in tr_e]); yr=np.array([t.year for t,_ in tr_e])
print("\n=== (2) PER-YEAR (net spread 1.2pip) ===")
print("  "+" ".join(f"{y}:{R[yr==y].sum():+.0f}({(yr==y).sum()})" for y in sorted(set(yr))))
print("\n=== (3) BETA NULL: real vs random long in EURUSD uptrend (same N, gross) ===")
o,h,l,c=(eur[k].values for k in ("open","high","low","close")); atr=ta.atr(eur["high"],eur["low"],eur["close"],14).values
dc_=eur["close"].resample("1D").last().dropna(); dsma=dc_.rolling(150).mean()
dup=((dc_>dsma).shift(1)).reindex(eur.index,method="ffill").fillna(False).values; n=len(c)
tr_g=build(eur); Rg=np.array([r for _,r in tr_g]); realpf=Rg[Rg>0].sum()/abs(Rg[Rg<=0].sum()); N=len(Rg)
valid=np.where(dup&~np.isnan(atr)&(atr>0))[0]; valid=valid[(valid+1<n)&(valid>0)]; rng=np.random.default_rng(0); pfs=[]
for _ in range(300):
    pick=np.sort(rng.choice(valid,N,replace=False)); Rr=[]
    for i in pick:
        e=o[i+1];a=atr[i]
        if np.isnan(a) or a<=0: continue
        stop=e-1.5*a;tgt=e+5*a;r=None;xj=min(i+1+300,n-1)
        for m in range(i+1,min(i+1+300,n)):
            if l[m]<=stop:r=-1.0;xj=m;break
            if h[m]>=tgt:r=5/1.5;xj=m;break
        if r is None:r=(c[xj]-e)/a/1.5
        Rr.append(r)
    Rr=np.array(Rr)
    if len(Rr)>=15: pfs.append(Rr[Rr>0].sum()/abs(Rr[Rr<=0].sum()) if (Rr<=0).any() else 9.9)
print(f"  real PF={realpf:.2f} vs random-long: %ile={(np.array(pfs)<realpf).mean()*100:.0f}% (med {np.median(pfs):.2f})")
print("\n=== (4) OVERFIT AUDIT (clpos family, net 1.2pip) ===")
cfgs={f"clpos{th}":build(eur,0.00012,0.5,th) for th in (0.7,0.8,0.85,0.9)}
for k,v in cfgs.items(): rr=np.array([x for _,x in v]); print(f"  {k}: n={len(rr)} meanR={rr.mean():+.3f}")
audit(cfgs, flagship="clpos0.85")
print("\n=== (5) GOLD <-> EURUSD annual-R correlation (net) ===")
tr_gold=build(gold,0.6,0.5)
def ann(tr): 
    s=pd.Series([r for _,r in tr],index=pd.to_datetime([t for t,_ in tr])); return s.groupby(s.index.year).sum()
ag=ann(tr_gold); ae=ann(tr_e); al=pd.concat([ag,ae],axis=1).fillna(0); al.columns=["gold","eur"]
print(f"  annual-R corr(gold,EURUSD) = {al['gold'].corr(al['eur']):+.2f}")
print(f"  gold net R/yr={ag.sum()/((gold.index[-1]-gold.index[0]).days/365.25):+.2f}  EURUSD net R/yr={ae.sum()/span:+.2f}  combined={ (ag.sum()/((gold.index[-1]-gold.index[0]).days/365.25))+(ae.sum()/span):+.2f}")
