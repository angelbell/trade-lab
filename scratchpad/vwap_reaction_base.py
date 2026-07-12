"""FIRST: does each instrument REACT to VWAP at all? Raw VWAP-pullback reaction, NO gold filters.
STEP1 bounce-rate + excursion for: all-bars(ref) / VWAP down-cross / down-cross+reclaim. K=24."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
def prep(df):
    o,h,l,c=(df[k].values for k in ("open","high","low","close")); vol=df["volume"].values
    p=(h+l+c)/3; day=df.index.normalize()
    vwap=(pd.Series(p*vol,index=df.index).groupby(day).cumsum()/pd.Series(vol,index=df.index).groupby(day).cumsum().replace(0,np.nan)).values
    atr=ta.atr(df["high"],df["low"],df["close"],14).values; n=len(c)
    cu=np.zeros(n,bool); cu[1:]=(c[:-1]>=vwap[:-1])&(c[1:]<vwap[1:])
    return o,h,l,c,vwap,atr,cu,n,(df.index[-1]-df.index[0]).days/365.25
def react(mask, o,h,l,c,atr,n,span, need_reclaim=False, vwap=None, K=24):
    idx=np.where(mask)[0]; fav=[];adv=[];bw=[]
    for i in idx:
        ei=i+1
        if need_reclaim:  # wait for close back above VWAP within 3 bars (raw structure, no other filter)
            ri=None
            for j in range(i+1,min(i+4,n)):
                if c[j]>vwap[j]: ri=j+1; break
            if ri is None or ri>=n: continue
            ei=ri
        if ei>=n or np.isnan(atr[ei-1]) or atr[ei-1]<=0: continue
        e=o[ei];a=atr[ei-1];end=min(ei+K,n);mfe=0;mae=0;r=0
        for j in range(ei,end):
            hi=(h[j]-e)/a;lo=(e-l[j])/a;mfe=max(mfe,hi);mae=max(mae,lo)
            if r==0:
                if hi>=1.0:r=1
                elif lo>=1.0:r=-1
        fav.append(mfe);adv.append(mae)
        if r!=0:bw.append(1 if r==1 else 0)
    fav=np.array(fav);adv=np.array(adv);bw=np.array(bw)
    if len(fav)<15: return None
    return len(fav),len(fav)/span,bw.mean()*100 if len(bw) else 0,np.median(fav),np.median(adv),np.median(fav)/max(np.median(adv),1e-9)
insts=[("GOLD",load_mt5_csv("data/vantage_xauusd_m5.csv").resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()),
       ("USDJPY",load_mt5_csv("data/vantage_usdjpy_m15.csv")),("EURUSD",load_mt5_csv("data/vantage_eurusd_m15.csv")),
       ("GBPUSD",load_mt5_csv("data/vantage_gbpusd_m15.csv")),("XAGUSD",load_mt5_csv("data/vantage_xagusd_m15.csv")),
       ("BTCUSD",load_mt5_csv("data/vantage_btcusd_m15.csv"))]
print("Does each instrument REACT to VWAP? (raw, no RSI/ADX/clpos). bounce%(±1ATR) + fav/adv excursion")
print("coin-flip ref = bounce~50%, fav/adv~1.0. REACTS if down-cross bounce>52% or fav/adv>1.05\n")
print(f"  {'inst':<8}| {'all-bars':>18} | {'VWAP down-cross':>22} | {'down-cross+reclaim':>24}")
print(f"  {'':8}| {'bnc% fav/adv':>18} | {'N/yr bnc% fav/adv':>22} | {'N/yr bnc% fav/adv':>24}")
for nm,df in insts:
    o,h,l,c,vwap,atr,cu,n,span=prep(df)
    allb=np.zeros(n,bool); allb[20:]=True
    a=react(allb,o,h,l,c,atr,n,span)
    b=react(cu,o,h,l,c,atr,n,span)
    d=react(cu,o,h,l,c,atr,n,span,need_reclaim=True,vwap=vwap)
    fa=lambda r: f"{r[2]:>4.1f}% {r[5]:>4.2f}" if r else "   -"
    fb=lambda r: f"{r[1]:>4.0f} {r[2]:>4.1f}% {r[5]:>4.2f}" if r else "   -"
    print(f"  {nm:<8}| {fa(a):>18} | {fb(b):>22} | {fb(d):>24}")
print("\n(fav/adv>1 = favorable reaction. compare each vs its own all-bars & vs gold)")
