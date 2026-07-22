"""Per instrument, VWAP-RECLAIM reaction with progressive selection (proper order). Does any
instrument concentrate a reaction like gold (63%/1.98)? Level: reclaim -> +clpos>=.85 -> +RSI<40
-> +ADX<25&daily-up. Reaction only (bounce% + fav/adv). Find each instrument's OWN lever."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
def prep(df):
    o,h,l,c=(df[k].values for k in ("open","high","low","close")); vol=df["volume"].values
    p=(h+l+c)/3; day=df.index.normalize()
    vwap=(pd.Series(p*vol,index=df.index).groupby(day).cumsum()/pd.Series(vol,index=df.index).groupby(day).cumsum().replace(0,np.nan)).values
    rsi=ta.rsi(df["close"],14).values; atr=ta.atr(df["high"],df["low"],df["close"],14).values
    adx=ta.adx(df["high"],df["low"],df["close"],14)["ADX_14"].values
    dc_=df["close"].resample("1D").last().dropna(); dsma=dc_.rolling(150).mean()
    dup=((dc_>dsma).shift(1)).reindex(df.index,method="ffill").fillna(False).values
    n=len(c); clpos=(c-l)/np.maximum(h-l,1e-9)
    cu=np.zeros(n,bool); cu[1:]=(c[:-1]>=vwap[:-1])&(c[1:]<vwap[1:])
    return o,h,l,c,vwap,rsi,atr,adx,dup,clpos,cu,n,(df.index[-1]-df.index[0]).days/365.25
def entries(o,h,l,c,vwap,rsi,atr,adx,dup,clpos,cu,n, need_clpos,need_rsi,need_regime):
    setup=cu.copy()
    if need_rsi: setup=setup&(rsi<40)
    if need_regime: setup=setup&(adx<25)&dup
    E=[]
    for k in np.where(setup)[0]:
        for j in range(1,4):
            if k+j>=n: break
            if c[k+j]>vwap[k+j]:
                if (not need_clpos or clpos[k+j]>=0.85) and k+j+1<n: E.append(k+j+1)
                break
    return sorted(set(E))
def rx(E,o,h,l,c,atr,n,span,K=24):
    fav=[];adv=[];bw=[]
    for ei in E:
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
    return len(fav)/span, bw.mean()*100 if len(bw) else 0, np.median(fav)/max(np.median(adv),1e-9)
insts=[("GOLD",load_mt5_csv("data/vantage_xauusd_m5.csv").resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()),
       ("USDJPY",load_mt5_csv("data/vantage_usdjpy_m15.csv")),("EURUSD",load_mt5_csv("data/vantage_eurusd_m15.csv")),
       ("GBPUSD",load_mt5_csv("data/vantage_gbpusd_m15.csv")),("XAGUSD",load_mt5_csv("data/vantage_xagusd_m15.csv")),
       ("BTCUSD",load_mt5_csv("data/vantage_btcusd_m15.csv"))]
print("VWAP-reclaim reaction, progressive selection (bounce% / fav-adv). gold full=63.0%/1.98\n")
print(f"  {'inst':<8}{'reclaim':>16}{'+clpos':>16}{'+clpos+RSI40':>16}{'+full(gold)':>18}")
for nm,df in insts:
    P=prep(df); args=P[:12]
    def cell(nc,nr,nreg):
        r=rx(entries(*args,nc,nr,nreg),P[0],P[1],P[2],P[3],P[6],P[11],P[12])
        return f"{r[1]:>4.0f}%/{r[2]:.2f}" if r else "  -"
    print(f"  {nm:<8}{cell(False,False,False):>16}{cell(True,False,False):>16}{cell(True,True,False):>16}{cell(True,True,True):>18}")
print("\n(look for any instrument reaching bounce>58% & fav/adv>1.5 at some selection = a real reaction to concentrate)")
