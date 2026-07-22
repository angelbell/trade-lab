"""Transfer the validated gold VWAP-pullback + anti-knife LONG config to other instruments (15m).
Config: VWAP down-cross + RSI<40 + ADX<25 + daily-SMA150 uptrend + confirmed reclaim + clpos>=0.85,
ATR1.5 stop / 5ATR tgt. GROSS first (does the edge EXIST per instrument?). Same config, no tuning."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
def build(df):
    o,h,l,c=(df[k].values for k in ("open","high","low","close")); vol=df["volume"].values
    p=(h+l+c)/3; day=df.index.normalize()
    cvv=pd.Series(vol,index=df.index).groupby(day).cumsum()
    vwap=(pd.Series(p*vol,index=df.index).groupby(day).cumsum()/cvv.replace(0,np.nan)).values
    rsi=ta.rsi(df["close"],14).values; atr=ta.atr(df["high"],df["low"],df["close"],14).values
    adx=ta.adx(df["high"],df["low"],df["close"],14)["ADX_14"].values
    dc_=df["close"].resample("1D").last().dropna(); dsma=dc_.rolling(150).mean()
    dup=((dc_>dsma).shift(1)).reindex(df.index,method="ffill").fillna(False).values
    return o,h,l,c,vwap,rsi,atr,adx,dup
def run(df, spread=0.0, sslip=0.0):
    o,h,l,c,vwap,rsi,atr,adx,dup=build(df); n=len(c); span=(df.index[-1]-df.index[0]).days/365.25
    cu=np.zeros(n,bool); cu[1:]=(c[:-1]>=vwap[:-1])&(c[1:]<vwap[1:])
    setup=cu&(rsi<40)&(adx<25)&dup; R=[];ts=[];busy=-1
    for k in np.where(setup)[0]:
        for j in range(1,4):
            if k+j>=n: break
            if c[k+j]>vwap[k+j]:
                rj=k+j;ei=rj+1
                if ei>=n or atr[rj]<=0 or np.isnan(atr[rj]): break
                rng=max(h[rj]-l[rj],1e-9)
                if (c[rj]-l[rj])/rng<0.85: break
                if ei<=busy: break
                e=o[ei];a=atr[rj];risk=1.5*a;stop=e-risk;tgt=e+5*a;r=None;xj=min(ei+300,n-1)
                for m in range(ei,min(ei+300,n)):
                    if l[m]<=stop:over=stop-l[m];r=-1.0-sslip*over/risk;xj=m;break
                    if h[m]>=tgt:r=(tgt-e)/risk;xj=m;break
                if r is None:r=(c[xj]-e)/risk
                r-=spread/risk;R.append(r);ts.append(df.index[ei]);busy=xj; break
    R=np.array(R)
    if len(R)<12: return None
    y=np.array([t.year for t in ts]); pf=R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.9; half=np.median(y)
    ys=sorted(set(y)); green=sum(1 for yy in ys if R[y==yy].sum()>0)
    return len(R),len(R)/span,(R>0).mean()*100,pf,R.mean(),R[y<half].mean(),R[y>=half].mean(),green,len(ys),np.nanmedian(atr)
insts=[("GOLD",load_mt5_csv("data/vantage_xauusd_m5.csv").resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()),
       ("USDJPY",load_mt5_csv("data/vantage_usdjpy_m15.csv")),("EURUSD",load_mt5_csv("data/vantage_eurusd_m15.csv")),
       ("GBPUSD",load_mt5_csv("data/vantage_gbpusd_m15.csv")),("XAGUSD",load_mt5_csv("data/vantage_xagusd_m15.csv")),
       ("BTCUSD",load_mt5_csv("data/vantage_btcusd_m15.csv"))]
print("VWAP-pullback + anti-knife LONG, SAME config, per instrument (GROSS)\n")
print(f"  {'inst':<8}{'N':>5}{'N/yr':>6}{'win':>6}{'PF':>7}{'meanR':>8}{'IS/OOS':>13}{'green':>7}{'medATR':>9}")
for nm,df in insts:
    r=run(df)
    if r is None: print(f"  {nm:<8}  too few"); continue
    print(f"  {nm:<8}{r[0]:>5}{r[1]:>6.1f}{r[2]:>5.0f}%{r[3]:>7.2f}{r[4]:>+8.3f}   {r[5]:+.2f}/{r[6]:+.2f}   {r[7]}/{r[8]:<4}{r[9]:>9.4f}")
print("\n(edge EXISTS where PF>~1.3, meanR>+0.3, IS&OOS both+, green majority)")
