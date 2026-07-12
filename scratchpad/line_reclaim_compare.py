"""Does the RECLAIM reaction transfer from VWAP to price-MA lines (SMA/EMA, esp 200)? Same filters
(RSI<40 + ADX<25 + daily-up + clpos>=0.85), swap the reference line for the down-cross+reclaim.
Reaction only (bounce% + excursion). GOLD 15m. VWAP=63% is the bar to match."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
df=load_mt5_csv("data/vantage_xauusd_m5.csv").resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
o,h,l,c=(df[k].values for k in ("open","high","low","close")); vol=df["volume"].values
p=(h+l+c)/3; day=df.index.normalize()
vwap=(pd.Series(p*vol,index=df.index).groupby(day).cumsum()/pd.Series(vol,index=df.index).groupby(day).cumsum().replace(0,np.nan)).values
rsi=ta.rsi(df["close"],14).values; atr=ta.atr(df["high"],df["low"],df["close"],14).values
adx=ta.adx(df["high"],df["low"],df["close"],14)["ADX_14"].values
dc_=df["close"].resample("1D").last().dropna(); dsma=dc_.rolling(150).mean()
dup=((dc_>dsma).shift(1)).reindex(df.index,method="ffill").fillna(False).values
n=len(c); span=(df.index[-1]-df.index[0]).days/365.25; clpos=(c-l)/np.maximum(h-l,1e-9)
lines={"VWAP":vwap, "SMA20":df["close"].rolling(20).mean().values, "SMA50":df["close"].rolling(50).mean().values,
       "SMA100":df["close"].rolling(100).mean().values, "SMA200":df["close"].rolling(200).mean().values,
       "EMA20":df["close"].ewm(span=20,adjust=False).mean().values, "EMA50":df["close"].ewm(span=50,adjust=False).mean().values,
       "EMA200":df["close"].ewm(span=200,adjust=False).mean().values}
def reaction(line, name, K=24):
    cu=np.zeros(n,bool); cu[1:]=(c[:-1]>=line[:-1])&(c[1:]<line[1:])   # down-cross of the line
    setup=cu&(rsi<40)&(adx<25)&dup; E=[]
    for k in np.where(setup)[0]:
        for j in range(1,4):
            if k+j>=n: break
            if c[k+j]>line[k+j]:                # reclaim the line (close back above)
                if clpos[k+j]>=0.85 and k+j+1<n: E.append(k+j+1)
                break
    E=sorted(set(E)); fav=[];adv=[];bw=[]
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
    if len(fav)<15: print(f"  {name:<8} n={len(fav)} few"); return
    print(f"  {name:<8} n={len(fav):>4}({len(fav)/span:>4.1f}/yr) bounce%={bw.mean()*100:>4.1f}% "
          f"favMFE={np.median(fav):.2f} advMAE={np.median(adv):.2f} fav/adv={np.median(fav)/max(np.median(adv),1e-9):.2f}")
print("GOLD: RECLAIM reaction by reference line (down-cross+reclaim + RSI<40/ADX<25/up/clpos>=.85)\n")
for nm,ln in lines.items(): reaction(ln,nm)
print("\n(VWAP=63%/1.98 is the bar. does an MA reclaim match it, or is VWAP special?)")
