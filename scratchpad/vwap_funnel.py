"""Frequency FUNNEL: how much each filter cuts N/yr, and can any be loosened to recover freq
without losing the edge? Filters: VWAP down-cross -> RSI<40 -> ADX<25 -> daily-up -> confirmed
reclaim -> clpos>=0.85. PF/meanR at the confirmed stages (ATR1.5 stop / 5ATR tgt, gross)."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
d=load_mt5_csv("data/vantage_xauusd_m5.csv")
df=d.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
o,h,l,c=(df[k].values for k in ("open","high","low","close")); vol=df["volume"].values
p=(h+l+c)/3; day=df.index.normalize()
vwap=(pd.Series(p*vol,index=df.index).groupby(day).cumsum()/pd.Series(vol,index=df.index).groupby(day).cumsum().replace(0,np.nan)).values
rsi=ta.rsi(df["close"],14).values; atr=ta.atr(df["high"],df["low"],df["close"],14).values
adx=ta.adx(df["high"],df["low"],df["close"],14)["ADX_14"].values
dc=d["close"].resample("1D").last().dropna(); dsma=dc.rolling(150).mean()
dup=((dc>dsma).shift(1)).reindex(df.index,method="ffill").fillna(False).values
n=len(c); span=(df.index[-1]-df.index[0]).days/365.25
cu=np.zeros(n,bool); cu[1:]=(c[:-1]>=vwap[:-1])&(c[1:]<vwap[1:])
def npy(mask): return mask.sum()/span
print("FREQUENCY FUNNEL (N/yr surviving each cumulative filter):\n")
s1=cu
s2=s1&(rsi<40)
s3=s2&(adx<25)
s4=s3&dup
print(f"  1. VWAP down-cross            {s1.sum():>6} ({npy(s1):>5.0f}/yr)")
print(f"  2. + RSI<40                   {s2.sum():>6} ({npy(s2):>5.0f}/yr)   <- biggest cut (x{npy(s1)/npy(s2):.0f})")
print(f"  3. + ADX<25                   {s3.sum():>6} ({npy(s3):>5.0f}/yr)")
print(f"  4. + daily-SMA150 uptrend     {s4.sum():>6} ({npy(s4):>5.0f}/yr)")
# confirmed reclaim + clpos (need per-signal walk)
def entries(setupmask, clpos_th=0.0, rsi_th=None):
    if rsi_th is not None: setupmask=cu&(rsi<rsi_th)&(adx<25)&dup
    E=[]
    for k in np.where(setupmask)[0]:
        for j in range(1,4):
            if k+j>=n: break
            if c[k+j]>vwap[k+j]:
                rj=k+j;ei=rj+1
                if ei<n and atr[rj]>0 and not np.isnan(atr[rj]):
                    rng=max(h[rj]-l[rj],1e-9)
                    if (c[rj]-l[rj])/rng>=clpos_th: E.append(ei)
                break
    return sorted(set(E))
def walk(E):
    R=[];ts=[];busy=-1
    for ei in E:
        if ei<=busy or ei>=n: continue
        e=o[ei];a=atr[ei-1]
        if np.isnan(a) or a<=0: continue
        stop=e-1.5*a;tgt=e+5*a;r=None;xj=min(ei+300,n-1)
        for m in range(ei,min(ei+300,n)):
            if l[m]<=stop:r=-1.0;xj=m;break
            if h[m]>=tgt:r=5/1.5;xj=m;break
        if r is None:r=((c[xj]-e)/a)/1.5
        R.append(r/1.0);ts.append(df.index[ei]);busy=xj
    R=np.array(R)
    return R,ts
E5=entries(s4,0.0); R5,_=walk(E5)
E6=entries(s4,0.85); R6,_=walk(E6)
pf=lambda R: R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.9
print(f"  5. + confirmed VWAP reclaim   {len(E5):>6} ({len(E5)/span:>5.1f}/yr)  PF={pf(R5):.2f} meanR={R5.mean():+.3f}")
print(f"  6. + clpos>=0.85 (anti-knife) {len(E6):>6} ({len(E6)/span:>5.1f}/yr)  PF={pf(R6):.2f} meanR={R6.mean():+.3f}")
print("\n== can we recover the BIG cut (RSI<40)? loosen RSI, keep clpos>=0.85 ==")
print(f"  {'RSI<':>6}{'N/yr':>7}{'PF':>7}{'meanR':>9}")
for rt in [40,45,50,55]:
    E=entries(None,0.85,rsi_th=rt); R,_=walk(E)
    if len(R)>=12: print(f"  {rt:>6}{len(E)/span:>7.1f}{pf(R):>7.2f}{R.mean():>+9.3f}")
