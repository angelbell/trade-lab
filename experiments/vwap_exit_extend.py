"""User: it's a trend-follow bounce -> can we EXTEND the exit (let winners run)? Test fixed-target
sweep + ATR trailing + scale-out on the 15m confirmed VWAP-pullback long. Report meanR AND ret/DD
(lumpiness) + IS/OOS + green -- because extending often lifts meanR but wrecks CAGR/DD (the 200MA lesson)."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
d=load_mt5_csv("data/vantage_xauusd_m5.csv")
df=d.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
o,h,l,c=(df[k].values for k in ("open","high","low","close")); vol=df["volume"].values
hlc3=(h+l+c)/3; day=df.index.normalize()
vwap=(pd.Series(hlc3*vol,index=df.index).groupby(day).cumsum()/pd.Series(vol,index=df.index).groupby(day).cumsum().replace(0,np.nan)).values
rsi=ta.rsi(df["close"],14).values; atr=ta.atr(df["high"],df["low"],df["close"],14).values
adx=ta.adx(df["high"],df["low"],df["close"],14)["ADX_14"].values
dc=d["close"].resample("1D").last().dropna(); dsma=dc.rolling(150).mean()
dup=((dc>dsma).shift(1)).reindex(df.index,method="ffill").fillna(False).values
n=len(c); span=(df.index[-1]-df.index[0]).days/365.25
cu=np.zeros(n,bool); cu[1:]=(c[:-1]>=vwap[:-1])&(c[1:]<vwap[1:])
setup=cu&(rsi<40)&(adx<25)&dup
# entries: confirmed 15m VWAP reclaim within 3 bars
entries=[]
for k in np.where(setup)[0]:
    for j in range(1,4):
        if k+j>=n: break
        if c[k+j]>vwap[k+j]:
            if k+j+1<n: entries.append(k+j+1)   # enter at next 15m open
            break
entries=sorted(set(entries))
def stats(R,ts,name):
    R=np.array(R); y=np.array([t.year for t in ts])
    if len(R)<15: print(f"  {name:<26} n={len(R)} too few"); return
    pf=R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.9; half=np.median(y); ys=sorted(set(y))
    green=sum(1 for yy in ys if R[y==yy].sum()>0)
    cum=np.cumsum(R); ddp=(np.maximum.accumulate(cum)-cum).max(); retdd=R.sum()/ddp if ddp>1e-9 else 9.9
    print(f"  {name:<26} n={len(R):>4}({len(R)/span:>3.0f}/yr) PF={pf:.2f} win={(R>0).mean()*100:>3.0f}% "
          f"meanR={R.mean():+.3f} IS/OOS={R[y<half].mean():+.2f}/{R[y>=half].mean():+.2f} grn={green}/{len(ys)} ret/DD={retdd:+.1f}")
def fixed(sl,tp):
    R=[];ts=[];busy=-1
    for i in entries:
        if i<=busy or i>=n: continue
        e=o[i];a=atr[i-1]
        if np.isnan(a) or a<=0: continue
        stop=e-sl*a;tgt=e+tp*a;r=None;xj=min(i+300,n-1)
        for j in range(i,min(i+300,n)):
            if l[j]<=stop:r=-sl;xj=j;break
            if h[j]>=tgt:r=tp;xj=j;break
        if r is None:r=(c[xj]-e)/a
        R.append(r/sl);ts.append(df.index[i]);busy=xj
    return R,ts
def trail(sl,tk,arm):
    """chandelier trail: after price gains 'arm'*ATR, stop=max(highest_high-tk*ATR). let winners run."""
    R=[];ts=[];busy=-1
    for i in entries:
        if i<=busy or i>=n: continue
        e=o[i];a=atr[i-1]
        if np.isnan(a) or a<=0: continue
        stop=e-sl*a; hh=e; armed=False; r=None; xj=min(i+300,n-1)
        for j in range(i,min(i+300,n)):
            hh=max(hh,h[j])
            if (hh-e)/a>=arm: armed=True
            if armed: stop=max(stop, hh-tk*a)
            if l[j]<=stop: r=(stop-e)/a; xj=j; break
        if r is None: r=(c[xj]-e)/a
        R.append(r/sl); ts.append(df.index[i]); busy=xj
    return R,ts
print("15m confirmed VWAP-pullback long | EXIT sweep (stop 1.5ATR). GROSS. can we extend?\n")
print("-- fixed target sweep --")
for tp in [2.5,3.0,4.0,5.0,6.0]:
    R,ts=fixed(1.5,tp); stats(R,ts,f"fixed tgt{tp}")
print("-- ATR trailing (let winners run) --")
for tk,arm in [(2.0,1.0),(2.5,1.0),(3.0,1.5),(2.0,2.0)]:
    R,ts=trail(1.5,tk,arm); stats(R,ts,f"trail k{tk} arm{arm}")
