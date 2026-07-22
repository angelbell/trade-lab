"""User: scale out half at VWAP+1sigma, half at +2sigma (VWAP bands). Better or worse than the
trailing / fixed-far-target? Session-anchored VWAP + volume-weighted std bands. Compare exits on
the same 15m confirmed VWAP-pullback long entries. GROSS. Also report what 1sigma is in ATR."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
d=load_mt5_csv("data/vantage_xauusd_m5.csv")
df=d.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
o,h,l,c=(df[k].values for k in ("open","high","low","close")); vol=df["volume"].values
p=(h+l+c)/3; day=df.index.normalize()
cv=pd.Series(vol,index=df.index).groupby(day).cumsum()
cpv=pd.Series(p*vol,index=df.index).groupby(day).cumsum()
cpv2=pd.Series(p*p*vol,index=df.index).groupby(day).cumsum()
vwap=(cpv/cv.replace(0,np.nan)).values
var=(cpv2/cv.replace(0,np.nan)).values - vwap**2
std=np.sqrt(np.clip(var,0,None))
rsi=ta.rsi(df["close"],14).values; atr=ta.atr(df["high"],df["low"],df["close"],14).values
adx=ta.adx(df["high"],df["low"],df["close"],14)["ADX_14"].values
dc=d["close"].resample("1D").last().dropna(); dsma=dc.rolling(150).mean()
dup=((dc>dsma).shift(1)).reindex(df.index,method="ffill").fillna(False).values
n=len(c); span=(df.index[-1]-df.index[0]).days/365.25
cu=np.zeros(n,bool); cu[1:]=(c[:-1]>=vwap[:-1])&(c[1:]<vwap[1:])
setup=cu&(rsi<40)&(adx<25)&dup
entries=[]
for k in np.where(setup)[0]:
    for j in range(1,4):
        if k+j>=n: break
        if c[k+j]>vwap[k+j]:
            if k+j+1<n: entries.append(k+j+1)
            break
entries=sorted(set(entries))
# how big is 1sigma vs ATR at entry?
sig_atr=[std[i-1]/atr[i-1] for i in entries if not np.isnan(std[i-1]) and atr[i-1]>0]
print(f"entries={len(entries)} | 1sigma/ATR at entry: med={np.median(sig_atr):.2f} p25/p75={np.percentile(sig_atr,25):.2f}/{np.percentile(sig_atr,75):.2f} (so +2sigma ~ {2*np.median(sig_atr):.1f} ATR overhead from VWAP)\n")
def stats(R,ts,name):
    R=np.array(R); y=np.array([t.year for t in ts])
    if len(R)<15: print(f"  {name:<30} n={len(R)} too few"); return
    pf=R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.9; half=np.median(y); ys=sorted(set(y))
    green=sum(1 for yy in ys if R[y==yy].sum()>0); cum=np.cumsum(R); ddp=(np.maximum.accumulate(cum)-cum).max()
    print(f"  {name:<30} n={len(R):>4}({len(R)/span:>3.0f}/yr) PF={pf:.2f} win={(R>0).mean()*100:>3.0f}% "
          f"meanR={R.mean():+.3f} IS/OOS={R[y<half].mean():+.2f}/{R[y>=half].mean():+.2f} grn={green}/{len(ys)} ret/DD={R.sum()/ddp if ddp>1e-9 else 9.9:+.1f}")
def band_split(be_after1=True):
    R=[];ts=[]; busy=-1
    for i in entries:
        if i<=busy or i>=n: continue
        e=o[i]; a=atr[i-1]
        if np.isnan(a) or a<=0: continue
        risk=1.5*a; stop=e-risk; t1_hit=False; realized=0.0; r=None; xj=min(i+300,n-1)
        for j in range(i,min(i+300,n)):
            u1=vwap[j]+std[j]; u2=vwap[j]+2*std[j]
            if l[j]<=stop:
                realized += (0.5 if t1_hit else 1.0)*((stop-e)/risk); r=realized; xj=j; break
            if not t1_hit and h[j]>=u1 and u1>e:
                realized += 0.5*((u1-e)/risk); t1_hit=True
                if be_after1: stop=e
            if t1_hit and h[j]>=u2 and u2>e:
                realized += 0.5*((u2-e)/risk); r=realized; xj=j; break
        if r is None: realized += (0.5 if t1_hit else 1.0)*((c[xj]-e)/risk); r=realized
        R.append(r); ts.append(df.index[i]); busy=xj
    return R,ts
def fixed(sl,tp):
    R=[];ts=[];busy=-1
    for i in entries:
        if i<=busy or i>=n: continue
        e=o[i];a=atr[i-1]
        if np.isnan(a) or a<=0: continue
        stop=e-sl*a;tgt=e+tp*a;r=None;xj=min(i+300,n-1)
        for j in range(i,min(i+300,n)):
            if l[j]<=stop:r=-1.0;xj=j;break
            if h[j]>=tgt:r=tp/sl;xj=j;break
        if r is None:r=((c[xj]-e)/a)/sl
        R.append(r);ts.append(df.index[i]);busy=xj
    return R,ts
def trail(sl,tk,arm):
    R=[];ts=[];busy=-1
    for i in entries:
        if i<=busy or i>=n: continue
        e=o[i];a=atr[i-1]
        if np.isnan(a) or a<=0: continue
        stop=e-sl*a;hh=e;armed=False;r=None;xj=min(i+300,n-1)
        for j in range(i,min(i+300,n)):
            hh=max(hh,h[j])
            if (hh-e)/a>=arm: armed=True
            if armed: stop=max(stop,hh-tk*a)
            if l[j]<=stop: r=((stop-e)/a)/sl; xj=j;break
        if r is None: r=((c[xj]-e)/a)/sl
        R.append(r);ts.append(df.index[i]);busy=xj
    return R,ts
print("15m confirmed VWAP-pullback long | EXIT comparison (stop 1.5ATR, GROSS)\n")
R,ts=band_split(True);  stats(R,ts,"VWAP-band split 1s/2s (BE after 1s)")
R,ts=band_split(False); stats(R,ts,"VWAP-band split 1s/2s (no BE)")
R,ts=trail(1.5,3.0,1.5); stats(R,ts,"trail k3.0 arm1.5")
R,ts=fixed(1.5,2.5); stats(R,ts,"fixed tgt2.5")
R,ts=fixed(1.5,5.0); stats(R,ts,"fixed tgt5.0")
