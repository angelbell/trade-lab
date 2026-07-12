"""Sweep the ADX threshold on the anti-knife clpos>=0.85 VWAP-pullback long. Freq(N/yr) vs PF
trade-off, gross + net $0.6. (ADX<25 is current; lower=tighter/fewer, higher=looser/more.)"""
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
def run(adx_th, spread=0.0, sslip=0.0, clpos_th=0.85):
    setup=cu&(rsi<40)&(adx<adx_th)&dup; R=[];ts=[];busy=-1
    for k in np.where(setup)[0]:
        for j in range(1,4):
            if k+j>=n: break
            if c[k+j]>vwap[k+j]:
                rj=k+j; ei=rj+1
                if ei>=n or atr[rj]<=0 or np.isnan(atr[rj]): break
                rng=max(h[rj]-l[rj],1e-9); clpos=(c[rj]-l[rj])/rng
                if clpos<clpos_th: break
                if ei<=busy: break
                e=o[ei];a=atr[rj];risk=1.5*a;stop=e-risk;tgt=e+5*a;r=None;xj=min(ei+300,n-1)
                for m in range(ei,min(ei+300,n)):
                    if l[m]<=stop:over=stop-l[m];r=-1.0-sslip*over/risk;xj=m;break
                    if h[m]>=tgt:r=(tgt-e)/risk;xj=m;break
                if r is None:r=(c[xj]-e)/risk
                r-=spread/risk;R.append(r);ts.append(df.index[ei]);busy=xj; break
    R=np.array(R);y=np.array([t.year for t in ts])
    if len(R)<12: return None
    pf=R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.9; half=np.median(y)
    return len(R),len(R)/span,(R>0).mean()*100,pf,R.mean(),R[y<half].mean(),R[y>=half].mean()
print("clpos>=0.85 VWAP-pullback long | ADX threshold sweep\n")
print(f"  {'ADX<':>6} | {'N':>4}{'N/yr':>6}{'win':>6}{'PF':>7}{'meanR':>8}  (gross)   ||  net $0.6+slip: {'PF':>5}{'meanR':>8}")
for th in [15,20,25,30,35,100]:
    g=run(th); nt=run(th,0.6,0.5)
    lab="off" if th==100 else str(th)
    if g is None: print(f"  {lab:>6} | too few"); continue
    ns=f"{nt[3]:.2f}{nt[4]:+8.3f}" if nt else "  -"
    print(f"  {lab:>6} | {g[0]:>4}{g[1]:>6.1f}{g[2]:>5.0f}%{g[3]:>7.2f}{g[4]:>+8.3f}  IS/OOS {g[5]:+.2f}/{g[6]:+.2f} || {ns}")
