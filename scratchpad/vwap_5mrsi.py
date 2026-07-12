"""Execution on 15m, but use the 5m RSI (>=60 = fast-TF momentum turned up) at the reclaim, instead
of / in addition to the 15m RSI<40. 5m RSI aligned to each 15m bar's close (no lookahead). Compare:
baseline(15mRSI<40) vs replace(5mRSI>=X) vs add. clpos anti-knife kept. gross + net."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
d0=load_mt5_csv("data/vantage_xauusd_m5.csv")
m5rsi=ta.rsi(d0["close"],14)
r5_15=m5rsi.resample("15min").last()   # 5m RSI at each 15m bar's close (known at bar close, no lookahead)
df=d0.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
o,h,l,c=(df[k].values for k in ("open","high","low","close")); vol=df["volume"].values
p=(h+l+c)/3; day=df.index.normalize()
vwap=(pd.Series(p*vol,index=df.index).groupby(day).cumsum()/pd.Series(vol,index=df.index).groupby(day).cumsum().replace(0,np.nan)).values
rsi15=ta.rsi(df["close"],14).values; atr=ta.atr(df["high"],df["low"],df["close"],14).values
adx=ta.adx(df["high"],df["low"],df["close"],14)["ADX_14"].values
r5=r5_15.reindex(df.index).values   # aligned 5m RSI per 15m bar
dc_=d0["close"].resample("1D").last().dropna(); dsma=dc_.rolling(150).mean()
dup=((dc_>dsma).shift(1)).reindex(df.index,method="ffill").fillna(False).values
n=len(c); span=(df.index[-1]-df.index[0]).days/365.25
cu=np.zeros(n,bool); cu[1:]=(c[:-1]>=vwap[:-1])&(c[1:]<vwap[1:])
def run(use15rsi=True, m5rsi_th=None, clpos_th=0.85, spread=0.0, sslip=0.0):
    setup=cu&(adx<25)&dup
    if use15rsi: setup=setup&(rsi15<40)
    R=[];ts=[];busy=-1
    for k in np.where(setup)[0]:
        for j in range(1,4):
            if k+j>=n: break
            if c[k+j]>vwap[k+j]:
                rj=k+j;ei=rj+1
                if ei>=n or atr[rj]<=0 or np.isnan(atr[rj]): break
                if m5rsi_th is not None and not (r5[rj]>=m5rsi_th): break  # 5m RSI at reclaim bar close
                rng=max(h[rj]-l[rj],1e-9)
                if (c[rj]-l[rj])/rng<clpos_th: break
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
def show(name,r):
    if r is None: print(f"  {name:<34} too few"); return
    print(f"  {name:<34} N={r[0]:>3}({r[1]:>4.1f}/yr) win={r[2]:>3.0f}% PF={r[3]:.2f} meanR={r[4]:+.3f} IS/OOS={r[5]:+.2f}/{r[6]:+.2f}")
print("15m execution. 5m-RSI momentum condition at reclaim (clpos>=0.85 kept). GROSS\n")
show("BASE 15mRSI<40",                run(True,None))
print("-- REPLACE 15mRSI<40 with 5mRSI>=X --")
for th in [50,55,60,65,70]: show(f"  5mRSI>={th} (no 15m RSI)", run(False,th))
print("-- ADD 5mRSI>=X to base (15mRSI<40 kept) --")
for th in [55,60,65]: show(f"  15mRSI<40 & 5mRSI>={th}", run(True,th))
print("\nnet $0.6+slip on the best couple:")
for nm,kw in [("REPLACE 5mRSI>=60",dict(use15rsi=False,m5rsi_th=60)),("ADD 5mRSI>=60",dict(use15rsi=True,m5rsi_th=60)),("BASE",dict(use15rsi=True,m5rsi_th=None))]:
    r=run(**kw,spread=0.6,sslip=0.5); 
    if r: print(f"  {nm:<20} N/yr={r[1]:.1f} net meanR={r[4]:+.3f} PF={r[3]:.2f}")
