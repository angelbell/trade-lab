"""Run the VALIDATED long config (VWAP down-cross + RSI<40 + ADX<25 + daily-up + confirmed reclaim
+ clpos>=0.85 anti-knife, ATR1.5 stop / 5ATR tgt) on 5m execution vs 15m. Everything (RSI/ADX/ATR/
VWAP/reclaim/clpos) native to the execution TF. Freq + PF + meanR, gross + net (same abs spread)."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
d0=load_mt5_csv("data/vantage_xauusd_m5.csv")
dc_=d0["close"].resample("1D").last().dropna(); dsma=dc_.rolling(150).mean()
dupD=(dc_>dsma).shift(1)
def build(rule, clpos_th=0.85, adx_th=25, rsi_th=40, spread=0.0, sslip=0.0, confirm_bars=3):
    df = d0 if rule=="5min" else d0.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    o,h,l,c=(df[k].values for k in ("open","high","low","close")); vol=df["volume"].values
    p=(h+l+c)/3; day=df.index.normalize()
    vwap=(pd.Series(p*vol,index=df.index).groupby(day).cumsum()/pd.Series(vol,index=df.index).groupby(day).cumsum().replace(0,np.nan)).values
    rsi=ta.rsi(df["close"],14).values; atr=ta.atr(df["high"],df["low"],df["close"],14).values
    adx=ta.adx(df["high"],df["low"],df["close"],14)["ADX_14"].values
    dup=dupD.reindex(df.index,method="ffill").fillna(False).values
    n=len(c); span=(df.index[-1]-df.index[0]).days/365.25
    cu=np.zeros(n,bool); cu[1:]=(c[:-1]>=vwap[:-1])&(c[1:]<vwap[1:])
    setup=cu&(rsi<rsi_th)&(adx<adx_th)&dup
    R=[];ts=[];busy=-1
    for k in np.where(setup)[0]:
        for j in range(1,confirm_bars+1):
            if k+j>=n: break
            if c[k+j]>vwap[k+j]:
                rj=k+j;ei=rj+1
                if ei>=n or atr[rj]<=0 or np.isnan(atr[rj]): break
                rng=max(h[rj]-l[rj],1e-9)
                if (c[rj]-l[rj])/rng<clpos_th: break
                if ei<=busy: break
                e=o[ei];a=atr[rj];risk=1.5*a;stop=e-risk;tgt=e+5*a;r=None;xj=min(ei+600,n-1)
                for m in range(ei,min(ei+600,n)):
                    if l[m]<=stop:over=stop-l[m];r=-1.0-sslip*over/risk;xj=m;break
                    if h[m]>=tgt:r=(tgt-e)/risk;xj=m;break
                if r is None:r=(c[xj]-e)/risk
                r-=spread/risk;R.append(r);ts.append(df.index[ei]);busy=xj; break
    R=np.array(R);y=np.array([t.year for t in ts])
    if len(R)<12: return None
    pf=R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.9; half=np.median(y)
    medatr=np.nanmedian(atr)
    return len(R),len(R)/span,(R>0).mean()*100,pf,R.mean(),R[y<half].mean(),R[y>=half].mean(),medatr
print("VALIDATED long config on 5m vs 15m execution (clpos>=0.85 anti-knife)\n")
print(f"  {'TF':>5}{'medATR$':>9}{'N':>5}{'N/yr':>6}{'win':>6}{'PF':>7}{'meanR':>8}{'IS/OOS':>13}")
for rule,lab in [("5min","5m"),("5min","5m(cf9)"),("15min","15m")]:
    cb=9 if "cf9" in lab else 3
    r=build(rule,confirm_bars=cb)
    if r: print(f"  {lab:>5}{r[7]:>9.2f}{r[0]:>5}{r[1]:>6.1f}{r[2]:>5.0f}%{r[3]:>7.2f}{r[4]:>+8.3f}   {r[5]:+.2f}/{r[6]:+.2f}  (gross)")
print("\n  net (spread $0.4 & $0.8 + slip0.5): meanR/PF")
for rule,lab in [("5min","5m"),("15min","15m")]:
    row=f"  {lab:>5}"
    for sp in [0.0,0.4,0.8]:
        r=build(rule,spread=sp,sslip=0.5 if sp>0 else 0.0)
        if r: row+=f" | ${sp}: {r[4]:+.3f}/PF{r[3]:.2f}"
    print(row)
