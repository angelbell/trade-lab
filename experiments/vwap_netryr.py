"""User: '4x freq / half PF is fine.' The right metric = NET R/yr (= N/yr * net-meanR) = compounding
fuel. Cost is ~fixed per trade, edge is cost-marginal -> loosening dilutes net. Find the config that
maximizes NET R/yr. spread $0.6 + slip0.5."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
d0=load_mt5_csv("data/vantage_xauusd_m5.csv")
df=d0.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
o,h,l,c=(df[k].values for k in ("open","high","low","close")); vol=df["volume"].values
p=(h+l+c)/3; day=df.index.normalize()
vwap=(pd.Series(p*vol,index=df.index).groupby(day).cumsum()/pd.Series(vol,index=df.index).groupby(day).cumsum().replace(0,np.nan)).values
rsi=ta.rsi(df["close"],14).values; atr=ta.atr(df["high"],df["low"],df["close"],14).values
adx=ta.adx(df["high"],df["low"],df["close"],14)["ADX_14"].values
dc_=d0["close"].resample("1D").last().dropna(); dsma=dc_.rolling(150).mean()
dup=((dc_>dsma).shift(1)).reindex(df.index,method="ffill").fillna(False).values
n=len(c); span=(df.index[-1]-df.index[0]).days/365.25
cu=np.zeros(n,bool); cu[1:]=(c[:-1]>=vwap[:-1])&(c[1:]<vwap[1:])
def run(rsi_th,clpos_th,adx_th,spread,sslip):
    setup=cu&(rsi<rsi_th)&(adx<adx_th)&dup; R=[];ts=[];busy=-1
    for k in np.where(setup)[0]:
        for j in range(1,4):
            if k+j>=n: break
            if c[k+j]>vwap[k+j]:
                rj=k+j;ei=rj+1
                if ei>=n or atr[rj]<=0 or np.isnan(atr[rj]): break
                rng=max(h[rj]-l[rj],1e-9)
                if (c[rj]-l[rj])/rng<clpos_th: break
                if ei<=busy: break
                e=o[ei];a=atr[rj];risk=1.5*a;stop=e-risk;tgt=e+5*a;r=None;xj=min(ei+300,n-1)
                for m in range(ei,min(ei+300,n)):
                    if l[m]<=stop:over=stop-l[m];r=-1.0-sslip*over/risk;xj=m;break
                    if h[m]>=tgt:r=(tgt-e)/risk;xj=m;break
                if r is None:r=(c[xj]-e)/risk
                r-=spread/risk;R.append(r);ts.append(df.index[ei]);busy=xj; break
    R=np.array(R)
    if len(R)<12: return None
    pf=R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.9
    y=np.array([t.year for t in ts]); half=np.median(y)
    return len(R)/span, R.mean(), pf, len(R)/span*R.mean(), R[y<half].mean(), R[y>=half].mean()
cfgs=[("RSI40 clpos.85 ADX25 (BASE)",40,0.85,25),("RSI42 clpos.85 ADX25",42,0.85,25),
      ("RSI45 clpos.85 ADX25",45,0.85,25),("RSI50 clpos.85 ADX25",50,0.85,25),
      ("RSI40 clpos.80 ADX25",40,0.80,25),("RSI40 clpos.70 ADX25",40,0.70,25),
      ("RSI40 clpos.85 ADX30",40,0.85,30),("RSI40 clpos.85 ADX100",40,0.85,100),
      ("RSI45 clpos.80 ADX30",45,0.80,30),("RSI50 clpos.70 ADX30",50,0.70,30),
      ("RSI45 clpos.85 ADX100",45,0.85,100),("RSI50 clpos.80 ADX100",50,0.80,100)]
print("NET metrics @ spread $0.6+slip0.5. KEY = NET R/yr (compounding fuel). BASE net R/yr:\n")
print(f"  {'config':<30}{'N/yr':>6}{'netMeanR':>9}{'netPF':>7}{'NET R/yr':>9}{'IS/OOS':>13}")
rows=[]
for nm,rt,cp,ax in cfgs:
    r=run(rt,cp,ax,0.6,0.5)
    if r: rows.append((nm,*r))
for nm,npy,mr,pf,ryr,is_,oos in sorted(rows,key=lambda x:-x[4]):
    star=" <-- BASE" if "BASE" in nm else ""
    print(f"  {nm:<30}{npy:>6.1f}{mr:>+9.3f}{pf:>7.2f}{ryr:>+9.2f}{is_:>+7.2f}/{oos:+.2f}{star}")
print("\n(higher NET R/yr = faster compounding. does loosening for 4x freq raise or lower it?)")
