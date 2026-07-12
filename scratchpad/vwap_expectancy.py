"""1-year account growth for gold BASE (4.2/yr, net) at various risk fractions f. Bootstrap the
annual outcome (Poisson N per year, compound 1+f*r). Report median/std/quantiles + P(double)/P(loss)
-- honest about the huge single-year variance at ~4 trades/yr. Then a 5-leg basket projection."""
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
setup=cu&(rsi<40)&(adx<25)&dup; R=[];busy=-1
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
                if l[m]<=stop:over=stop-l[m];r=-1.0-0.5*over/risk;xj=m;break
                if h[m]>=tgt:r=(tgt-e)/risk;xj=m;break
            if r is None:r=(c[xj]-e)/risk
            r-=0.6/risk;R.append(r);busy=xj; break
R=np.array(R); Npy=len(R)/span
print(f"gold BASE net R series: n={len(R)}, {Npy:.1f}/yr, meanR={R.mean():+.3f}, net R/yr={Npy*R.mean():+.2f}")
print(f"  win={ (R>0).mean()*100:.0f}%  R dist: win≈{R[R>0].mean():+.2f} / loss≈{R[R<=0].mean():+.2f}\n")
rng=np.random.default_rng(0); T=50000
def sim(nlegs, Npy, f, T=T):
    out=[]
    for _ in range(T):
        eq=1.0
        for _leg in range(nlegs):
            k=rng.poisson(Npy)
            if k>0:
                rs=rng.choice(R,k)
                eq*=np.prod(1+f*rs)
        out.append(eq)
    return np.array(out)
print("== GOLD BASE alone (1 leg, ~4.2 trades/yr) — 1yr account multiplier ==")
print(f"  {'f/trade':>8}{'median':>8}{'mean':>8}{'std':>7}{'P(>=2x)':>9}{'P(<1)':>8}{'p10/p90':>13}")
for f in [0.01,0.02,0.05,0.10,0.15]:
    x=sim(1,Npy,f); print(f"  {f*100:>6.0f}%{np.median(x):>8.2f}{x.mean():>8.2f}{x.std():>7.2f}{(x>=2).mean()*100:>8.0f}%{(x<1).mean()*100:>7.0f}%{np.percentile(x,10):>6.2f}/{np.percentile(x,90):<6.2f}")
print("\n== BASKET PROJECTION: 5 independent legs like gold BASE (~21 trades/yr total) ==")
print("  (assumes 5 instruments each matching gold's net R dist, uncorrelated -- PROJECTION, needs data)")
print(f"  {'f/trade':>8}{'median':>8}{'mean':>8}{'std':>7}{'P(>=2x)':>9}{'P(<1)':>8}{'p10/p90':>13}")
for f in [0.01,0.02,0.05,0.10]:
    x=sim(5,Npy,f); print(f"  {f*100:>6.0f}%{np.median(x):>8.2f}{x.mean():>8.2f}{x.std():>7.2f}{(x>=2).mean()*100:>8.0f}%{(x<1).mean()*100:>7.0f}%{np.percentile(x,10):>6.2f}/{np.percentile(x,90):<6.2f}")
