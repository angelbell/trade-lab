"""MIRROR short: VWAP up-cross + RSI>60 + ADX<25 + daily-DOWNtrend + confirmed reclaim-below-VWAP
+ clpos-LOW (strong bearish reclaim) -> SHORT, ATR1.5 stop / 5ATR tgt. Does the pullback-sell work
in DOWNtrends (regime-orthogonal to the long -> adds N without diluting)? gross + net + combined."""
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
dupB=(dc_>dsma).shift(1); ddnB=(dc_<dsma).shift(1)
dup=dupB.reindex(df.index,method="ffill").fillna(False).values
ddn=ddnB.reindex(df.index,method="ffill").fillna(False).values
n=len(c); span=(df.index[-1]-df.index[0]).days/365.25
cu=np.zeros(n,bool); cu[1:]=(c[:-1]>=vwap[:-1])&(c[1:]<vwap[1:])   # down-cross
co=np.zeros(n,bool); co[1:]=(c[:-1]<=vwap[:-1])&(c[1:]>vwap[1:])   # up-cross
def run(side, spread=0.0, sslip=0.0, clpos_th=0.85):
    if side>0: setup=cu&(rsi<40)&(adx<25)&dup    # long
    else:      setup=co&(rsi>60)&(adx<25)&ddn    # mirror short
    R=[];ts=[];busy=-1
    for k in np.where(setup)[0]:
        for j in range(1,4):
            if k+j>=n: break
            reclaim = (c[k+j]>vwap[k+j]) if side>0 else (c[k+j]<vwap[k+j])
            if reclaim:
                rj=k+j;ei=rj+1
                if ei>=n or atr[rj]<=0 or np.isnan(atr[rj]): break
                rng=max(h[rj]-l[rj],1e-9)
                cp=(c[rj]-l[rj])/rng
                strong = cp>=clpos_th if side>0 else cp<=(1-clpos_th)   # long: close near high / short: near low
                if not strong: break
                if ei<=busy: break
                e=o[ei];a=atr[rj];risk=1.5*a
                stop=e-side*risk; tgt=e+side*5*a; r=None; xj=min(ei+300,n-1)
                for m in range(ei,min(ei+300,n)):
                    if side>0:
                        if l[m]<=stop:over=stop-l[m];r=-1.0-sslip*over/risk;xj=m;break
                        if h[m]>=tgt:r=(tgt-e)/risk;xj=m;break
                    else:
                        if h[m]>=stop:over=h[m]-stop;r=-1.0-sslip*over/risk;xj=m;break
                        if l[m]<=tgt:r=(e-tgt)/risk;xj=m;break
                if r is None:r=side*(c[xj]-e)/risk
                r-=spread/risk;R.append(r);ts.append(df.index[ei]);busy=xj; break
    return R,ts
def rep(R,ts,name):
    R=np.array(R)
    if len(R)<12: print(f"  {name:<22} n={len(R)} too few"); return
    y=np.array([t.year for t in ts]); pf=R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.9; half=np.median(y)
    ys=sorted(set(y)); green=sum(1 for yy in ys if R[y==yy].sum()>0)
    print(f"  {name:<22} n={len(R):>3}({len(R)/span:>4.1f}/yr) win={(R>0).mean()*100:>3.0f}% PF={pf:.2f} meanR={R.mean():+.3f} IS/OOS={R[y<half].mean():+.2f}/{R[y>=half].mean():+.2f} grn={green}/{len(ys)}")
print("MIRROR short (pullback-sell in daily downtrend) vs long. clpos anti-knife. GROSS\n")
Rl,tl=run(1); rep(Rl,tl,"LONG (uptrend)")
Rs,ts=run(-1); rep(Rs,ts,"MIRROR SHORT (downtr)")
print("\ncombined LONG+SHORT (total N):")
Rc=list(zip(tl,Rl))+list(zip(ts,Rs)); Rc.sort()
Rca=np.array([r for _,r in Rc]); tca=[t for t,_ in Rc]
rep(Rca,tca,"LONG+SHORT")
print("\nnet $0.6+slip:")
for s,nm in [(1,"long"),(-1,"short")]:
    R,ts=run(s,0.6,0.5); R=np.array(R)
    if len(R)>=12: print(f"  {nm:<6} N/yr={len(R)/span:.1f} net meanR={R.mean():+.3f} PF={R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.9:.2f}")
