"""RE-DO in the correct bounce-family order: STEP1 bounce-rate (does it revert vs coin-flip),
STEP2 excursion (how far the reversion travels) -- NO fixed RR/exit baked in. gold 15m VWAP-reversal
signal. reversion direction = long on down-cross+RSI<40 / short on up-cross+RSI>60. GROSS. No-lookahead."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
d = load_mt5_csv("data/vantage_xauusd_m5.csv")
vc="volume"
df = d.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last",vc:"sum"}).dropna()
o,h,l,c=(df[k].values for k in ("open","high","low","close")); vol=df[vc].values
hlc3=(h+l+c)/3; day=df.index.normalize()
vwap=(pd.Series(hlc3*vol,index=df.index).groupby(day).cumsum()/pd.Series(vol,index=df.index).groupby(day).cumsum().replace(0,np.nan)).values
rsi=ta.rsi(df["close"],14).values; atr=ta.atr(df["high"],df["low"],df["close"],14).values
hr=df.index.hour.values; n=len(c); span=(df.index[-1]-df.index[0]).days/365.25
cu=np.zeros(n,bool);co=np.zeros(n,bool)
cu[1:]=(c[:-1]>=vwap[:-1])&(c[1:]<vwap[1:]); co[1:]=(c[:-1]<=vwap[:-1])&(c[1:]>vwap[1:])

def excursion(sigmask, s_dir, name, K=24):
    """s_dir=+1 long reversion / -1 short. measure fav MFE & adv MAE over K bars, NO stop."""
    idx=np.where(sigmask)[0]; idx=idx[idx+1<n]
    fav=[];adv=[];bwin=[]  # bwin: +1ATR fav before -1ATR adv
    for i in idx:
        e=o[i+1]; a=atr[i]
        if np.isnan(a) or a<=0: continue
        end=min(i+1+K,n); mfe=0;mae=0;res=0
        for j in range(i+1,end):
            f=s_dir*(h[j]-e)/a if s_dir>0 else s_dir*(l[j]-e)/a   # favorable extreme
            ad=s_dir*(l[j]-e)/a if s_dir>0 else s_dir*(h[j]-e)/a  # adverse extreme (negative)
            mfe=max(mfe,f); mae=min(mae,ad)
            if res==0:
                if f>=1.0 and mfe>= -mae: res=1
                if -ad>=1.0 and (-mae)>mfe: res=-1
        # symmetric barrier properly: reach +1ATR fav before -1ATR adv
        r=0
        for j in range(i+1,end):
            hi=(h[j]-e)/a; lo=(e-l[j])/a
            if s_dir>0:
                if hi>=1.0: r=1;break
                if lo>=1.0: r=-1;break
            else:
                if lo>=1.0: r=1;break
                if hi>=1.0: r=-1;break
        fav.append(mfe); adv.append(-mae); bwin.append(1 if r==1 else (0 if r==-1 else np.nan))
    fav=np.array(fav); adv=np.array(adv); bw=np.array([x for x in bwin if not np.isnan(x)])
    br = bw.mean()*100 if len(bw) else 0
    print(f"  {name:<26} n={len(fav):>4}({len(fav)/span:>3.0f}/yr) | STEP1 bounce%(±1ATR)={br:>4.1f}% reach0.5ATR={ (fav>=0.5).mean()*100:>3.0f}% "
          f"| STEP2 favMFE med={np.median(fav):.2f} p90={np.percentile(fav,90):.2f} std={fav.std():.2f} advMAE med={np.median(adv):.2f}")

print("gold 15m VWAP-reversal | STEP1 bounce-rate -> STEP2 excursion (K=24 bars, NO exit baked in, GROSS)")
print("coin-flip ref: bounce%≈50, favMFE≈advMAE\n")
# baselines
excursion(np.ones(n,bool)&(np.arange(n)>20), +1, "all bars long ref")
print("-- LONG reversion (down-cross + RSI<40) --")
excursion(cu, +1, "VWAP down-cross only")
excursion(cu&(rsi<40), +1, "down-cross + RSI<40")
excursion(cu&(rsi<30), +1, "down-cross + RSI<30 (deeper)")
print("-- SHORT reversion (up-cross + RSI>60) --")
excursion(co, -1, "VWAP up-cross only")
excursion(co&(rsi>60), -1, "up-cross + RSI>60")
excursion(co&(rsi>70), -1, "up-cross + RSI>70 (deeper)")
