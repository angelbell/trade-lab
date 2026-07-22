"""Faithful mechanization of the pasted 'XAUUSD 15m VWAP Reversal' strategy. Test AS-WRITTEN first
(gross, cost=0), then strip to the all-signals BASE (VWAP-cross fade only) to see if the filters
concentrate a real edge or a coin-flip. Session-anchored VWAP (daily UTC reset). No-lookahead:
signal on closed bar, enter next open, intrabar stop-first."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
AGG={"open":"first","high":"max","low":"min","close":"last","tickvol":"sum"}
d = load_mt5_csv("data/vantage_xauusd_m5.csv")
volcol = "tickvol" if "tickvol" in d.columns else ("volume" if "volume" in d.columns else None)
print("vol column:", volcol, "| cols:", list(d.columns))
if volcol is None:
    d["tickvol"]=1.0; volcol="tickvol"
agg={k:v for k,v in AGG.items() if k in d.columns or k=="tickvol"}
agg[volcol]="sum"
df = d.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last",volcol:"sum"}).dropna()
o,h,l,c = (df[k].values for k in ("open","high","low","close")); vol=df[volcol].values
hlc3=(h+l+c)/3
# session-anchored VWAP: reset each UTC day
day = df.index.normalize()
cpv = pd.Series(hlc3*vol,index=df.index).groupby(day).cumsum()
cv  = pd.Series(vol,index=df.index).groupby(day).cumsum()
vwap=(cpv/cv.replace(0,np.nan)).values
rsi = ta.rsi(df["close"],14).values
volsma = pd.Series(vol).rolling(20).mean().values
adx = ta.adx(df["high"],df["low"],df["close"],14)["ADX_14"].values
atr = ta.atr(df["high"],df["low"],df["close"],14).values
hr = df.index.hour.values
n=len(c); yr=np.array([t.year for t in df.index]); span=(df.index[-1]-df.index[0]).days/365.25
cu = np.zeros(n,bool); co=np.zeros(n,bool)   # crossunder / crossover of close vs vwap
cu[1:] = (c[:-1]>=vwap[:-1]) & (c[1:]<vwap[1:])
co[1:] = (c[:-1]<=vwap[:-1]) & (c[1:]>vwap[1:])
sess = (hr>=7)&(hr<21)
hv = vol > volsma*1.5
adxok = adx < 25
def walk(longsig, shortsig, sl=1.5, tp=2.0, cost=0.0):
    side=np.zeros(n); side[longsig]=1; side[shortsig]=-1
    idx=np.where(side!=0)[0]; idx=idx[idx+1<n]; busy=-1; R=[];ts=[];sd=[]
    for i in idx:
        if i<=busy: continue
        s=side[i]; e=o[i+1]; a=atr[i]
        if np.isnan(a) or a<=0: continue
        stop=e-s*sl*a; tgt=e+s*tp*a; r=None; xj=min(i+1+300,n-1)
        for j in range(i+1,min(i+1+300,n)):
            if s>0:
                if l[j]<=stop: r=-sl; xj=j;break
                if h[j]>=tgt: r=tp; xj=j;break
            else:
                if h[j]>=stop: r=-sl; xj=j;break
                if l[j]<=tgt: r=tp; xj=j;break
        if r is None: r=s*(c[xj]-e)/a
        r-=cost/a; R.append(r/sl); ts.append(df.index[i]); sd.append(s); busy=xj   # R in stop-units
    R=np.array(R);sd=np.array(sd);y=np.array([t.year for t in ts])
    return R,sd,y
def rep(name,R,sd,y):
    if len(R)<10: print(f"  {name:<30} n={len(R)} (too few)"); return
    pf=R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.9
    half=np.median(y)
    def sub(m):
        Rm=R[m]; return f"n={len(Rm):>4} PF={ (Rm[Rm>0].sum()/abs(Rm[Rm<=0].sum()) if (Rm<=0).any() else 9.9):.2f} meanR={Rm.mean():+.2f}"
    print(f"  {name:<30} n={len(R):>4}({len(R)/span:>3.0f}/yr) PF={pf:.2f} win={ (R>0).mean()*100:>3.0f}% meanR={R.mean():+.3f} "
          f"IS/OOS={R[y<half].mean():+.2f}/{R[y>=half].mean():+.2f}")
    if (sd>0).any() and (sd<0).any():
        print(f"      long-fade:  {sub(sd>0)}   |  short-fade: {sub(sd<0)}")
print("\n== AS-WRITTEN (gross, cost=0) ==")
ls = co & (rsi>60) & hv & sess & adxok   # shortSignal
ll = cu & (rsi<40) & hv & sess & adxok   # longSignal
R,sd,y=walk(ll,ls); rep("as-written RR2/1.5",R,sd,y)
print("\n== STRIP to BASE (does any edge exist before filters?) ==")
R,sd,y=walk(cu,co); rep("VWAP-cross fade ONLY",R,sd,y)
R,sd,y=walk(cu&(rsi<40),co&(rsi>60)); rep("+RSI40/60",R,sd,y)
R,sd,y=walk(cu&(rsi<40)&hv,co&(rsi>60)&hv); rep("+RSI+highvol",R,sd,y)
R,sd,y=walk(cu&(rsi<40)&sess,co&(rsi>60)&sess); rep("+RSI+session",R,sd,y)
print("\n== beta check: TREND-FOLLOW the same VWAP cross (opposite direction) ==")
R,sd,y=walk(co,cu); rep("VWAP-cross MOMENTUM (long on up-cross)",R,sd,y)
