"""Structural stop (below the reversal/confirmation candle low, or the pullback low) vs fixed
1.5-ATR stop, on the 15m confirmed VWAP-pullback long. How does the risk distance differ, and
which performs better? Target: RR-based (keep RR~3.3) and fixed 5-ATR. GROSS."""
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
setup=cu&(rsi<40)&(adx<25)&dup
# entries: (entry_i, reversal_candle_low, pullback_low)
E=[]
for k in np.where(setup)[0]:
    for j in range(1,4):
        if k+j>=n: break
        if c[k+j]>vwap[k+j]:
            ei=k+j+1
            if ei<n:
                revlow=l[k+j]                    # confirmation/reversal candle low
                pblow=l[k:k+j+1].min()           # dip low from down-cross to reclaim
                E.append((ei,revlow,pblow))
            break
seen=set(); E=[e for e in E if not (e[0] in seen or seen.add(e[0]))]
def stats(R,ts,rd,name):
    R=np.array(R); y=np.array([t.year for t in ts]); rd=np.array(rd)
    if len(R)<15: print(f"  {name:<32} n={len(R)} too few"); return
    pf=R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.9; half=np.median(y); ys=sorted(set(y))
    green=sum(1 for yy in ys if R[y==yy].sum()>0); cum=np.cumsum(R); ddp=(np.maximum.accumulate(cum)-cum).max()
    print(f"  {name:<32} n={len(R):>4} risk(ATR)med={np.median(rd):.2f}[{np.percentile(rd,25):.2f}-{np.percentile(rd,75):.2f}] "
          f"PF={pf:.2f} win={(R>0).mean()*100:>3.0f}% meanR={R.mean():+.3f} IS/OOS={R[y<half].mean():+.2f}/{R[y>=half].mean():+.2f} ret/DD={R.sum()/ddp if ddp>1e-9 else 9.9:+.1f}")
def run(stopfn, tgtmode, name, rr=3.33, tp_atr=5.0, buf=0.1):
    R=[];ts=[];rd=[];busy=-1
    for (i,revlow,pblow) in E:
        if i<=busy or i>=n: continue
        e=o[i]; a=atr[i-1]
        if np.isnan(a) or a<=0: continue
        stop=stopfn(e,a,revlow,pblow,buf)
        risk=e-stop
        if risk<=0.05*a: continue                # skip degenerate (stop above/at entry)
        tgt = e+rr*risk if tgtmode=="rr" else e+tp_atr*a
        r=None; xj=min(i+300,n-1)
        for j in range(i,min(i+300,n)):
            if l[j]<=stop: r=-1.0; xj=j; break
            if h[j]>=tgt: r=(tgt-e)/risk; xj=j; break
        if r is None: r=(c[xj]-e)/risk
        R.append(r); ts.append(df.index[i]); rd.append(risk/a); busy=xj
    return R,ts,rd
print(f"entries={len(E)} | 15m confirmed VWAP-pullback long. STRUCT stop vs ATR stop (GROSS)\n")
# ATR baseline
R,ts,rd=run(lambda e,a,rl,pl,b: e-1.5*a, "atr5", "ATR1.5 stop + fixed5ATR tgt"); stats(R,ts,rd,"ATR1.5 + fixed5ATR")
R,ts,rd=run(lambda e,a,rl,pl,b: e-1.5*a, "rr",   "ATR1.5 stop + RR3.3 tgt");     stats(R,ts,rd,"ATR1.5 + RR3.3")
print("-- structural: below reversal-candle low --")
R,ts,rd=run(lambda e,a,rl,pl,b: rl-b*a, "rr",   "revlow + RR3.3");  stats(R,ts,rd,"revlow-0.1ATR + RR3.3")
R,ts,rd=run(lambda e,a,rl,pl,b: rl-b*a, "atr5", "revlow + fixed5"); stats(R,ts,rd,"revlow-0.1ATR + fixed5ATR")
print("-- structural: below pullback (dip) low --")
R,ts,rd=run(lambda e,a,rl,pl,b: pl-b*a, "rr",   "pblow + RR3.3");   stats(R,ts,rd,"pullback-low-0.1ATR + RR3.3")
R,ts,rd=run(lambda e,a,rl,pl,b: pl-b*a, "atr5", "pblow + fixed5");  stats(R,ts,rd,"pullback-low-0.1ATR + fixed5ATR")
