"""Diagnose WHY the VWAP-pullback long loses (esp recent). Test user's 2 hypotheses:
(A) falling knife = entries with no bounce / steep prior drop; (B) daily-200/150 too coarse for 15m
-> use a FINER trend (1H/4H). Flagship: RSI40/ADX25/dailyUp confirmed, ATR1.5 stop, 5ATR tgt."""
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
# finer trends (no-lookahead: shift the HTF)
h1=d.resample("60min").agg({"close":"last"}).dropna(); h1e=h1["close"].ewm(span=50,adjust=False).mean()
h1up=((h1e>h1e.shift(1)).shift(1)).reindex(df.index,method="ffill").fillna(False).values
h1e15=h1e.shift(1).reindex(df.index,method="ffill").values
h4=d.resample("240min").agg({"close":"last"}).dropna(); h4e=h4["close"].ewm(span=50,adjust=False).mean()
h4up=((h4e>h4e.shift(1)).shift(1)).reindex(df.index,method="ffill").fillna(False).values
n=len(c); span=(df.index[-1]-df.index[0]).days/365.25
cu=np.zeros(n,bool); cu[1:]=(c[:-1]>=vwap[:-1])&(c[1:]<vwap[1:])
setup=cu&(rsi<40)&(adx<25)&dup
# entries with context: prior drop steepness (recent 20-bar high -> reclaim), reclaim candle
E=[]
for k in np.where(setup)[0]:
    for j in range(1,4):
        if k+j>=n: break
        if c[k+j]>vwap[k+j]:
            ei=k+j+1
            if ei<n:
                recenthigh=h[max(0,k-20):k+1].max(); pblow=l[k:k+j+1].min()
                drop=(recenthigh-pblow)/atr[k] if atr[k]>0 else 0     # prior drop in ATR
                E.append((ei,drop,k)); 
            break
        if l[k+j]<l[k]: pass
seen=set(); E=[e for e in E if not (e[0] in seen or seen.add(e[0]))]
def outcome(i,tp=5.0,sl=1.5):
    e=o[i];a=atr[i-1]
    if np.isnan(a) or a<=0: return None
    stop=e-sl*a;tgt=e+tp*a;mfe=0;r=None;xj=min(i+300,n-1)
    for j in range(i,min(i+300,n)):
        mfe=max(mfe,(h[j]-e)/a)
        if l[j]<=stop:r=-1.0;xj=j;break
        if h[j]>=tgt:r=tp/sl;xj=j;break
    if r is None:r=((c[xj]-e)/a)/sl
    return r,mfe,xj-i
res=[]
for (i,drop,k) in E:
    ou=outcome(i)
    if ou is None: continue
    r,mfe,hold=ou; res.append((i,r,mfe,hold,drop,df.index[i].year,h1up[i],h1e15[i]<c[i-1] if not np.isnan(h1e15[i]) else False,h4up[i]))
res=np.array([(r,mfe,hold,drop,yr,int(hu),int(pg),int(h4)) for (_,r,mfe,hold,drop,yr,hu,pg,h4) in res],float)
R=res[:,0];MFE=res[:,1];HOLD=res[:,2];DROP=res[:,3];YR=res[:,4];H1UP=res[:,5];PGT=res[:,6];H4UP=res[:,7]
W=R>0; L=~W
print(f"flagship n={len(R)} win={W.mean()*100:.0f}% meanR={R.mean():+.3f}\n")
print("== ANATOMY winners vs losers ==")
print(f"  MFE(ATR)     win med={np.median(MFE[W]):.2f}  loss med={np.median(MFE[L]):.2f}  (loss low MFE = never bounced = knife)")
print(f"  prior-drop   win med={np.median(DROP[W]):.2f}  loss med={np.median(DROP[L]):.2f}  (loss deeper drop = steeper knife?)")
print(f"  hold(bars)   win med={np.median(HOLD[W]):.0f}   loss med={np.median(HOLD[L]):.0f}")
print(f"  frac MFE<0.5 (pure knife, no bounce):  win={ (MFE[W]<0.5).mean()*100:.0f}%  loss={(MFE[L]<0.5).mean()*100:.0f}%")
print("\n== finer-trend mismatch at entry (daily says UP; is 1H/4H actually up?) ==")
print(f"  1H-EMA rising:  win-rate when 1H-up={R[H1UP==1].mean()>0 and (R[H1UP==1]>0).mean()*100 or (R[H1UP==1]>0).mean()*100:.0f}% (n{int((H1UP==1).sum())})  "
      f"1H-down={ (R[H1UP==0]>0).mean()*100:.0f}% (n{int((H1UP==0).sum())})")
print(f"  price>1H-EMA :  win={ (R[PGT==1]>0).mean()*100:.0f}% (n{int((PGT==1).sum())})  price<1H-EMA={ (R[PGT==0]>0).mean()*100:.0f}% (n{int((PGT==0).sum())})")
print(f"  4H-EMA rising:  win={ (R[H4UP==1]>0).mean()*100:.0f}% (n{int((H4UP==1).sum())})  4H-down={ (R[H4UP==0]>0).mean()*100:.0f}% (n{int((H4UP==0).sum())})")
def perf(mask,name):
    Rm=R[mask]; 
    if len(Rm)<15: print(f"  {name:<28} n={len(Rm)} too few"); return
    pf=Rm[Rm>0].sum()/abs(Rm[Rm<=0].sum()) if (Rm<=0).any() else 9.9; ym=YR[mask]; half=np.median(ym)
    print(f"  {name:<28} n={len(Rm):>4} PF={pf:.2f} win={(Rm>0).mean()*100:>3.0f}% meanR={Rm.mean():+.3f} IS/OOS={Rm[ym<half].mean():+.2f}/{Rm[ym>=half].mean():+.2f}")
print("\n== TEST improvements (gross) ==")
perf(np.ones(len(R),bool),"flagship (all)")
perf(H1UP==1,"+ 1H-EMA rising (finer)")
perf(H4UP==1,"+ 4H-EMA rising")
perf((H1UP==1)&(PGT==1),"+ 1H up & price>1H-EMA")
perf(DROP<=3.0,"+ prior-drop<=3ATR (anti-knife)")
perf(DROP<=2.0,"+ prior-drop<=2ATR (tighter)")
perf((H1UP==1)&(DROP<=3.0),"+ 1H up & drop<=3ATR")
