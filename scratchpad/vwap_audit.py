"""Full falsification on the 15m confirmed VWAP-pullback long (ATR1.5 stop + fixed 5ATR tgt).
(1) cost + stop-slip stress (2) BETA NULL vs random long in-uptrend (3) overfit audit over the
target family (4) per-year. R in stop-units (1.5ATR risk ~ 1% account)."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from research.edge_harness import audit
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
def entries_for(rsi_th):
    setup=cu&(rsi<rsi_th)&(adx<25)&dup; E=[]
    for k in np.where(setup)[0]:
        for j in range(1,4):
            if k+j>=n: break
            if c[k+j]>vwap[k+j]:
                if k+j+1<n: E.append(k+j+1)
                break
    return sorted(set(E))
def walk(E, tp=5.0, sl=1.5, spread=0.0, sslip=0.0):
    R=[];ts=[];busy=-1
    for i in E:
        if i<=busy or i>=n: continue
        e=o[i];a=atr[i-1]
        if np.isnan(a) or a<=0: continue
        risk=sl*a; stop=e-risk; tgt=e+tp*a; r=None; xj=min(i+300,n-1)
        for j in range(i,min(i+300,n)):
            if l[j]<=stop: over=stop-l[j]; r=-1.0-sslip*over/risk; xj=j; break
            if h[j]>=tgt: r=(tgt-e)/risk; xj=j; break
        if r is None: r=(c[xj]-e)/risk
        r-=spread/risk; R.append(r); ts.append(df.index[i]); busy=xj
    return R,ts
def m(R): R=np.array(R); return R.mean(),(R>0).mean()*100,(R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.9)
E=entries_for(40)
print(f"flagship: RSI40/ADX25/dailyUp confirmed, ATR1.5 stop, fixed5ATR tgt. n={len(walk(E)[0])}\n")
print("== (1) COST + STOP-SLIP stress (gold 15m realistic spread $0.4-0.8) ==")
print(f"  {'spread$':>8}{'sslip':>7}{'meanR':>9}{'win%':>7}{'PF':>7}")
for sp,ss in [(0.0,0.0),(0.4,0.5),(0.8,0.5),(0.8,1.0),(1.5,1.0)]:
    R,_=walk(E,spread=sp,sslip=ss); mr,w,pf=m(R); print(f"  {sp:>8.1f}{ss:>7.1f}{mr:>+9.3f}{w:>6.0f}%{pf:>7.2f}")
print("\n== (2) BETA NULL: real vs RANDOM long in daily-uptrend (same N, same ATR stop/tgt, gross) ==")
Rr,_=walk(E); realpf=m(Rr)[2]; realmr=np.array(Rr).mean(); N=len(Rr)
valid=np.where(dup & ~np.isnan(atr) & (atr>0))[0]; valid=valid[(valid+1<n)&(valid>0)]
rng=np.random.default_rng(0); pfs=[];mrs=[]
for _ in range(400):
    pick=np.sort(rng.choice(valid,N,replace=False)); R,_=walk(list(pick+1))  # +1: enter next open-ish
    if len(R)>=20: mm=np.array(R); pfs.append(mm[mm>0].sum()/abs(mm[mm<=0].sum()) if (mm<=0).any() else 9.9); mrs.append(mm.mean())
pfs=np.array(pfs);mrs=np.array(mrs)
print(f"  real PF={realpf:.2f} meanR={realmr:+.3f}  vs random-long: PF %ile={(pfs<realpf).mean()*100:.0f}% (med {np.median(pfs):.2f}) "
      f"meanR %ile={(mrs<realmr).mean()*100:.0f}% (med {np.median(mrs):+.3f})")
print("  (>90%ile = real selection beyond long-gold-drift; <70 = just beta)")
print("\n== (3) OVERFIT AUDIT over target family {3,4,5,6} @ spread $0.6 ==")
cfgs={}
for tp in (3.0,4.0,5.0,6.0):
    R,ts=walk(E,tp=tp,spread=0.6,sslip=0.5); cfgs[f"tgt{tp:.0f}"]=list(zip(ts,R))
audit(cfgs, flagship="tgt5")
print("\n== (4) PER-YEAR (flagship, gross) ==")
R,ts=walk(E); R=np.array(R); yr=np.array([t.year for t in ts])
print("  "+" ".join(f"{y}:{R[yr==y].sum():+.0f}({(yr==y).sum()})" for y in sorted(set(yr))))
