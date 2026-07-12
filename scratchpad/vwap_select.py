"""STEP3 selectability for the long-fade (VWAP down-cross + RSI<40). Which added condition lifts
bounce% and/or excursion above the base (54.5% / favMFE 2.32)? GROSS, no exit baked in, K=24."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
d=load_mt5_csv("data/vantage_xauusd_m5.csv")
df=d.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
o,h,l,c=(df[k].values for k in ("open","high","low","close")); vol=df["volume"].values
hlc3=(h+l+c)/3; day=df.index.normalize()
vwap=(pd.Series(hlc3*vol,index=df.index).groupby(day).cumsum()/pd.Series(vol,index=df.index).groupby(day).cumsum().replace(0,np.nan)).values
rsi=ta.rsi(df["close"],14).values; atr=ta.atr(df["high"],df["low"],df["close"],14).values
adx=ta.adx(df["high"],df["low"],df["close"],14)["ADX_14"].values
volsma=pd.Series(vol).rolling(20).mean().values; hr=df.index.hour.values
dc=d["close"].resample("1D").last().dropna(); dsma=dc.rolling(150).mean()
dup=((dc>dsma).shift(1)).reindex(df.index,method="ffill").fillna(False).values
n=len(c); span=(df.index[-1]-df.index[0]).days/365.25
cu=np.zeros(n,bool); cu[1:]=(c[:-1]>=vwap[:-1])&(c[1:]<vwap[1:])
dip=(vwap-c)/atr   # how far below VWAP at signal (ATR)
def exc(mask,name,K=24):
    idx=np.where(mask)[0]; idx=idx[idx+1<n]; fav=[];adv=[];bw=[]
    for i in idx:
        e=o[i+1];a=atr[i]
        if np.isnan(a) or a<=0: continue
        end=min(i+1+K,n); mfe=0;mae=0;r=0
        for j in range(i+1,end):
            hi=(h[j]-e)/a; lo=(e-l[j])/a; mfe=max(mfe,hi); mae=max(mae,lo)
            if r==0:
                if hi>=1.0: r=1
                elif lo>=1.0: r=-1
        fav.append(mfe);adv.append(mae)
        if r!=0: bw.append(1 if r==1 else 0)
    fav=np.array(fav);adv=np.array(adv);bw=np.array(bw)
    if len(fav)<15: print(f"  {name:<30} n={len(fav)} (too few)"); return
    print(f"  {name:<30} n={len(fav):>4}({len(fav)/span:>3.0f}/yr) bounce%={bw.mean()*100:>4.1f}% "
          f"favMFE med={np.median(fav):.2f} p90={np.percentile(fav,90):.2f} adv med={np.median(adv):.2f} fav/adv={np.median(fav)/max(np.median(adv),1e-9):.2f}")
base=cu&(rsi<40)
print("gold15m long-fade selectability (base = down-cross+RSI<40: 54.5% / favMFE2.32)\n")
exc(base,"BASE down-cross+RSI<40")
print("-- regime/context filters --")
exc(base&(adx<25),"+ ADX<25 (strategy's filter)")
exc(base&(adx>=25),"+ ADX>=25 (trending, opposite)")
exc(base&dup,"+ daily UPtrend (SMA150)")
exc(base&~dup,"+ daily DOWNtrend")
exc(base&((hr>=7)&(hr<21)),"+ session 07-21 UTC")
exc(base&((hr>=12)&(hr<21)),"+ NY session 12-21")
print("-- setup depth/strength --")
exc(base&(vol>volsma*1.5),"+ high volume (>1.5x)")
exc(base&(dip>=0.5),"+ deep dip >=0.5ATR below VWAP")
exc(base&(dip>=1.0),"+ deep dip >=1.0ATR")
exc(base&(rsi<35),"+ RSI<35")
print("-- confirmation (reclaim VWAP within 3 bars = enter later) --")
def exc_confirm(mask,name,M=3,K=24):
    idx=np.where(mask)[0]; fav=[];adv=[];bw=[];fill=0
    for i in idx:
        tj=None
        for j in range(i+1,min(i+1+M,n)):
            if c[j]>vwap[j]: tj=j; break   # reclaimed VWAP = reversal proven
        if tj is None or tj+1>=n: continue
        fill+=1; e=o[tj+1];a=atr[tj]
        if np.isnan(a) or a<=0: continue
        end=min(tj+1+K,n);mfe=0;mae=0;r=0
        for j in range(tj+1,end):
            hi=(h[j]-e)/a;lo=(e-l[j])/a;mfe=max(mfe,hi);mae=max(mae,lo)
            if r==0:
                if hi>=1.0:r=1
                elif lo>=1.0:r=-1
        fav.append(mfe);adv.append(mae)
        if r!=0:bw.append(1 if r==1 else 0)
    fav=np.array(fav);adv=np.array(adv);bw=np.array(bw)
    print(f"  {name:<30} n={len(fav):>4}({len(fav)/span:>3.0f}/yr fill{fill/max(len(idx),1)*100:.0f}%) bounce%={bw.mean()*100:>4.1f}% "
          f"favMFE med={np.median(fav):.2f} adv med={np.median(adv):.2f} fav/adv={np.median(fav)/max(np.median(adv),1e-9):.2f}")
exc_confirm(base,"confirmed reclaim: base")
exc_confirm(base&dup,"confirmed reclaim + daily UP")

print("\n\n== STEP4: RR from excursion (fav~2.3/adv~1.7 ATR). selected long-only configs, GROSS ==")
def walk_rr(mask, name, sl, tp, confirm=False, K=300):
    idx=np.where(mask)[0]; R=[];ts=[];busy=-1
    for i in idx:
        st_i=i
        if confirm:
            tj=None
            for j in range(i+1,min(i+4,n)):
                if c[j]>vwap[j]: tj=j;break
            if tj is None: continue
            st_i=tj
        if st_i<=busy or st_i+1>=n: continue
        e=o[st_i+1];a=atr[st_i]
        if np.isnan(a) or a<=0: continue
        stop=e-sl*a; tgt=e+tp*a; r=None; xj=min(st_i+1+K,n-1)
        for j in range(st_i+1,min(st_i+1+K,n)):
            if l[j]<=stop: r=-sl;xj=j;break
            if h[j]>=tgt: r=tp;xj=j;break
        if r is None: r=(c[xj]-e)/a
        R.append(r/sl);ts.append(df.index[st_i]);busy=xj
    R=np.array(R); y=np.array([t.year for t in ts])
    if len(R)<15: print(f"  {name:<34} n={len(R)} too few"); return
    pf=R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.9
    ys=sorted(set(y)); green=sum(1 for yy in ys if R[y==yy].sum()>0); half=np.median(y)
    print(f"  {name:<34} n={len(R):>4}({len(R)/span:>3.0f}/yr) PF={pf:.2f} win={(R>0).mean()*100:>3.0f}% meanR={R.mean():+.3f} "
          f"IS/OOS={R[y<half].mean():+.2f}/{R[y>=half].mean():+.2f} green={green}/{len(ys)}")
base=cu&(rsi<40)
reg = base & (adx<25) & dup                    # regime stack: RSI<40 + ADX<25 + daily-up
print("regime stack = down-cross + RSI<40 + ADX<25 + daily-UPtrend:")
for sl,tp in [(2.0,2.0),(1.5,2.5),(2.0,3.0),(1.5,2.0)]:
    walk_rr(reg, f"  RR stop{sl}/tgt{tp}", sl, tp)
print("+ confirmation (VWAP reclaim):")
for sl,tp in [(2.0,2.0),(1.5,2.5),(1.5,2.0)]:
    walk_rr(reg, f"  confirmed stop{sl}/tgt{tp}", sl, tp, confirm=True)
print("NY session variant (down-cross+RSI<40 + NY 12-21, no daily/adx):")
ny = base & ((hr>=12)&(hr<21))
walk_rr(ny, "  NY stop2.0/tgt2.5", 2.0, 2.5)
