"""BTC 15m trendline-break BASE test (proper order): causal ATR-ZigZag -> line thru last-2 pivots
(descending highs=long resistance / ascending lows=short support) -> confirmed-close break -> enter
next open. Measure reaction (continuation) GROSS: bounce/hold% (±1ATR), MFE/MAE. Both directions."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
df=load_mt5_csv("data/vantage_btcusd_m15.csv")
o,h,l,c=(df[k].values for k in ("open","high","low","close"))
atr=ta.atr(df["high"],df["low"],df["close"],14).values; n=len(c); span=(df.index[-1]-df.index[0]).days/365.25
def zigzag(h,l,atr,k):
    piv=[]; trend=1; ext=h[0]; exti=0
    for i in range(1,n):
        thr=k*atr[i]
        if np.isnan(thr) or thr<=0: continue
        if trend>0:
            if h[i]>ext: ext=h[i]; exti=i
            elif ext-l[i]>thr: piv.append((exti,ext,1,i)); trend=-1; ext=l[i]; exti=i  # (bar,price,kind,confirm_bar)
        else:
            if l[i]<ext: ext=l[i]; exti=i
            elif h[i]-ext>thr: piv.append((exti,ext,-1,i)); trend=1; ext=h[i]; exti=i
    return piv
def signals(k):
    piv=zigzag(h,l,atr,k)
    highs=[(b,p,cf) for (b,p,kd,cf) in piv if kd==1]
    lows =[(b,p,cf) for (b,p,kd,cf) in piv if kd==-1]
    longs=[]; shorts=[]
    # LONG: last-2 confirmed pivot HIGHS descending -> break above line (confirmed close)
    for idx in range(1,len(highs)):
        (b1,p1,_),(b2,p2,cf2)=highs[idx-1],highs[idx]
        if b2<=b1 or p2>=p1: continue        # need descending (resistance) line
        slope=(p2-p1)/(b2-b1)
        # valid from cf2 (line known) until the NEXT high pivot confirms
        end=highs[idx+1][2] if idx+1<len(highs) else n
        for i in range(max(cf2,b2+1),min(end,n-1)):
            line_i=p2+slope*(i-b2); line_p=p2+slope*(i-1-b2)
            if c[i]>line_i and c[i-1]<=line_p: longs.append(i+1); break
    # SHORT: last-2 confirmed pivot LOWS ascending -> break below line
    for idx in range(1,len(lows)):
        (b1,p1,_),(b2,p2,cf2)=lows[idx-1],lows[idx]
        if b2<=b1 or p2<=p1: continue         # need ascending (support) line
        slope=(p2-p1)/(b2-b1)
        end=lows[idx+1][2] if idx+1<len(lows) else n
        for i in range(max(cf2,b2+1),min(end,n-1)):
            line_i=p2+slope*(i-b2); line_p=p2+slope*(i-1-b2)
            if c[i]<line_i and c[i-1]>=line_p: shorts.append(i+1); break
    return sorted(set(longs)),sorted(set(shorts))
def reaction(entry_bars, side, K=48):
    fav=[];adv=[];bw=[]
    for ei in entry_bars:
        if ei>=n or np.isnan(atr[ei-1]) or atr[ei-1]<=0: continue
        e=o[ei];a=atr[ei-1];end=min(ei+K,n);mfe=0;mae=0;r=0
        for j in range(ei,end):
            f=side*(h[j]-e)/a if side>0 else side*(l[j]-e)/a
            ad=side*(l[j]-e)/a if side>0 else side*(h[j]-e)/a
            mfe=max(mfe,f); mae=min(mae,ad)
            if r==0:
                if f>=1.0: r=1
                elif -ad>=1.0: r=-1
        fav.append(mfe);adv.append(-mae)
        if r!=0:bw.append(1 if r==1 else 0)
    fav=np.array(fav);adv=np.array(adv);bw=np.array(bw)
    if len(fav)<12: return None
    return len(fav),len(fav)/span,bw.mean()*100 if len(bw) else 0,np.median(fav),np.median(adv),np.median(fav)/max(np.median(adv),1e-9)
print("BTC 15m trendline-break BASE (gross, K=48 forward). continuation-hold%(±1ATR) + MFE/MAE\n")
print(f"  {'zz_k':>5}{'side':>7}{'N':>5}{'N/yr':>6}{'hold%':>7}{'favMFE':>8}{'advMAE':>8}{'fav/adv':>8}")
for k in [2.0,3.0,4.0]:
    L,S=signals(k)
    for nm,bars,sd in [("long",L,1),("short",S,-1)]:
        r=reaction(bars,sd)
        if r: print(f"  {k:>5}{nm:>7}{r[0]:>5}{r[1]:>6.1f}{r[2]:>6.1f}%{r[3]:>8.2f}{r[4]:>8.2f}{r[5]:>8.2f}")
print("\n(coin-flip ref: hold~50%, fav/adv~1.0. edge if hold>55% or fav/adv>1.1 = break continues)")

print("\n\n#### SELECTIVITY on descending-break-up (long), zz_k=3 ####")
from breakout_wave import kama_adaptive
vol=df["volume"].values; volsma=pd.Series(vol).rolling(20).mean().values
clpos=(c-l)/np.maximum(h-l,1e-9)
dck=df["close"].resample("1D").last().dropna(); kmg=kama_adaptive(dck,14)
kup=((kmg>kmg.shift(1)).shift(1)).reindex(df.index,method="ffill").fillna(False).values
sma200=df["close"].rolling(200).mean().values
def signals_feat(k):
    piv=zigzag(h,l,atr,k); highs=[(b,p,cf) for (b,p,kd,cf) in piv if kd==1]; out=[]
    for idx in range(1,len(highs)):
        (b1,p1,_),(b2,p2,cf2)=highs[idx-1],highs[idx]
        if b2<=b1 or p2>=p1: continue
        slope=(p2-p1)/(b2-b1); end=highs[idx+1][2] if idx+1<len(highs) else n
        for i in range(max(cf2,b2+1),min(end,n-1)):
            line_i=p2+slope*(i-b2); line_p=p2+slope*(i-1-b2)
            if c[i]>line_i and c[i-1]<=line_p:
                out.append((i, i+1)); break   # (break_bar, entry_bar)
    return out
S=signals_feat(3.0)
def react_mask(pairs, K=48):
    fav=[];adv=[];bw=[]
    for bi,ei in pairs:
        if ei>=n or np.isnan(atr[ei-1]) or atr[ei-1]<=0: continue
        e=o[ei];a=atr[ei-1];end=min(ei+K,n);mfe=0;mae=0;r=0
        for j in range(ei,end):
            hi=(h[j]-e)/a;lo=(e-l[j])/a;mfe=max(mfe,hi);mae=max(mae,lo)
            if r==0:
                if hi>=1.0:r=1
                elif lo>=1.0:r=-1
        fav.append(mfe);adv.append(mae)
        if r!=0:bw.append(1 if r==1 else 0)
    fav=np.array(fav);adv=np.array(adv);bw=np.array(bw)
    if len(fav)<12: return None
    return len(fav),len(fav)/span,bw.mean()*100 if len(bw) else 0,np.median(fav)/max(np.median(adv),1e-9)
def show(pairs,name):
    r=react_mask(pairs)
    if r: print(f"  {name:<34} N={r[0]:>4}({r[1]:>5.1f}/yr) hold%={r[2]:>4.1f}% fav/adv={r[3]:.2f}")
    else: print(f"  {name:<34} too few")
show(S,"base (all descending-break-up)")
show([(bi,ei) for bi,ei in S if clpos[bi]>=0.7],"+ strong break candle clpos>=0.7")
show([(bi,ei) for bi,ei in S if clpos[bi]>=0.85],"+ strong break clpos>=0.85")
show([(bi,ei) for bi,ei in S if vol[bi]>volsma[bi]*1.5],"+ volume>1.5x (conviction)")
show([(bi,ei) for bi,ei in S if kup[bi]],"+ daily-KAMA rising (HTF up=cont)")
show([(bi,ei) for bi,ei in S if not kup[bi]],"+ daily-KAMA falling (reversal)")
show([(bi,ei) for bi,ei in S if c[bi]<sma200[bi]],"+ below SMA200 (deep reversal)")
show([(bi,ei) for bi,ei in S if c[bi]>sma200[bi]],"+ above SMA200 (continuation)")
