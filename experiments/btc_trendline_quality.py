"""BTC 15m: filter trendline breaks to QUALITY lines only (>=3 touches within tol*ATR, min span).
Does concentrating to 'real' multi-touch descending lines lift the continuation hold%? (base=coin-flip)
Long = break above a quality descending resistance line, confirmed close, enter next open."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
df=load_mt5_csv("data/vantage_btcusd_m15.csv")
o,h,l,c=(df[k].values for k in ("open","high","low","close"))
atr=ta.atr(df["high"],df["low"],df["close"],14).values; n=len(c); span=(df.index[-1]-df.index[0]).days/365.25
def zigzag(k):
    piv=[]; trend=1; ext=h[0]; exti=0
    for i in range(1,n):
        thr=k*atr[i]
        if np.isnan(thr) or thr<=0: continue
        if trend>0:
            if h[i]>ext: ext=h[i]; exti=i
            elif ext-l[i]>thr: piv.append((exti,ext,1,i)); trend=-1; ext=l[i]; exti=i
        else:
            if l[i]<ext: ext=l[i]; exti=i
            elif h[i]-ext>thr: piv.append((exti,ext,-1,i)); trend=1; ext=h[i]; exti=i
    return piv
def quality_break(k, min_touch, tol, min_span):
    """descending resistance lines with >=min_touch pivot-high touches within tol*ATR, span>=min_span.
    break = first confirmed close above the line after the last anchor pivot confirms."""
    piv=zigzag(k); highs=[(b,p,cf) for (b,p,kd,cf) in piv if kd==1]
    breaks=[]
    for a in range(len(highs)):
        for b in range(a+1,len(highs)):
            b1,p1,_=highs[a]; b2,p2,cf2=highs[b]
            if b2-b1<min_span or p2>=p1: continue      # descending + span
            slope=(p2-p1)/(b2-b1)
            # count touches: pivot highs within tol*ATR of the line (between b1 and b2)
            touch=0
            for (bx,px,_) in highs:
                if b1<=bx<=b2:
                    lp=p1+slope*(bx-b1)
                    if abs(px-lp)<=tol*atr[bx]: touch+=1
            if touch<min_touch: continue
            # find first confirmed close above the line after cf2
            for i in range(cf2,min(cf2+200,n-1)):
                line_i=p1+slope*(i-b1); line_p=p1+slope*(i-1-b1)
                if c[i]>line_i and c[i-1]<=line_p:
                    breaks.append((i,i+1)); break
    # dedup by entry bar, sort
    seen=set(); out=[]
    for bi,ei in sorted(breaks,key=lambda x:x[1]):
        if ei in seen: continue
        seen.add(ei); out.append((bi,ei))
    return out
def reaction(pairs,K=48):
    fav=[];adv=[];bw=[]
    for bi,ei in pairs:
        if ei>=n or np.isnan(atr[ei-1]) or atr[ei-1]<=0: continue
        e=o[ei];aa=atr[ei-1];end=min(ei+K,n);mfe=0;mae=0;r=0
        for j in range(ei,end):
            hi=(h[j]-e)/aa;lo=(e-l[j])/aa;mfe=max(mfe,hi);mae=max(mae,lo)
            if r==0:
                if hi>=1.0:r=1
                elif lo>=1.0:r=-1
        fav.append(mfe);adv.append(mae)
        if r!=0:bw.append(1 if r==1 else 0)
    fav=np.array(fav);adv=np.array(adv);bw=np.array(bw)
    if len(fav)<12: return None
    return len(fav),len(fav)/span,bw.mean()*100 if len(bw) else 0,np.median(fav)/max(np.median(adv),1e-9)
print("BTC 15m QUALITY descending-line break-up (>=touch, span, tol). does quality lift hold%?\n")
print(f"  {'zz_k':>5}{'touch':>6}{'span':>6}{'tol':>5}{'N':>5}{'N/yr':>6}{'hold%':>7}{'fav/adv':>8}")
for k in [3.0,4.0]:
    for mt in [3,4]:
        for spn,tol in [(30,0.5),(50,0.75)]:
            r=reaction(quality_break(k,mt,tol,spn))
            if r: print(f"  {k:>5}{mt:>6}{spn:>6}{tol:>5}{r[0]:>5}{r[1]:>6.1f}{r[2]:>6.1f}%{r[3]:>8.2f}")
print("\n(coin-flip=50%/1.0. edge if hold>55% or fav/adv>1.1)")
