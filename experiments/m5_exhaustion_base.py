"""gold M5 exhaustion-bounce: STEP 1 (all-signals base bounce-rate, GROSS) + STEP 2 (selectability).
Bounce-family order: base -> selectability -> excursion -> (RR later). Cost=0. No-lookahead: signal on
closed bar i, enter i+1 open, symmetric-barrier bounce-rate + forward MFE/MAE. Compare to coin-flip base."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
d = load_mt5_csv("data/vantage_xauusd_m5.csv")
o,h,l,c = (d[k].values for k in ("open","high","low","close"))
atr = ta.atr(d["high"],d["low"],d["close"],14).values
n = len(c); rng = np.maximum(h-l,1e-9)
lowerwick = (np.minimum(o,c)-l)/rng           # lower-wick fraction of range
closepos  = (c-l)/rng                          # close position in range (1=top)
# daily uptrend gate (prior daily close > daily SMA150, shift to avoid lookahead)
dc = d["close"].resample("1D").last().dropna(); dsma = dc.rolling(150).mean()
dup = ((dc>dsma).shift(1)).reindex(d.index,method="ffill").fillna(False).values
yr = np.array([t.year for t in d.index])
span = (d.index[-1]-d.index[0]).days/365.25

def newlow(W):
    nl = np.zeros(n,bool)
    for i in range(W,n):
        if l[i] <= l[i-W:i].min(): nl[i]=True
    return nl

def measure(sig, name, K=12, bar=1.0):
    idx = np.where(sig)[0]; idx = idx[(idx+1<n)]
    # symmetric-barrier bounce-rate from next open: +bar*ATR before -bar*ATR
    win=0; los=0; mfes=[]; maes=[]; fret=[]
    for i in idx:
        e=o[i+1]; a=atr[i]
        if np.isnan(a) or a<=0: continue
        up=e+bar*a; dn=e-bar*a; res=0
        end=min(i+1+300,n)
        mfe=0; mae=0
        for j in range(i+1,end):
            mfe=max(mfe,(h[j]-e)/a); mae=max(mae,(e-l[j])/a)
            if h[j]>=up: res=1; break
            if l[j]<=dn: res=-1; break
        if res==1: win+=1
        elif res==-1: los+=1
        mfes.append(mfe); maes.append(mae)
        k=min(i+1+K,n-1); fret.append((c[k]-e)/a)
    tot=win+los; br=win/tot*100 if tot else 0
    mfes=np.array(mfes); maes=np.array(maes); fret=np.array(fret)
    ratio=mfes.mean()/maes.mean() if maes.mean()>0 else 0
    print(f"  {name:<34} n={len(fret):>5} ({len(fret)/span:>4.0f}/yr) | bounce%(±{bar}ATR)={br:>4.1f}% | "
          f"fwd{K}barR med={np.median(fret):+.2f} mean={fret.mean():+.2f} std={fret.std():.2f} | MFE/MAE={ratio:.2f}")
    return fret

print("gold M5 exhaustion-bounce | STEP1 base + STEP2 selectability (GROSS, no-lookahead)")
print("coin-flip ref: bounce%≈50, MFE/MAE≈1.0, fwdR med≈0\n")
allbars = np.zeros(n,bool); allbars[20:]=True
measure(allbars,"ALL BARS (coin-flip baseline)")
nl20 = newlow(20)
measure(nl20,"BASE: new-20bar-low only")
sig_wick = nl20 & (lowerwick>=0.5) & (closepos>=0.5)
measure(sig_wick,"+ lower-wick rejection (>=0.5, close top half)")
sig_wick_strong = nl20 & (lowerwick>=0.6) & (closepos>=0.6) & (c>o)
measure(sig_wick_strong,"+ strong wick + bullish close")
# STEP2 selectability: daily uptrend gate
dupv = dup
measure(sig_wick & dupv,"+ wick + DAILY UPTREND gate")
measure(sig_wick & ~dupv,"+ wick + daily DOWNtrend (counter)")

print("\n== CONFIRMED-reversal entry (the actual exhaustion-bounce lever): wait for close>signal-bar-high ==")
def measure_confirmed(sig, name, M=6, K=12, bar=1.0):
    idx=np.where(sig)[0]; win=0;los=0;mfes=[];maes=[];fret=[];filled=0
    for i in idx:
        trig=None; clo=l[i]  # invalid if cluster/signal low breaks first
        for j in range(i+1,min(i+1+M,n)):
            if l[j]<clo: break
            if c[j]>h[i]: trig=j; break
        if trig is None or trig+1>=n: continue
        filled+=1; e=o[trig+1]; a=atr[trig]
        if np.isnan(a) or a<=0: continue
        up=e+bar*a; dn=e-bar*a; res=0; mfe=0;mae=0; end=min(trig+1+300,n)
        for j in range(trig+1,end):
            mfe=max(mfe,(h[j]-e)/a); mae=max(mae,(e-l[j])/a)
            if h[j]>=up: res=1;break
            if l[j]<=dn: res=-1;break
        if res==1:win+=1
        elif res==-1:los+=1
        mfes.append(mfe);maes.append(mae); k=min(trig+1+K,n-1); fret.append((c[k]-e)/a)
    tot=win+los; br=win/tot*100 if tot else 0
    mfes=np.array(mfes);maes=np.array(maes);fret=np.array(fret)
    ratio=mfes.mean()/maes.mean() if maes.mean()>0 else 0
    print(f"  {name:<34} n={len(fret):>5} ({len(fret)/span:>4.0f}/yr fill{filled/max(len(idx),1)*100:.0f}%) | "
          f"bounce%={br:>4.1f}% | fwd{K}R med={np.median(fret):+.2f} mean={fret.mean():+.2f} | MFE/MAE={ratio:.2f}")
measure_confirmed(sig_wick,"confirmed: wick rejection")
measure_confirmed(sig_wick & dupv,"confirmed: wick + daily UPtrend")
measure_confirmed(sig_wick_strong,"confirmed: strong wick + bullish")
# also: does confirmation help vs the coin-flip? random new-low bars confirmed
measure_confirmed(nl20,"confirmed: new-low only (no wick)")
