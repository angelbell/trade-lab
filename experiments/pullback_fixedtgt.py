"""User's idea: keep the SAME stop (L2) AND the SAME target LEVEL (original RR4 tgt, fixed
in price), lower ONLY the entry to a pullback limit. Then realized RR balloons + cost is a
smaller share of a bigger reward. Tax = win-rate (entry nearer stop) + missed runaways
(strong breaks that hit tgt before pulling back). Faithful port of breakout_wave Pattern-B."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag
AGG = {"open":"first","high":"max","low":"min","close":"last"}
RR, BO, FWD, COST = 4.0, 20, 500, 0.001

d = load_mt5_csv("data/vantage_xauusd_m5.csv").resample("15min").agg(AGG).dropna()
h,l,c = d["high"].values, d["low"].values, d["close"].values
a = ta.atr(d["high"],d["low"],d["close"],14).values
es = d["close"].ewm(span=80,adjust=False).mean().values
dc = d["close"].resample("1D").last().dropna(); sma = dc.rolling(150).mean()
up = ((dc>sma) & (sma>sma.shift(10))).shift(1)
reg = up.reindex(d.index,method="ffill").fillna(False).values
ext = ((dc-sma)/sma*100.0).shift(1); ext_arr = ext.reindex(d.index,method="ffill").values
sw = swings_zigzag(h,l,a,2.0)

def first_breakout(level, after):
    for j in range(after, min(after+BO, len(c))):
        if c[j] > level: return j
    return None

entries=[]
for t in range(2,len(sw)):
    (cL2,iL2,pL2,kL2),(cH1,iH1,pH1,kH1),(cL0,iL0,pL0,kL0)=sw[t],sw[t-1],sw[t-2]
    if not (kL2==-1 and kH1==+1 and kL0==-1): continue
    if pL2<=pL0 or pH1-pL0<=0: continue
    if es is not None and not np.isnan(es[cL2]) and pH1<es[cL2]: continue
    e_i=first_breakout(pH1,cL2+1)
    if e_i is None: continue
    if not reg[e_i]: continue
    if not np.isnan(ext_arr[e_i]) and ext_arr[e_i]>8: continue
    e=c[e_i]; stop=pL2; risk=e-stop
    if risk<=0: continue
    tgt=e+RR*risk
    if tgt<=e: continue
    entries.append((e_i,e,stop,tgt))
entries.sort(key=lambda x:x[0])
seen=set(); uniq=[]
for en in entries:
    if en[0] in seen: continue
    seen.add(en[0]); uniq.append(en)
entries=uniq

def stats(trades):
    R=np.array([r for _,r in trades]); ts=[t for t,_ in trades]
    yr=np.array([t.year for t in ts]); span=(ts[-1]-ts[0]).days/365.25
    pf=R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.99
    cum=np.cumsum(R); dd=(np.maximum.accumulate(cum)-cum).max()
    yrs=np.unique(yr); half=yrs[len(yrs)//2]
    grn=sum(1 for y in yrs if R[yr==y].sum()>0)
    return dict(N=len(R),npy=len(R)/span,win=(R>0).mean()*100,pf=pf,meanR=R.mean(),
                IS=R[yr<half].mean(),OOS=R[yr>=half].mean(),retdd=R.sum()/dd if dd>0 else np.inf,
                grn=grn,ny=len(yrs))

def eval_market():
    busy=-1; tr=[]
    for (i,e,stop,tgt) in entries:
        if i<=busy: continue
        risk=e-stop; reward=tgt-e; exit_j=min(i+FWD,len(c)-1); R=None
        for j in range(i+1,min(i+1+FWD,len(c))):
            if l[j]<=stop: R=-1.0; exit_j=j; break
            if h[j]>=tgt: R=reward/risk; exit_j=j; break
        if R is None: R=(c[exit_j]-e)/risk
        R-=COST/risk*e; tr.append((d.index[i],R)); busy=exit_j
    return tr

def eval_pullback(frac):
    """limit at e - frac*(e-L2). fill if low<=lim BEFORE high>=tgt (else runaway=missed).
    stop=L2 & tgt fixed. R in units of realized risk (lim-L2)."""
    busy=-1; tr=[]; missed=0
    for (i,e,stop,tgt) in entries:
        if i<=busy: continue
        lim=e-frac*(e-stop); fill_j=None
        for j in range(i+1,min(i+1+FWD,len(c))):
            if h[j]>=tgt: break              # ran to target first = MISSED (adverse selection)
            if l[j]<=lim: fill_j=j; break
            if l[j]<=stop: break             # (can't happen: lim>stop) safety
        if fill_j is None: missed+=1; continue
        risk=lim-stop; reward=tgt-lim
        if risk<=0: missed+=1; continue
        if l[fill_j]<=stop: R=-1.0; exit_j=fill_j   # same bar dumped through to stop
        else:
            exit_j=min(fill_j+FWD,len(c)-1); R=None
            for j in range(fill_j+1,min(fill_j+1+FWD,len(c))):
                if l[j]<=stop: R=-1.0; exit_j=j; break
                if h[j]>=tgt: R=reward/risk; exit_j=j; break
            if R is None: R=(c[exit_j]-lim)/risk
        R-=COST/risk*lim; tr.append((d.index[fill_j],R)); busy=exit_j
    return tr, missed

m=eval_market(); s=stats(m)
print(f"MARKET baseline (entry=break close, stop=L2, tgt=e+4R):")
print(f"  N={s['N']} N/yr={s['npy']:.0f} win={s['win']:.0f}% PF={s['pf']:.2f} meanR={s['meanR']:+.3f} "
      f"IS/OOS={s['IS']:+.2f}/{s['OOS']:+.2f} green={s['grn']}/{s['ny']} ret/DD={s['retdd']:+.2f}")
print(f"\nPULLBACK-LIMIT + FIXED stop(L2) & tgt(orig 4R level), entry lowered by frac of risk:")
print(f"  {'frac':>5}{'N':>5}{'miss%':>7}{'N/yr':>6}{'win':>6}{'PF':>7}{'meanR':>8}{'IS/OOS':>13}{'grn':>6}{'ret/DD':>8}{'effRR':>7}")
for frac in (0.25,0.5,0.75,1.0):
    tr,missed=eval_pullback(frac); 
    if len(tr)<12: print(f"  {frac:>5}  too few"); continue
    s=stats(tr); miss_pct=missed/(missed+len(tr))*100
    effRR=(s['win']/100)  # placeholder
    # realized RR = mean winning reward/risk approx via frac geometry: (1-frac)+RR effective... report win-side avg
    R=np.array([r for _,r in tr]); avgwin=R[R>0].mean() if (R>0).any() else 0
    isoos=f"{s['IS']:+.2f}/{s['OOS']:+.2f}"
    print(f"  {frac:>5}{s['N']:>5}{miss_pct:>6.0f}%{s['npy']:>6.0f}{s['win']:>5.0f}%{s['pf']:>7.2f}{s['meanR']:>+8.3f}"
          f"{isoos:>13}{s['grn']:>3}/{s['ny']}{s['retdd']:>8.2f}{avgwin:>7.1f}")
print("  (miss% = strong breaks that ran to tgt before pulling back = lost. effRR≈avg winning R)")

print("\n==== COST STRESS: does pullback degrade SLOWER than market? (the thesis) ====")
print("  cost is a fraction of notional (0.001 ~ $2/trade on gold). meanR at rising cost:")
print(f"  {'cost':>7}{'market':>9}{'frac0.25':>10}{'frac0.5':>10}")
for cst in (0.0, 0.001, 0.002, 0.003, 0.005):
    globals()['COST']=cst
    m=stats(eval_market())['meanR']
    p25=stats(eval_pullback(0.25)[0])['meanR']; p50=stats(eval_pullback(0.5)[0])['meanR']
    print(f"  {cst:>7}{m:>+9.3f}{p25:>+10.3f}{p50:>+10.3f}")
globals()['COST']=0.001
print("  (thesis TRUE if market falls faster into the red than pullback as cost rises)")

print("\n==== PER-YEAR meanR (era-concentration check) ====")
def peryear(trades,label):
    R=np.array([r for _,r in trades]); ts=[t for t,_ in trades]; yr=np.array([t.year for t in ts])
    ys=sorted(set(yr)); print(f"  {label:<10}"+" ".join(f"{y}:{R[yr==y].mean():+.2f}({(yr==y).sum()})" for y in ys))
peryear(eval_market(),"market")
peryear(eval_pullback(0.25)[0],"frac0.25")
peryear(eval_pullback(0.5)[0],"frac0.5")
