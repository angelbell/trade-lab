"""Find an anti-knife SEPARATOR known AT ENTRY. Knife = R<0 & MFE<0.5 (reclaimed then dropped =
weak/fake reclaim). Features about RECLAIM QUALITY. Report W-vs-L separation, then IS/OOS + the
random-drop null on the best filter (guard vs IS-only overfit)."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from research.overfit_audit import cdd_R
d=load_mt5_csv("data/vantage_xauusd_m5.csv")
df=d.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
o,h,l,c=(df[k].values for k in ("open","high","low","close")); vol=df["volume"].values
p=(h+l+c)/3; day=df.index.normalize()
vwap=(pd.Series(p*vol,index=df.index).groupby(day).cumsum()/pd.Series(vol,index=df.index).groupby(day).cumsum().replace(0,np.nan)).values
rsi=ta.rsi(df["close"],14).values; atr=ta.atr(df["high"],df["low"],df["close"],14).values
adx=ta.adx(df["high"],df["low"],df["close"],14)["ADX_14"].values
volsma=pd.Series(vol).rolling(20).mean().values
dc=d["close"].resample("1D").last().dropna(); dsma=dc.rolling(150).mean()
dup=((dc>dsma).shift(1)).reindex(df.index,method="ffill").fillna(False).values
n=len(c); span=(df.index[-1]-df.index[0]).days/365.25
cu=np.zeros(n,bool); cu[1:]=(c[:-1]>=vwap[:-1])&(c[1:]<vwap[1:])
setup=cu&(rsi<40)&(adx<25)&dup
rows=[]  # (entry_i, feats..., R, MFE, year)
for k in np.where(setup)[0]:
    for j in range(1,4):
        if k+j>=n: break
        if c[k+j]>vwap[k+j]:
            rj=k+j; ei=rj+1
            if ei>=n or atr[rj]<=0 or np.isnan(atr[rj]): break
            a=atr[rj]; rng=max(h[rj]-l[rj],1e-9); pblow=l[k:rj+1].min()
            f=dict(body=(c[rj]-o[rj])/a, clpos=(c[rj]-l[rj])/rng, vwmarg=(c[rj]-vwap[rj])/a,
                   speed=j, rsiturn=rsi[rj]-rsi[k], rsidip=np.nanmin(rsi[k:rj+1]),
                   volr=vol[rj]/volsma[rj] if volsma[rj]>0 else 1, lowwick=(min(o[rj],c[rj])-l[rj])/rng,
                   e_vs_pb=(o[ei]-pblow)/a, vwslope=(vwap[rj]-vwap[max(rj-4,0)])/a)
            # outcome (ATR1.5 stop, 5ATR tgt)
            e=o[ei];stop=e-1.5*a;tgt=e+5*a;mfe=0;R=None;xj=min(ei+300,n-1)
            for m in range(ei,min(ei+300,n)):
                mfe=max(mfe,(h[m]-e)/a)
                if l[m]<=stop:R=-1.0;xj=m;break
                if h[m]>=tgt:R=5/1.5;xj=m;break
            if R is None:R=((c[xj]-e)/a)/1.5
            rows.append((ei,f,R,mfe,df.index[ei].year)); break
seen=set(); rows=[r for r in rows if not (r[0] in seen or seen.add(r[0]))]
R=np.array([r[2] for r in rows]); MFE=np.array([r[3] for r in rows]); YR=np.array([r[4] for r in rows])
feats=list(rows[0][1].keys()); F={k:np.array([r[1][k] for r in rows]) for k in feats}
W=R>0; KN=(R<0)&(MFE<0.5)
print(f"n={len(R)} win={W.mean()*100:.0f}% knife(R<0&MFE<0.5)={KN.mean()*100:.0f}%\n")
print("== feature separation: median W vs knife, and win-rate high vs low half ==")
print(f"  {'feature':<10}{'W med':>8}{'knife med':>10}{'win@high':>10}{'win@low':>9}")
for k in feats:
    x=F[k]; med=np.median(x); hi=x>=med
    print(f"  {k:<10}{np.median(x[W]):>8.2f}{np.median(x[KN]):>10.2f}{(R[hi]>0).mean()*100:>9.0f}%{(R[~hi]>0).mean()*100:>8.0f}%")
def check(mask,name):
    Rm=R[mask]; ym=YR[mask]
    if len(Rm)<15: print(f"  {name:<30} n={len(Rm)} too few"); return
    pf=Rm[Rm>0].sum()/abs(Rm[Rm<=0].sum()) if (Rm<=0).any() else 9.9; half=np.median(ym)
    # random-drop null on meanR: keep len(Rm) at random from full R
    rng=np.random.default_rng(0); nul=[np.mean(rng.choice(R,len(Rm),replace=False)) for _ in range(3000)]
    pct=(np.array(nul)<Rm.mean()).mean()*100
    print(f"  {name:<30} n={len(Rm):>3} PF={pf:.2f} win={(Rm>0).mean()*100:>3.0f}% meanR={Rm.mean():+.3f} "
          f"IS/OOS={Rm[ym<half].mean():+.2f}/{Rm[ym>=half].mean():+.2f} rd-null%ile={pct:.0f}")
print("\n== TEST candidate anti-knife filters (with random-drop null vs base) ==")
check(np.ones(len(R),bool),"BASE (all)")
for k in feats:
    x=F[k]; med=np.median(x)
    # keep the side with higher win-rate
    hi=x>=med
    side = hi if (R[hi]>0).mean()>=(R[~hi]>0).mean() else ~hi
    check(side, f"keep {k} {'>=' if side is hi else '<'}med")

print("\n\n#### VALIDATE clpos anti-knife filter ####")
clp=F["clpos"]; vm=F["vwmarg"]
def walk_cost(mask, spread, sslip):
    # recompute R with cost+slip on the masked entries (need entry idx + atr); approximate via stored R adjust
    pass
# threshold plateau (gross meanR + rd-null)
rng=np.random.default_rng(1)
print("-- clpos threshold PLATEAU (gross) --")
for th in [0.6,0.7,0.8,0.85,0.9]:
    m=clp>=th; Rm=R[m]
    if len(Rm)<15: print(f"  clpos>={th}: n={len(Rm)} few"); continue
    nul=[np.mean(rng.choice(R,len(Rm),replace=False)) for _ in range(3000)]
    print(f"  clpos>={th}: n={len(Rm):>3}({len(Rm)/span:.0f}/yr) win={(Rm>0).mean()*100:.0f}% meanR={Rm.mean():+.3f} rd-null%ile={(np.array(nul)<Rm.mean()).mean()*100:.0f}")
print("-- clpos + vwmarg (both reclaim-strength; stack or redundant?) --")
for (a,b) in [(0.85,0.2),(0.85,0.4),(0.9,0.3)]:
    m=(clp>=a)&(vm>=b); Rm=R[m]
    if len(Rm)<15: print(f"  clpos>={a}&vwmarg>={b}: n={len(Rm)} few"); continue
    ym=YR[m]; half=np.median(ym)
    print(f"  clpos>={a}&vwmarg>={b}: n={len(Rm):>3} win={(Rm>0).mean()*100:.0f}% meanR={Rm.mean():+.3f} IS/OOS={Rm[ym<half].mean():+.2f}/{Rm[ym>=half].mean():+.2f}")
print("-- PER-YEAR: does clpos fix recent (base had 2021/2025 red)? --")
for lab,m in [("BASE",np.ones(len(R),bool)),("clpos>=0.85",clp>=0.85)]:
    Rm=R[m];ym=YR[m]
    print(f"  {lab:<12} "+" ".join(f"{y}:{Rm[ym==y].sum():+.0f}({(ym==y).sum()})" for y in sorted(set(ym)) if (ym==y).sum()>=2))

print("\n#### COST STRESS on clpos-filtered (does the fatter edge cross the cost wall?) ####")
eis=np.array([r[0] for r in rows]); clp_arr=np.array([r[1]["clpos"] for r in rows]); vm_arr=np.array([r[1]["vwmarg"] for r in rows])
def cwalk(mask,spread,sslip,tp=5.0,sl=1.5):
    Rc=[];busy=-1
    for ei in np.sort(eis[mask]):
        if ei<=busy or ei>=n: continue
        e=o[ei];a=atr[ei-1]
        if np.isnan(a) or a<=0: continue
        risk=sl*a;stop=e-risk;tgt=e+tp*a;r=None;xj=min(ei+300,n-1)
        for j in range(ei,min(ei+300,n)):
            if l[j]<=stop:over=stop-l[j];r=-1.0-sslip*over/risk;xj=j;break
            if h[j]>=tgt:r=(tgt-e)/risk;xj=j;break
        if r is None:r=(c[xj]-e)/risk
        r-=spread/risk;Rc.append(r);busy=xj
    return np.array(Rc)
for lab,mask in [("BASE(all)",np.ones(len(eis),bool)),("clpos>=0.8",clp_arr>=0.8),("clpos>=0.85",clp_arr>=0.85),("clpos>=0.85&vwmarg>=0.2",(clp_arr>=0.85)&(vm_arr>=0.2))]:
    row=f"  {lab:<24}"
    for sp,ss in [(0.0,0.0),(0.4,0.5),(0.8,0.5)]:
        Rc=cwalk(mask,sp,ss); 
        if len(Rc)>=15: row+=f" |${sp}:meanR{Rc.mean():+.2f}/PF{(Rc[Rc>0].sum()/abs(Rc[Rc<=0].sum()) if (Rc<=0).any() else 9.9):.2f}"
    print(row+f"  (n={mask.sum()})")

print("\n#### OVERFIT AUDIT on clpos-filtered (net $0.6) -- does it survive the trial haircut? ####")
from research.edge_harness import audit
cfgs={}
for th in (0.7,0.8,0.85,0.9):
    mask=clp_arr>=th; Rc=[];busy=-1; TS=[]
    for ei in np.sort(eis[mask]):
        if ei<=busy or ei>=n: continue
        e=o[ei];a=atr[ei-1]
        if np.isnan(a) or a<=0: continue
        risk=1.5*a;stop=e-risk;tgt=e+5*a;r=None;xj=min(ei+300,n-1)
        for j in range(ei,min(ei+300,n)):
            if l[j]<=stop:over=stop-l[j];r=-1.0-0.5*over/risk;xj=j;break
            if h[j]>=tgt:r=(tgt-e)/risk;xj=j;break
        if r is None:r=(c[xj]-e)/risk
        r-=0.6/risk;Rc.append(r);TS.append(df.index[ei]);busy=xj
    cfgs[f"clpos{th}"]=list(zip(TS,Rc))
for k,v in cfgs.items():
    rr=np.array([x for _,x in v]); print(f"  {k}: n={len(rr)} meanR={rr.mean():+.3f} totR={rr.sum():+.1f}")
audit(cfgs, flagship="clpos0.85")
