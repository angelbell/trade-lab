import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
exec(open("experiments/pullback_fixedtgt.py").read().split("m=eval_market()")[0])  # reuse setup+funcs
print(f"gold 15m price range: {c.min():.0f} - {c.max():.0f}  (median {np.median(c):.0f})")

# ABSOLUTE-dollar spread cost model: cost_R = spread$/risk (realistic for gold).
def eval_market_abs(spread):
    busy=-1; tr=[]
    for (i,e,stop,tgt) in entries:
        if i<=busy: continue
        risk=e-stop; reward=tgt-e; exit_j=min(i+FWD,len(c)-1); R=None
        for j in range(i+1,min(i+1+FWD,len(c))):
            if l[j]<=stop: R=-1.0; exit_j=j; break
            if h[j]>=tgt: R=reward/risk; exit_j=j; break
        if R is None: R=(c[exit_j]-e)/risk
        R-=spread/risk; tr.append((d.index[i],R)); busy=exit_j
    return tr
def eval_pull_abs(frac,spread):
    busy=-1; tr=[]
    for (i,e,stop,tgt) in entries:
        if i<=busy: continue
        lim=e-frac*(e-stop); fill_j=None
        for j in range(i+1,min(i+1+FWD,len(c))):
            if h[j]>=tgt: break
            if l[j]<=lim: fill_j=j; break
        if fill_j is None: continue
        risk=lim-stop; reward=tgt-lim
        if risk<=0: continue
        if l[fill_j]<=stop: R=-1.0; exit_j=fill_j
        else:
            exit_j=min(fill_j+FWD,len(c)-1); R=None
            for j in range(fill_j+1,min(fill_j+1+FWD,len(c))):
                if l[j]<=stop: R=-1.0; exit_j=j; break
                if h[j]>=tgt: R=reward/risk; exit_j=j; break
            if R is None: R=(c[exit_j]-lim)/risk
        R-=spread/risk; tr.append((d.index[fill_j],R)); busy=exit_j
    return tr

print("\n==== ABSOLUTE spread$ cost (realistic gold ~$0.3-0.8 rt). meanR / ret/DD: ====")
print(f"  {'spread$':>8}{'market':>16}{'frac0.25':>16}{'frac0.5':>16}")
for sp in (0.0, 0.4, 0.8, 1.5, 3.0):
    sm=stats(eval_market_abs(sp)); s25=stats(eval_pull_abs(0.25,sp)); s50=stats(eval_pull_abs(0.5,sp))
    print(f"  {sp:>8.1f}   {sm['meanR']:+.3f}/{sm['retdd']:>5.2f}   {s25['meanR']:+.3f}/{s25['retdd']:>5.2f}   {s50['meanR']:+.3f}/{s50['retdd']:>5.2f}")
print("  (realistic gold rt spread ~0.4-0.8. does pullback still beat market there?)")

print("\n\n==== OVERFIT GAUNTLET @ realistic spread $0.6 (market vs frac family) ====")
from research.edge_harness import audit, random_drop_null
SP=0.6
cfgs = {"market":[(t,r) for t,r in eval_market_abs(SP)],
        "frac0.25":[(t,r) for t,r in eval_pull_abs(0.25,SP)],
        "frac0.5":[(t,r) for t,r in eval_pull_abs(0.5,SP)],
        "frac0.75":[(t,r) for t,r in eval_pull_abs(0.75,SP)]}
for k,v in cfgs.items():
    R=np.array([r for _,r in v]); print(f"  {k:<9} n={len(R)} meanR={R.mean():+.3f} totR={R.sum():+.1f}")
audit(cfgs, flagship="frac0.25")
print("\n-- is frac0.25 real vs just dropping/replacing market trades at random? --")
random_drop_null(cfgs["market"], cfgs["frac0.25"], years=(d.index[-1]-d.index[0]).days/365.25)
