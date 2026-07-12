"""Is the pullback depth a Fibonacci structure or just a monotonic 'deeper=more effRR' ramp?
frac (of risk = e-L2) IS the retracement fraction of the L2->e breakout leg. Fine-sweep and
check whether Fib levels (.236/.382/.5/.618/.786) BUMP above their non-Fib neighbors, or the
curve is smooth (=> Fib irrelevant). Realistic cost $0.6."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
exec(open("scratchpad/pullback_fixedtgt.py").read().split("m=eval_market()")[0])  # setup + entries + stats
SP=0.6
def eval_pull_abs(frac,spread=SP):
    busy=-1; tr=[]; miss=0
    for (i,e,stop,tgt) in entries:
        if i<=busy: continue
        lim=e-frac*(e-stop); fill_j=None
        for j in range(i+1,min(i+1+FWD,len(c))):
            if h[j]>=tgt: break
            if l[j]<=lim: fill_j=j; break
        if fill_j is None: miss+=1; continue
        risk=lim-stop; reward=tgt-lim
        if risk<=0: miss+=1; continue
        if l[fill_j]<=stop: R=-1.0; exit_j=fill_j
        else:
            exit_j=min(fill_j+FWD,len(c)-1); R=None
            for j in range(fill_j+1,min(fill_j+1+FWD,len(c))):
                if l[j]<=stop: R=-1.0; exit_j=j; break
                if h[j]>=tgt: R=reward/risk; exit_j=j; break
            if R is None: R=(c[exit_j]-lim)/risk
        R-=spread/risk; tr.append((d.index[fill_j],R)); busy=exit_j
    return tr,miss
FIB={0.236,0.382,0.5,0.618,0.786}
print("fine depth sweep (frac of risk = retracement of L2->e leg). * = Fibonacci level:")
print(f"  {'frac':>6}{'':2}{'N':>5}{'miss%':>7}{'win':>6}{'PF':>7}{'meanR':>8}{'IS/OOS':>13}{'ret/DD':>8}")
for frac in [round(x,3) for x in np.arange(0.15,0.83,0.05)]+sorted(FIB):
    pass
grid=sorted(set([round(x,3) for x in np.arange(0.15,0.82,0.0384)]) | FIB)  # ~fine + exact fibs
for frac in grid:
    tr,miss=eval_pull_abs(frac)
    if len(tr)<12: continue
    s=stats(tr); mp=miss/(miss+len(tr))*100; star=" *" if round(frac,3) in {round(f,3) for f in FIB} else "  "
    print(f"  {frac:>6.3f}{star}{s['N']:>5}{mp:>6.0f}%{s['win']:>5.0f}%{s['pf']:>7.2f}{s['meanR']:>+8.3f}"
          f"{s['IS']:>+6.2f}/{s['OOS']:>+.2f}{s['retdd']:>8.2f}")
print("  (smooth ramp => Fib irrelevant=just depth; bumps AT * vs neighbors => structural)")

print("\n==== PBO: was 0.62 from mixing DEEP lumpy configs? shallow-only grid should drop it ====")
from research.edge_harness import audit
def cfgset(fracs):
    return {f"f{fr}":[(t,r) for t,r in eval_pull_abs(fr)[0]] for fr in fracs}
print("\n-- SHALLOW robust grid {0.19,0.27,0.34} (IS~OOS zone) --")
audit(cfgset([0.19,0.27,0.34]), flagship="f0.27")
print("\n-- DEEP grid {0.5,0.62,0.75} (IS-loaded/lumpy zone) --")
audit(cfgset([0.5,0.62,0.75]), flagship="f0.62")
