import sys, subprocess, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import numpy as np, pandas as pd
CFG=["--csv","data/vantage_xauusd_m5.csv","--tf","15min","--pattern","B","--swing","zigzag",
     "--zz-k","2","--trend-ema","80","--bo-window","20","--tp-mode","rr","--rr","4","--fwd","500",
     "--daily-sma","150","--daily-slope-k","10","--risk","0.01","--cost","0.0002","--ext-cap","8"]
out=subprocess.run([".venv/bin/python","breakout_wave.py",*CFG,"--dump-trades"],
                   capture_output=True,text=True,cwd="/home/angelbell/dev/auto-trade").stdout.splitlines()
i=next(k for k,l in enumerate(out) if l.startswith("entry_time,"))
t=pd.read_csv(pd.io.common.StringIO("\n".join(out[i:])))
t["time"]=pd.to_datetime(t["entry_time"]); t["y"]=t["time"].dt.year
t["h"]=t["time"].dt.hour; t["dow"]=t["time"].dt.dayofweek
SPLIT=2022

print(f"n={len(t)} meanR={t.R.mean():+.2f}\n## by HOUR-OF-DAY (UTC broker)")
print(f"  {'hr':>3}{'n':>5}{'win%':>6}{'meanR':>7}{'IS':>7}{'OOS':>7}  flag")
for h in range(24):
    g=t[t.h==h]
    if len(g)==0: continue
    isr=g[g.y<SPLIT].R; oosr=g[g.y>=SPLIT].R
    consist = "" 
    if len(g)>=15 and isr.mean()>0.3 and oosr.mean()>0.3: consist="++ both good"
    if len(g)>=15 and isr.mean()<0.0 and oosr.mean()<0.0: consist="-- both bad"
    print(f"  {h:>3}{len(g):>5}{(g.R>0).mean()*100:>6.0f}{g.R.mean():>+7.2f}{isr.mean():>+7.2f}{oosr.mean():>+7.2f}  {consist}")

# coarse 3-hour windows (less noise)
print("\n## by 3-HOUR WINDOW")
print(f"  {'win':>7}{'n':>5}{'win%':>6}{'meanR':>7}{'IS':>7}{'OOS':>7}")
t["w3"]=(t.h//3)*3
for w,g in t.groupby("w3"):
    isr=g[g.y<SPLIT].R; oosr=g[g.y>=SPLIT].R
    print(f"  {w:>2}-{w+3:<4}{len(g):>5}{(g.R>0).mean()*100:>6.0f}{g.R.mean():>+7.2f}{isr.mean():>+7.2f}{oosr.mean():>+7.2f}")

# --- null test: drop the dead window, vs random-drop on CAGR/DD ---
def cagr_dd(tt):
    tt=tt.sort_values("time"); eq=(1+0.01*tt.R).cumprod()
    dd=((eq.cummax()-eq)/eq.cummax()).max()*100
    span=max((tt.time.iloc[-1]-tt.time.iloc[0]).days/365.25,.5)
    return (eq.iloc[-1]**(1/span)-1)*100/max(dd,1e-9)
def null(mask,iters=4000,seed=2):
    rng=np.random.default_rng(seed); k=int(mask.sum()); n=len(t); idx=np.arange(n)
    real=cagr_dd(t[mask]); dr=np.array([cagr_dd(t.iloc[np.sort(rng.choice(idx,k,replace=False))]) for _ in range(iters)])
    return real,(dr<real).mean()*100,np.median(dr)
print(f"\nBASE CAGR/DD={cagr_dd(t):.2f}  (n={len(t)})")
for lbl,m in [("drop 9-12",~t.h.between(9,11)),("drop 9-15",~t.h.between(9,14)),
              ("drop 10-14",~t.h.between(10,13)),("drop 9-12 & 21-24",~(t.h.between(9,11)|t.h.between(21,23)))]:
    r,p,md=null(m.values); print(f"  {lbl:<20} keep={int(m.sum()):>3}  CAGR/DD={r:.2f}  null_med={md:.2f}  pctile={p:.0f}%")

print("\n## drop-9-15 leg: robustness")
g=t[~t.h.between(9,14)]
isr=g[g.y<SPLIT].R; oosr=g[g.y>=SPLIT].R
print(f"  n={len(g)} meanR={g.R.mean():+.2f} win={(g.R>0).mean()*100:.0f}%  IS={isr.mean():+.2f} OOS={oosr.mean():+.2f}")
print("  per-year totR (kept):")
for y,gg in g.groupby("y"):
    print(f"    {y}: {gg.R.sum():+6.1f} (n={len(gg)})")
print("  dropped set (9-15):")
dr=t[t.h.between(9,14)]
print(f"    n={len(dr)} meanR={dr.R.mean():+.2f}  IS={dr[dr.y<SPLIT].R.mean():+.2f} OOS={dr[dr.y>=SPLIT].R.mean():+.2f}")
