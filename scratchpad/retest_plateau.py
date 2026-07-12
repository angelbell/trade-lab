import numpy as np, pandas as pd, io
def load(f):
    lines=open(f).read().splitlines(); h=next(i for i,l in enumerate(lines) if l.startswith("entry_time"))
    d=pd.read_csv(io.StringIO("\n".join(lines[h:]))); d["entry_time"]=pd.to_datetime(d["entry_time"])
    return d.sort_values("entry_time").reset_index(drop=True)
def retdd(R):
    c=np.cumsum(R); dd=np.maximum.accumulate(c)-c; return R.sum()/dd.max() if dd.max()>1e-9 else np.inf
bR=load("scratchpad/mkt_trades.csv")["R"].values
rng=np.random.default_rng(0); T=5000
print(f"market base: n={len(bR)} meanR={bR.mean():+.3f} ret/DD={retdd(bR):+.2f}")
print(f"{'retest':>7}{'n':>6}{'meanR':>8}{'ret/DD':>8}{'rd-pctile':>11}")
for w in [5,10,15,20,30,40]:
    kR=load(f"scratchpad/rt_{w}.csv")["R"].values if w!=20 else load("scratchpad/rt20_trades.csv")["R"].values
    nul=np.array([retdd(bR[np.sort(rng.choice(len(bR),len(kR),replace=False))]) for _ in range(T)])
    obs=retdd(kR); pct=(nul<obs).mean()*100
    print(f"{w:>7}{len(kR):>6}{kR.mean():>+8.3f}{obs:>+8.2f}{pct:>10.0f}%")
print("(plateau across windows = real; lone spike = overfit)")
