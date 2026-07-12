import numpy as np, pandas as pd, io
def load(f):
    lines = open(f).read().splitlines()
    hdr = next(i for i,l in enumerate(lines) if l.startswith("entry_time"))
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    return df.sort_values("entry_time").reset_index(drop=True)
mkt = load("scratchpad/mkt_trades.csv"); rt = load("scratchpad/rt20_trades.csv")
def retdd(R):
    c = np.cumsum(R); dd = np.maximum.accumulate(c)-c
    return R.sum()/dd.max() if dd.max()>1e-9 else np.inf
bR = mkt["R"].values; kR = rt["R"].values
print(f"market  : n={len(bR)} meanR={bR.mean():+.3f} ret/DD={retdd(bR):+.2f}")
print(f"retest20: n={len(kR)} meanR={kR.mean():+.3f} ret/DD={retdd(kR):+.2f}")
rng = np.random.default_rng(0); n_keep = len(kR); T = 5000
nul_rd = []; nul_mr = []
for _ in range(T):
    idx = np.sort(rng.choice(len(bR), n_keep, replace=False))
    s = bR[idx]; nul_rd.append(retdd(s)); nul_mr.append(s.mean())
nul_rd = np.array(nul_rd); nul_mr = np.array(nul_mr); obs = retdd(kR)
print(f"\nrandom-drop null (keep {n_keep} of {len(bR)} at random, {T} trials):")
print(f"  ret/DD: obs={obs:+.2f}  null med={np.median(nul_rd):+.2f} [5/95={np.percentile(nul_rd,5):+.2f}/{np.percentile(nul_rd,95):+.2f}]  pctile={(nul_rd<obs).mean()*100:.0f}%")
print(f"  meanR : obs={kR.mean():+.3f}  null med={np.median(nul_mr):+.3f}  pctile={(nul_mr<kR.mean()).mean()*100:.0f}%")
print("  (>90%ile = retest adds real selection, not n-trimming)")
