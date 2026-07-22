"""PDH selector at 5m: the mechanism says it should discriminate MORE (even more entries
trigger inside yesterday's range at 5m) -- but must clear the 5m cost+slip wall.
GOLD 5m: canonical gauntlet walk ($0.3 + slip0.27), entries PRE-filtered by e>PDH
(deployment-faithful: the freed no-overlap slots re-arm) + equal-count RANDOM-entry null.
BTC 5m: breakout_wave leg (kama1d, frac0.3), post-hoc label, $10/$15."""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------- GOLD 5m ----------
src = open("experiments/pullback_5m_realcost.py").read()
ns = {}
exec(src.split("\nB5r4  = build")[0], ns)
build, eval_pull, stats = ns["build"], ns["eval_pull"], ns["stats"]
B = build(None, 4.0)
d5 = B["d"]
span = (d5.index[-1] - d5.index[0]).days / 365.25
dh = d5["high"].resample("1D").max().dropna()
pdh = dh.shift(1).reindex(d5.index, method="ffill").values

def card(tag, tr):
    s = stats(tr, span)
    if s is None: print(f"  {tag:<22} too few"); return None
    print(f"  {tag:<22} N={s['N']:4d} N/yr={s['npy']:5.1f} win={s['win']:4.1f}% PF={s['pf']:4.2f} "
          f"meanR={s['meanR']:+.3f} IS/OOS={s['IS']:+.2f}/{s['OOS']:+.2f} "
          f"totR/yr={s['totR']/span:+5.1f} DD={s['dd']:5.1f}R ret/DD={s['retdd']:5.2f} grn={s['grn']}/{s['ny']}")
    return s

lab = np.array([e > pdh[i] and not np.isnan(pdh[i]) for (i, e, *_ ) in B["entries"]])
print(f"GOLD 5m ($0.3+slip0.27): entries={len(B['entries'])}, e>PDH={lab.sum()} ({lab.mean()*100:.0f}%)")
FR = lambda e, s, H: e - 0.25 * (e - s)
above = dict(B); above["entries"] = [en for en, L in zip(B["entries"], lab) if L]
below = dict(B); below["entries"] = [en for en, L in zip(B["entries"], lab) if not L]
tr_a, _ = eval_pull(above, FR, 0.3, stop_slip=0.27)
tr_b, _ = eval_pull(below, FR, 0.3, stop_slip=0.27)
tr_full, _ = eval_pull(B, FR, 0.3, stop_slip=0.27)
card("full leg", tr_full); sa = card("e>PDH (新値圏)", tr_a); card("e<=PDH (レンジ内)", tr_b)
rng = np.random.default_rng(7)
nulls = []
for _ in range(100):
    idx = np.sort(rng.choice(len(B["entries"]), lab.sum(), replace=False))
    Bx = dict(B); Bx["entries"] = [B["entries"][k] for k in idx]
    trx, _ = eval_pull(Bx, FR, 0.3, stop_slip=0.27)
    sx = stats(trx, span)
    nulls.append(sx["retdd"] if sx else 0.0)
nulls = np.array(nulls)
print(f"  random-entry null (equal count, 100 trials): ret/DD med={np.median(nulls):.2f} "
      f"sd={nulls.std():.2f}  e>PDH %ile={(nulls < sa['retdd']).mean()*100:.0f}")

# ---------- BTC 5m ----------
from radar_gate_race import BASE
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
d = load_mt5_csv("data/vantage_btcusd_m5.csv")
cnt = d.groupby(d.index.date).size()
ok = cnt[cnt.rolling(30).median() >= 150]
d5b = resample(d[d.index.date >= ok.index[0]], "5min")
spanb = (d5b.index[-1] - d5b.index[0]).days / 365.25
t = run(d5b, SimpleNamespace(**{**BASE, "tf": "5min", "gate_kama": 14, "pullback_frac": 0.3}))
dhb = d5b["high"].resample("1D").max().dropna()
pdhb = dhb.shift(1).reindex(d5b.index, method="ffill").values
pos = d5b.index.get_indexer(t["time"])
e = t["e_px"].values
m = (e > pdhb[pos]) & ~np.isnan(pdhb[pos])
yr = t["time"].dt.year.values; half = np.median(yr)
print(f"\nBTC 5m: N={len(t)}, e>PDH={m.sum()} ({m.mean()*100:.0f}%)")
for rt in (10.0, 15.0):
    Rn = t["R"].values - rt/t["risk"].values
    for tag, mm in [("full", np.ones(len(Rn), bool)), ("e>PDH", m), ("e<=PDH", ~m)]:
        r = Rn[mm]; pf = r[r>0].sum()/abs(r[r<=0].sum())
        eq = np.cumsum(r); dd = (np.maximum.accumulate(eq)-eq).max()
        print(f"  ${rt:.0f} {tag:<7} n={mm.sum():4d} N/yr={mm.sum()/spanb:5.1f} PF={pf:4.2f} "
              f"meanR={r.mean():+.3f} IS/OOS={Rn[mm&(yr<half)].mean():+.2f}/{Rn[mm&(yr>=half)].mean():+.2f} "
              f"totR/yr={r.sum()/spanb:+5.1f} DD={dd:5.1f}R ret/DD={r.sum()/dd:5.2f}")
    def rd(x):
        eq = np.cumsum(x); dd = (np.maximum.accumulate(eq)-eq).max()
        return x.sum()/max(dd,1e-9)
    real = rd(Rn[m])
    nl = [rd(Rn[np.sort(rng.choice(len(Rn), m.sum(), replace=False))]) for _ in range(1000)]
    print(f"  ${rt:.0f} e>PDH equal-keep null %ile = {(np.array(nl) < real).mean()*100:.0f}")
