"""Fear & Greed as a BTC deploy gate: (1) redundancy vs KAMA states (the DXY-death test),
(2) bucket labeling of the BTC 15m leg (kama4h variant) by PRIOR-day F&G, (3) veto test
(skip Extreme Greed / skip Extreme Fear) vs equal-keep random-drop null. Causal: F&G[t-1]."""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radar_gate_race import BASE, kama_up
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample

fj = json.load(open("/tmp/claude-1000/-home-angelbell-dev-auto-trade/a84f0ce0-f380-45b1-956c-321740d60f31/scratchpad/fng.json"))["data"]
fng = pd.Series({pd.Timestamp(int(r["timestamp"]), unit="s", tz="UTC").normalize(): float(r["value"])
                 for r in fj}).sort_index()
fng.to_csv("data/ext_feargreed.csv")
d = load_mt5_csv("data/vantage_btcusd_m15.csv").loc["2018-10-01":]
d15 = resample(d, "15min")
span = (d15.index[-1]-d15.index[0]).days/365.25
fg = fng.shift(1).reindex(d15.index, method="ffill").values     # prior day's value

k1, k4 = kama_up(d15, "1D"), kama_up(d15, "240min")
m = ~np.isnan(fg)
print("(1) redundancy: state overlap on all bars")
for tag, g in [("kama1d", k1), ("kama4h", k4)]:
    for th, lab in [(50, "F&G>=50"), (75, ">=75"), (25, "<25")]:
        s = (fg >= th) if th != 25 else (fg < 25)
        both = (g & s & m).sum()
        print(f"   {tag} vs {lab}: P({lab}|gateON)={both/(g&m).sum():.2f} "
              f"P(gateON|{lab})={both/max((s&m).sum(),1):.2f} phi={np.corrcoef(g[m], s[m])[0,1]:+.2f}")

t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3}))
Rn = t["R"].values - 15.0/t["risk"].values
pos = d15.index.get_indexer(t["time"])
f_e = fg[pos]
yr = t["time"].dt.year.values
print(f"\n(2) BTC15m kama4h leg (N={len(Rn)}) by prior-day F&G bucket:")
for lab, mm in [("<25 extreme fear", f_e < 25), ("25-50 fear", (f_e >= 25) & (f_e < 50)),
                ("50-75 greed", (f_e >= 50) & (f_e < 75)), (">=75 extreme greed", f_e >= 75)]:
    if mm.sum() < 20: print(f"   {lab:<20} n={mm.sum()} few"); continue
    r = Rn[mm]; pf = r[r>0].sum()/abs(r[r<=0].sum())
    print(f"   {lab:<20} n={mm.sum():4d}({mm.mean()*100:3.0f}%)  meanR={r.mean():+.3f}  PF={pf:4.2f}"
          f"  totR/yr={r.sum()/span:+5.1f}")
def rd(x):
    eq = np.cumsum(x); dd = (np.maximum.accumulate(eq)-eq).max()
    return x.sum()/max(dd, 1e-9)
rng = np.random.default_rng(7)
print("\n(3) veto tests vs equal-keep random-drop null:")
for lab, keep in [("skip >=75 (天井回避)", f_e < 75), ("skip <25 (総悲観回避)", f_e >= 25),
                  ("keep 25-75 (中庸のみ)", (f_e >= 25) & (f_e < 75))]:
    real = rd(Rn[keep])
    nl = [rd(Rn[np.sort(rng.choice(len(Rn), keep.sum(), replace=False))]) for _ in range(1000)]
    r = Rn[keep]
    print(f"   {lab:<22} n={keep.sum():4d}  meanR={r.mean():+.3f}  totR/yr={r.sum()/span:+5.1f}"
          f"  totR/DD={real:5.2f}  null%ile={(np.array(nl)<real).mean()*100:3.0f}")
