"""Tier-2: (1) component differentiation at a TIGHTER matched ON% (stack==3 level) where
thresholds actually bite; (2) per-year + IS/OOS of the direction-gate winners vs existing
gates; (3) combos: BTC kama4h&stack4h, gold stack4h&extcap8."""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radar_gate_race import (BASE, comps_tf, kama_up, sma_gate, matched, cell,
                             load_mt5_csv, resample, run)

def peryear(tag, Rn, yr):
    ys = sorted(np.unique(yr))
    print(f"    {tag:<24} " + " ".join(f"{y}:{Rn[yr==y].sum():+.0f}" for y in ys))
    half = ys[len(ys)//2]
    is_, oos = Rn[yr < half], Rn[yr >= half]
    print(f"    {'':<24} IS meanR={is_.mean():+.3f}(n={len(is_)})  OOS={oos.mean():+.3f}(n={len(oos)})")

for name, csv, frac, rt, start in [("GOLD","data/vantage_xauusd_m15.csv",0.25,0.6,None),
                                   ("BTC","data/vantage_btcusd_m15.csv",0.3,15.0,"2018-10-01")]:
    d = load_mt5_csv(csv)
    if start: d = d[d.index >= start]
    d15 = resample(d, "15min")
    span = (d15.index[-1]-d15.index[0]).days/365.25
    t = run(d15, SimpleNamespace(**BASE, pullback_frac=frac))
    Rn = t["R"].values - rt/t["risk"].values
    yr = t["time"].dt.year.values
    pos = d15.index.get_indexer(t["time"])
    C4 = comps_tf(d15, "240min"); up4 = C4["stack"] > 0
    full = C4["stack"] >= 3
    target = full.mean()          # tier-2 selectivity = stack==3 level
    print(f"\n===== {name}: 成分分解 tier-2 (ON%={target*100:.0f}%に一致, {span:.1f}yr) =====")
    rows = {"stack4h==3 (基準)": full}
    for ck, lab in [("er","ER"),("adx","ADX"),("slope","slope"),("atrexp","ATRexp"),("s10","合成s")]:
        rows[f"up4h&{lab}"] = matched(up4, C4[ck], target)
    if name == "BTC":
        rows["kama4h&stack4h>0"] = kama_up(d15,"240min") & up4
        rows["kama4h&1d (参考)"] = kama_up(d15,"240min") & kama_up(d15,"1D")
    else:
        rows["stack4h>0&extcap8"] = up4 & sma_gate(d15,150,10,cap=8.0)  # extcap needs sma-gate ON too
        rows["up4h&ER&extcap8"] = matched(up4, C4["er"], target) & sma_gate(d15,150,10,cap=8.0)
    for tag, g in rows.items():
        m = g[pos]
        cell(f"{tag} [ON{g.mean()*100:3.0f}%]", Rn[m], yr[m], span)
    print(f"  --- 年別・IS/OOS（向きゲート vs 既存） ---")
    if name == "GOLD":
        g_ex = sma_gate(d15,150,10)[pos]; g_dir = up4[pos]
        peryear("SMA150d+slope(既存)", Rn[g_ex], yr[g_ex])
        peryear("stack4h>0", Rn[g_dir], yr[g_dir])
    else:
        g_ex = kama_up(d15,"240min")[pos]; g_dir = up4[pos]
        peryear("kama4h(C1)", Rn[g_ex], yr[g_ex])
        peryear("stack4h>0", Rn[g_dir], yr[g_dir])
        g_c = (kama_up(d15,"240min") & up4)[pos]
        peryear("kama4h&stack4h", Rn[g_c], yr[g_c])
