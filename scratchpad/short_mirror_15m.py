"""A8: SHORT mirror of the 15m breakout legs via price inversion (p' = C - p).
Long machinery on inverted bars == short on real bars: downward Pattern-B break,
RALLY-limit entry (pullback-limit mirror), stop above the lower-high, RR4 down-target.
Gates (mirrored automatically by inversion): gold SMA150-below+falling / BTC KAMA-falling
(1D and 4h). PDL air = mirror of PDH (entry below prior-day low). ext-cap disabled (ratio
breaks under inversion; noted). Net $0.3 gold / $15 BTC post-hoc. Cells: ungated base ->
gated -> gated&PDL-air. Pre-registered kill: base gross <= 0."""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radar_gate_race import BASE
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample

def invert(d):
    C = 2 * d["high"].max()
    return pd.DataFrame({"open": C - d["open"], "high": C - d["low"],
                         "low": C - d["high"], "close": C - d["close"]}, index=d.index)

def card(tag, Rn, ts, span):
    if len(Rn) < 20: print(f"  {tag:<26} n={len(Rn)} few"); return
    yr = ts.year.values; half = np.median(yr)
    pf = Rn[Rn>0].sum()/abs(Rn[Rn<=0].sum())
    eq = np.cumsum(Rn); dd = (np.maximum.accumulate(eq)-eq).max()
    g = sum(Rn[yr==y].sum() > 0 for y in np.unique(yr))
    print(f"  {tag:<26} N/yr={len(Rn)/span:5.1f} win={(Rn>0).mean()*100:4.1f}% PF={pf:4.2f} "
          f"meanR={Rn.mean():+.3f} IS/OOS={Rn[yr<half].mean():+.2f}/{Rn[yr>=half].mean():+.2f} "
          f"totR/yr={Rn.sum()/span:+5.1f} DD={dd:5.1f}R grn={g}/{len(np.unique(yr))}")

for name, csv, rt, frac, gates in [
    ("GOLD 15m SHORT", "data/vantage_xauusd_m5.csv", 0.3, 0.25,
     [("SMA150下&下向き", dict(daily_sma=150, daily_slope_k=10))]),
    ("BTC 15m SHORT", "data/vantage_btcusd_m15.csv", 15.0, 0.3,
     [("KAMA1D下向き", dict(gate_kama=14)), ("KAMA4h下向き", dict(gate_kama=14, gate_kama_tf="240min"))])]:
    d = load_mt5_csv(csv)
    cnt = d.groupby(d.index.date).size()
    ok = cnt[cnt.rolling(30).median() >= (150 if "m5" in csv else 80)]
    d = d[d.index.date >= ok.index[0]]
    d15 = resample(d, "15min")
    inv = invert(d15)
    span = (d15.index[-1]-d15.index[0]).days/365.25
    pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
    print(f"\n===== {name} ({span:.1f}yr, rally-limit frac{frac}, net ${rt}) =====")
    t0 = run(inv, SimpleNamespace(**{**BASE, "pullback_frac": frac}))
    R0 = t0["R"].values - rt/t0["risk"].values
    card("base 全シグナル (ゲート無)", R0, pd.DatetimeIndex(t0["time"]), span)
    gross = t0["R"].values
    print(f"    (gross meanR={gross.mean():+.3f} -- 事前登録: これが<=0なら即死)")
    for gtag, extra in gates:
        t = run(inv, SimpleNamespace(**{**BASE, **extra, "pullback_frac": frac}))
        Rn = t["R"].values - rt/t["risk"].values
        ts = pd.DatetimeIndex(t["time"])
        card(f"gate: {gtag}", Rn, ts, span)
        C = 2 * d15["high"].max()
        e_real = C - t["e_px"].values          # back to real price space
        m = (e_real < pdl[d15.index.get_indexer(t["time"])])
        card(f"  ∩ PDL新安値圏", Rn[m], ts[m], span)
