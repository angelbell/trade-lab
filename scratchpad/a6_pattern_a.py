"""a6_pattern_a.py -- proposals A6: Pattern A (break of the last lower high in a downtrend
= reversal wave-1 entry) as a PARALLEL signal set on the 15m cells, + the wave-3/5 split
of Pattern B (does either Elliott leg dominate the current B edge?).

Funnel per spec: A standalone base gross (<=0 = instant death) -> net -> gates ->
overlap/corr vs the B leg -> only if low-overlap AND standalone-positive, combine.
Same machinery (breakout_wave BASE), rally-limit execution, real costs.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE


def card(tag, t, rt, span, gross=False):
    if t is None or len(t) < 20:
        print(f"  {tag:<28} n={0 if t is None else len(t)} few"); return None
    Rn = t["R"].values - rt / t["risk"].values
    ts = pd.DatetimeIndex(t["time"])
    yr = ts.year.values
    half = np.median(yr)
    pf = Rn[Rn > 0].sum() / abs(Rn[Rn <= 0].sum())
    eq = np.cumsum(Rn)
    dd = (np.maximum.accumulate(eq) - eq).max()
    g = sum(Rn[yr == y].sum() > 0 for y in np.unique(yr))
    extra = f"  (gross meanR={t['R'].values.mean():+.3f})" if gross else ""
    print(f"  {tag:<28} N/yr={len(Rn)/span:5.1f} win={(Rn>0).mean()*100:4.1f}% PF={pf:4.2f} "
          f"meanR={Rn.mean():+.3f} IS/OOS={Rn[yr<half].mean():+.2f}/{Rn[yr>=half].mean():+.2f} "
          f"totR/yr={Rn.sum()/span:+5.1f} DD={dd:5.1f}R grn={g}/{len(np.unique(yr))}{extra}")
    return pd.Series(Rn, index=ts)


def overlap(sa, sb):
    da, db = set(sa.index.date), set(sb.index.date)
    ov = len(da & db) / max(1, len(da))
    ma = sa.groupby(sa.index.to_period("M")).sum()
    mb = sb.groupby(sb.index.to_period("M")).sum()
    midx = ma.index.union(mb.index)
    corr = ma.reindex(midx, fill_value=0).corr(mb.reindex(midx, fill_value=0))
    print(f"    -> B曲線との重複: A側の{ov*100:.0f}%の暦日がBと同日 / 月次R相関={corr:+.2f}")


def main():
    for name, csv, rt, frac, minbars, gates in [
        ("BTC 15m", "data/vantage_btcusd_m15.csv", 15.0, 0.3, 80,
         [("kama4h上向き", dict(gate_kama=14, gate_kama_tf="240min"))]),
        ("GOLD 15m", "data/vantage_xauusd_m5.csv", 0.3, 0.25, 150,
         [("SMA150上&上向き", dict(daily_sma=150, daily_slope_k=10)),
          ("SMA150+extcap8", dict(daily_sma=150, daily_slope_k=10, ext_cap=8.0))])]:
        d = load_mt5_csv(csv)
        cnt = d.groupby(d.index.date).size()
        okd = cnt[cnt.rolling(30).median() >= minbars]
        d15 = resample(d[d.index.date >= okd.index[0]], "15min")
        span = (d15.index[-1] - d15.index[0]).days / 365.25
        print(f"\n===== {name} ({span:.1f}yr, rally-limit frac{frac}, net ${rt}) =====")
        tb = run(d15, SimpleNamespace(**{**BASE, "pullback_frac": frac,
                                         **(gates[-1][1] if name == "GOLD 15m" else gates[0][1])}))
        sB = card("[B] 現行レッグ (参照)", tb, rt, span)
        ta_ = run(d15, SimpleNamespace(**{**BASE, "pattern": "A", "pullback_frac": frac}))
        sA = card("[A] base 全シグナル", ta_, rt, span, gross=True)
        if sA is not None and sB is not None:
            overlap(sA, sB)
        for gtag, extra in gates:
            tg = run(d15, SimpleNamespace(**{**BASE, "pattern": "A", "pullback_frac": frac, **extra}))
            sG = card(f"[A] gate: {gtag}", tg, rt, span)
            if sG is not None and sB is not None:
                overlap(sG, sB)
        # wave split of the CURRENT B leg (composition, not a filter proposal)
        for wv, wtag in [("3", "wave-3のみ (初動)"), ("5", "wave-5のみ (継続)")]:
            tw = run(d15, SimpleNamespace(**{**BASE, "pullback_frac": frac, "wave": wv,
                                             **(gates[-1][1] if name == "GOLD 15m" else gates[0][1])}))
            card(f"[B] {wtag}", tw, rt, span)


if __name__ == "__main__":
    main()
