"""a3_time_stop.py -- proposals A3 STEP1 (diagnosis only): do slot-occupying laggards pay?

For each trade of the two 15m legs: bars from entry-fill to FIRST touch of +1R (high >=
e_px + risk). The time-stop candidate group at horizon K = trades still open past K bars
that have NOT touched +1R by K. If that group's FINAL meanR ~ 0 or negative, cutting them
at K is free-or-better AND frees the busy slot (rotation -> N up). If clearly positive
(late bloomers), A3 dies. Report per K: group size, final R mean/median/sd, occupied
bars freed. No engine change yet -- pure diagnosis on the existing trade set.
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

KS = (4, 8, 16, 32, 64, 96, 192)   # 15m bars: 8=2h, 32=8h, 96=1day, 192=2days


def diagnose(name, d15, cell, rt):
    t = run(d15, SimpleNamespace(**cell))
    Rn = t["R"].values - rt / t["risk"].values
    span = (d15.index[-1] - d15.index[0]).days / 365.25
    h = d15["high"].values
    idx = d15.index
    e_bar = idx.get_indexer(pd.DatetimeIndex(t["time"]))
    assert (e_bar >= 0).all()
    exit_t = pd.DatetimeIndex(t["time"]) + pd.to_timedelta(t["hold"].values, unit="D")
    x_bar = np.searchsorted(idx.values, exit_t.values)          # first bar >= exit time
    lvl = t["e_px"].values + t["risk"].values                    # +1R price level
    n = len(t)
    reach = np.full(n, 10**9)                                    # bars to first +1R touch
    for i in range(n):
        j0, j1 = e_bar[i] + 1, min(x_bar[i] + 1, len(h))
        seg = h[j0:j1]
        w = np.nonzero(seg >= lvl[i])[0]
        if len(w):
            reach[i] = w[0] + 1
    dur = x_bar - e_bar                                          # trade duration in bars
    print(f"\n===== {name} ({span:.1f}yr)  n={n}  meanR(net)={Rn.mean():+.3f} =====")
    print(f"  +1R到達までのバー数: med={np.median(reach[reach<10**9]):.0f} "
          f"到達率={(reach<10**9).mean()*100:.0f}%  保有バー med={np.median(dur):.0f}")
    print(f"  {'K':>4} {'未達&生存n':>9} {'%':>4} {'最終meanR':>9} {'med':>6} {'sd':>5} "
          f"{'解放バー/枠占有%':>14}")
    tot_occ = dur.sum()
    cl = d15["close"].values
    for K in KS:
        m = (dur > K) & (reach > K)
        if m.sum() < 10:
            print(f"  {K:>4} {m.sum():>9} few"); continue
        freed = (dur[m] - K).sum()
        jK = np.minimum(e_bar[m] + K, len(cl) - 1)
        mark = (cl[jK] - t["e_px"].values[m]) / t["risk"].values[m]   # R if cut at K close
        forfeit = (t["R"].values[m] - mark).mean()                     # what cutting gives up
        print(f"  {K:>4} {m.sum():>9} {m.mean()*100:>3.0f}% {Rn[m].mean():>+9.3f} "
              f"{np.median(Rn[m]):>+6.2f} {Rn[m].std():>5.2f} {freed/tot_occ*100:>13.1f}%"
              f"  markR={mark.mean():+.2f} 切ると1本あたり{forfeit:+.2f}R放棄")


def main():
    b = load_mt5_csv("data/vantage_btcusd_m15.csv")
    cnt = b.groupby(b.index.date).size()
    ok = cnt[cnt.rolling(30).median() >= 80]
    d15 = resample(b[b.index.date >= ok.index[0]], "15min")
    diagnose("BTC 15m long (kama4h, frac0.3, net $15)", d15,
             {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3}, 15.0)

    g = load_mt5_csv("data/vantage_xauusd_m5.csv").loc["2018-09-14":]
    g15 = resample(g, "15min")
    diagnose("GOLD 15m canon (SMA150+extcap8, frac0.25, net $0.3)", g15,
             {**BASE, "daily_sma": 150, "daily_slope_k": 10, "ext_cap": 8.0,
              "pullback_frac": 0.25}, 0.3)


if __name__ == "__main__":
    main()
