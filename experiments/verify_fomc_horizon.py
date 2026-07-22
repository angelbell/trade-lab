"""Does holding longer turn the FOMC direction call into a tradeable SIZE?

Established by verify_fomc_robust.py: at H=5 the direction call is real (median and win rate
sit at the 100th percentile of the same-time random null even after deleting the one event
that carries the mean), but the size is 2-5 basis points on a $4,380 instrument -- about
150-300 yen per trade at the user's fixed 0.01 lot (= 1 oz). The edge is in WHETHER, not in
HOW MUCH.

So: extend the horizon. Law 9 says far targets are optimal for trend legs; the exit law says
only exit when the exit price beats the expectation from that point. Judge on the MEDIAN with
the outlier deleted, since the mean is not usable at n=29.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from experiments.event_scalp import build_scalp_table, null_scalp_table, COST_ROUNDTRIP
from experiments.event_scalp_cond import threshold_subset

COST = COST_ROUNDTRIP["GOLD"]["base"]
HSET = [5, 10, 15, 30, 60, 120, 240, 480]
B = 10000
JPY = 150.0
rng = np.random.default_rng(42)

ev = pd.read_csv("experiments/fomc_stmt_2019.csv", parse_dates=["dt_utc", "dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())
df = load_mt5_csv("data/vantage_xauusd_m1.csv")

real = build_scalp_table(df, events, 1, HSET, "hz")
null = null_scalp_table(df, events, 1, HSET, "hz", draws_target=3000)
sub, _ = threshold_subset(real, "confirm_move_atr", 0.50)
nsub, _ = threshold_subset(null, "confirm_move_atr", 0.50)
sub = sub.sort_values("t0").reset_index(drop=True)

# the event that carries the mean at H=5 -- delete the SAME event at every horizon
drop_t0 = sub.loc[(sub["g_5"] - COST).abs().idxmax(), "t0"]
print(f"外れ値として除く事象: {drop_t0.date()}   （全ホライズンで同じ1件を除く）")
print(f"{'H(分)':>6} {'n':>3} | {'平均':>7} {'中央値':>7} {'勝率%':>6} {'中央値%ile':>9} {'勝率%ile':>8} "
      f"| {'除外後 中央値':>12} {'勝率%':>6} {'中央値%ile':>9} {'円/トレード':>10}")

for h in HSET:
    g = sub[["t0", f"g_{h}"]].dropna()
    if len(g) < 10:
        print(f"{h:>6} {len(g):>3} | データ不足（決済時刻が未カバー）")
        continue
    net = (g[f"g_{h}"] - COST).values
    gw = g[g["t0"] != drop_t0]
    net_wo = (gw[f"g_{h}"] - COST).values
    nn = (nsub[f"g_{h}"].dropna() - COST).values

    def pct(x, stat):
        d = rng.integers(0, len(nn), size=(B, len(x)))
        return np.mean(stat(nn[d], axis=1) < stat(x)) * 100

    med_p = pct(net, np.median)
    win_p = np.mean(np.mean(nn[rng.integers(0, len(nn), size=(B, len(net)))] > 0, axis=1)
                    < np.mean(net > 0)) * 100
    med_p_wo = pct(net_wo, np.median)
    print(f"{h:>6} {len(net):>3} | {net.mean():+7.2f} {np.median(net):+7.2f} "
          f"{np.mean(net > 0)*100:6.1f} {med_p:9.1f} {win_p:8.1f} "
          f"| {np.median(net_wo):+12.2f} {np.mean(net_wo > 0)*100:6.1f} {med_p_wo:9.1f} "
          f"{np.median(net_wo)*JPY:10.0f}")
