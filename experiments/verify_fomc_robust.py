"""Re-judge the gold FOMC 1-min scalp on ROBUST statistics instead of the mean.

The mean (+1.97 $/oz) is carried by one event (2026-06-17 = 78.5% of the total sum) and its
standard error is 1.74 -- i.e. 1.13 sigma from zero. Everything the ledger claims for this
leg (null 100th percentile, IS/OOS both positive) was read off that mean. This asks the same
question with statistics a single event cannot move:

  1. bootstrap CI of the mean AND the median (10k resamples)
  2. the null percentile recomputed for the mean, the median, and with the top event removed
  3. win rate vs the null's win rate (a sign-style test that ignores magnitude)

Documented setting throughout: w_c=1, H=5, frac=0.50, cost $0.30/oz, F0 fill (as published).
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from experiments.event_scalp import build_scalp_table, null_scalp_table, COST_ROUNDTRIP
from experiments.event_scalp_cond import threshold_subset

COST = COST_ROUNDTRIP["GOLD"]["base"]
B = 10000
rng = np.random.default_rng(42)

ev = pd.read_csv("experiments/fomc_stmt_2019.csv", parse_dates=["dt_utc", "dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())
df = load_mt5_csv("data/vantage_xauusd_m1.csv")

real = build_scalp_table(df, events, 1, [5, 10, 15], "rob")
null = null_scalp_table(df, events, 1, [5, 10, 15], "rob", draws_target=3000)
sub, thr = threshold_subset(real, "confirm_move_atr", 0.50)
nsub, _ = threshold_subset(null, "confirm_move_atr", 0.50)

net = (sub["g_5"].dropna() - COST).values
nnet = (nsub["g_5"].dropna() - COST).values
n = len(net)
drop_i = np.argmax(np.abs(net))
net_wo = np.delete(net, drop_i)

def boot(x, stat, b=B):
    idx = rng.integers(0, len(x), size=(b, len(x)))
    return np.sort(stat(x[idx], axis=1))

def report(x, label):
    bm = boot(x, np.mean); bd = boot(x, np.median)
    print(f"\n[{label}]  n={len(x)}")
    print(f"  平均  {x.mean():+7.3f}  95%CI [{bm[int(.025*B)]:+7.3f}, {bm[int(.975*B)]:+7.3f}]  "
          f"P(平均≤0)={np.mean(bm <= 0)*100:5.1f}%")
    print(f"  中央値{np.median(x):+7.3f}  95%CI [{bd[int(.025*B)]:+7.3f}, {bd[int(.975*B)]:+7.3f}]  "
          f"P(中央値≤0)={np.mean(bd <= 0)*100:5.1f}%")
    print(f"  標準偏差 {x.std(ddof=1):6.2f}   勝率 {np.mean(x > 0)*100:5.1f}%   "
          f"標準誤差 {x.std(ddof=1)/np.sqrt(len(x)):5.2f}")

report(net, "全29件（台帳の設定）")
report(net_wo, "2026-06-17 を除く28件")

# --- null comparison on statistics a single event cannot move -----------------
print(f"\n[同時刻ランダムnull との照合]  null プール n={len(nnet)}  "
      f"(平均{nnet.mean():+.3f} / 中央値{np.median(nnet):+.3f})")
nd = rng.integers(0, len(nnet), size=(B, n))
null_means = np.mean(nnet[nd], axis=1)
null_meds = np.median(nnet[nd], axis=1)
null_wins = np.mean(nnet[nd] > 0, axis=1)
for lab, x in [("全29件", net), ("2026-06-17を除く28件", net_wo)]:
    k = len(x)
    ndk = rng.integers(0, len(nnet), size=(B, k))
    nm = np.mean(nnet[ndk], axis=1); nmd = np.median(nnet[ndk], axis=1); nw = np.mean(nnet[ndk] > 0, axis=1)
    print(f"  {lab}: 平均%ile={np.mean(nm < x.mean())*100:5.1f}  "
          f"中央値%ile={np.mean(nmd < np.median(x))*100:5.1f}  "
          f"勝率%ile={np.mean(nw < np.mean(x > 0))*100:5.1f}")
