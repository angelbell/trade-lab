"""Is "hold through the press conference" real, or is it just gold's uptrend?

verify_fomc_horizon.py found a dip at H=30-60 (exactly the 14:30 presser) and a recovery to
median +2.9 .. +4.8 $/oz at H=120-480, at the 100th percentile of the same-time random null
even after deleting the one event that carries the mean. Two ways that could be fake:

  1. BETA (falsifier 10): if most statements produce a LONG signal, an 8-hour hold in a gold
     bull market is drift, not edge. -> split by side, and report each side against the null.
  2. FREQUENCY / in-sample threshold: the frac=0.50 cut was chosen in-sample and gives only
     ~4 trades/yr. If the effect needs no threshold, frequency doubles AND the caveat dies.

Median-based throughout (n=29, the mean is unusable), same single outlier deleted everywhere.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from experiments.event_scalp import build_scalp_table, null_scalp_table, COST_ROUNDTRIP
from experiments.event_scalp_cond import threshold_subset

COST = COST_ROUNDTRIP["GOLD"]["base"]
HSET = [5, 120, 240, 480]
FRACS = [1.00, 0.50, 0.33]
B = 10000
JPY = 150.0
rng = np.random.default_rng(42)

ev = pd.read_csv("experiments/fomc_stmt_2019.csv", parse_dates=["dt_utc", "dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())
df = load_mt5_csv("data/vantage_xauusd_m1.csv")

real = build_scalp_table(df, events, 1, HSET, "ht")
null = null_scalp_table(df, events, 1, HSET, "ht", draws_target=3000)
s50, _ = threshold_subset(real, "confirm_move_atr", 0.50)
drop_t0 = s50.loc[(s50["g_5"] - COST).abs().idxmax(), "t0"]

def med_pct(x, nn):
    d = rng.integers(0, len(nn), size=(B, len(x)))
    return np.mean(np.median(nn[d], axis=1) < np.median(x)) * 100

print(f"外れ値 {drop_t0.date()} を全セルで除外。null=同時刻ランダム日（同じしきい値を適用）\n")
print("=== 1. しきい値ラダー（頻度と in-sample 問題）===")
print(f"{'frac':>5} {'H':>4} {'n':>3} {'本/年':>6} | {'中央値':>7} {'勝率%':>6} {'中央値%ile':>9} {'円/回':>7}")
span = (real["t0"].max() - real["t0"].min()).days / 365.25
for frac in FRACS:
    sub, _ = threshold_subset(real, "confirm_move_atr", frac)
    nsub, _ = threshold_subset(null, "confirm_move_atr", frac) if frac < 1 else (null, None)
    sub = sub[sub["t0"] != drop_t0]
    for h in HSET:
        g = sub[f"g_{h}"].dropna().values - COST
        nn = nsub[f"g_{h}"].dropna().values - COST
        print(f"{frac:>5.2f} {h:>4} {len(g):>3} {len(g)/span:>6.1f} | {np.median(g):+7.2f} "
              f"{np.mean(g > 0)*100:6.1f} {med_pct(g, nn):9.1f} {np.median(g)*JPY:7.0f}")

print("\n=== 2. ロング/ショート別（反証10: ドリフトに乗っているだけか）===")
print(f"{'frac':>5} {'H':>4} {'側':>4} {'n':>3} | {'中央値':>7} {'勝率%':>6} {'中央値%ile':>9}")
for frac in [1.00, 0.50]:
    sub, _ = threshold_subset(real, "confirm_move_atr", frac)
    nsub, _ = threshold_subset(null, "confirm_move_atr", frac) if frac < 1 else (null, None)
    sub = sub[sub["t0"] != drop_t0]
    for h in [5, 240, 480]:
        for side, lab in [(1, "買"), (-1, "売")]:
            m = sub[sub["d"] == side]
            g = m[f"g_{h}"].dropna().values - COST
            nm = nsub[nsub["d"] == side]
            nn = nm[f"g_{h}"].dropna().values - COST
            if len(g) < 5 or len(nn) < 30:
                print(f"{frac:>5.2f} {h:>4} {lab:>4} {len(g):>3} | n不足")
                continue
            print(f"{frac:>5.2f} {h:>4} {lab:>4} {len(g):>3} | {np.median(g):+7.2f} "
                  f"{np.mean(g > 0)*100:6.1f} {med_pct(g, nn):9.1f}")
