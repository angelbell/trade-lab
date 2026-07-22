"""Can this be traded by someone asleep in Japan, and what stop does a 4-hour hold need?

The statement lands at NY 14:00 = 03:00 JST (EU summer) / 04:00 JST (winter), so the entry is
the middle of the night for the user and the +4h exit is 07:00-08:00 JST. Two consequences the
spec ignored:

  1. holding unhedged while asleep makes a stop mandatory, not optional -- and the stop widths
     tested for the 1-minute scalp are useless here (they were sized off the pre-release 1-min
     ATR, which is far too tight for a 4-hour hold);
  2. if a meaningful part of the drift sits in the LATER hours, there may be a version that
     starts after the user wakes up (+4h = 07:00 JST) and needs no night-time order at all.

So: (a) a segment matrix -- entry offset x hold -- to locate where in the window the move is,
and (b) stops on the base trade, sized off the observed MAE distribution.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from scratchpad.fomc_event_study import price_before, candidate_dates

COST = 0.30
JPY = 150.0
B = 10000
rng = np.random.default_rng(42)

ev = pd.read_csv("scratchpad/fomc_stmt_2019.csv", parse_dates=["dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())
df = load_mt5_csv("data/vantage_xauusd_m1.csv")
T_END = df.index.max()
PX = float(df["close"].iloc[-1])
cand = candidate_dates(df, events)
clock = events[0].time()
ctl_ts = [pd.Timestamp.combine(d.date(), clock).tz_localize("UTC") for d in cand]


def seg(ts, e, h):
    """Long from t0+e minutes, held h minutes. Returns log return after cost."""
    t_in = ts + pd.Timedelta(minutes=e)
    t_out = t_in + pd.Timedelta(minutes=h)
    if t_out > T_END:
        return None
    pe, px = price_before(df, t_in), price_before(df, t_out)
    if pe is None or px is None or not np.isfinite(pe) or not np.isfinite(px) or pe <= 0:
        return None
    return np.log((px - COST) / pe)


def pctile(r, c):
    d = rng.integers(0, len(c), size=(B, len(r)))
    return np.mean(np.median(c[d], axis=1) < np.median(r)) * 100


print(f"gold {PX:.0f}$/oz  0.01ロット=1oz   声明=JST 03:00(夏)/04:00(冬)")
print("\n=== (a) 区間の分解: どこに値幅があるか（中央値% / 勝率% / null%ile / 円）===")
print(f"{'建てる時刻':>22} {'保有':>6} {'n':>3} | {'中央値%':>8} {'勝率%':>6} {'%ile':>6} {'円/回':>7}")
for e, jst in [(1, "JST 03:01/04:01"), (60, "JST 04:00/05:00"), (120, "JST 05:00/06:00"),
               (240, "JST 07:00/08:00"), (360, "JST 09:00/10:00")]:
    for h in [120, 240]:
        r = np.array([x for x in (seg(t, e, h) for t in events) if x is not None])
        c = np.array([x for x in (seg(t, e, h) for t in ctl_ts) if x is not None])
        if len(r) < 5:
            continue
        print(f"{jst:>22} {h:>5}分 {len(r):>3} | {np.median(r)*100:+8.3f} {np.mean(r > 0)*100:6.1f} "
              f"{pctile(r, c):6.1f} {np.median(r)*PX*JPY:7.0f}")

print("\n=== (b) 4時間保有に対する損切り（建値=t0+1分、時間決済240分、同足はストップ優先）===")


def stopped(ts, h, stop_frac):
    t_in = ts + pd.Timedelta(minutes=1)
    t_out = t_in + pd.Timedelta(minutes=h)
    if t_out > T_END:
        return None
    pe = price_before(df, t_in)
    if pe is None or not np.isfinite(pe) or pe <= 0:
        return None
    seg_ = df.loc[t_in:t_out]
    if len(seg_) < 2:
        return None
    if stop_frac is not None:
        lvl = pe * (1 - stop_frac)
        hit = seg_.index[seg_["low"] <= lvl]
        if len(hit):
            return np.log((lvl - COST) / pe), True
    px = price_before(df, t_out)
    if px is None or not np.isfinite(px):
        return None
    return np.log((px - COST) / pe), False


print(f"{'損切り':>10} {'n':>3} | {'中央値%':>8} {'平均%':>7} {'勝率%':>6} {'発動%':>6} "
      f"| {'円/回':>7} {'年間円':>8} {'最悪円':>8}")
span = (events[-1] - events[0]).days / 365.25
for sf in [None, 0.005, 0.010, 0.015, 0.020, 0.030]:
    out = [x for x in (stopped(t, 240, sf) for t in events) if x is not None]
    if not out:
        continue
    r = np.array([o[0] for o in out])
    hit = np.mean([o[1] for o in out]) * 100
    lab = "無し" if sf is None else f"{sf*100:.1f}%"
    print(f"{lab:>10} {len(r):>3} | {np.median(r)*100:+8.3f} {r.mean()*100:+7.3f} "
          f"{np.mean(r > 0)*100:6.1f} {hit:6.1f} | {np.median(r)*PX*JPY:7.0f} "
          f"{np.median(r)*PX*JPY*len(r)/span:8.0f} {r.min()*PX*JPY:8.0f}")

print("\n=== (c) 出口をずらせるか（建値=t0+1分固定、決済時刻だけ動かす）===")
print(f"{'保有':>6} {'JST決済':>16} {'中央値%':>8} {'勝率%':>6} {'円/回':>7}")
for h, jst in [(240, "07:00/08:00"), (300, "08:00/09:00"), (360, "09:00/10:00"),
               (420, "10:00/11:00"), (480, "11:00/12:00"), (600, "13:00/14:00")]:
    r = np.array([x for x in (seg(t, 1, h) for t in events) if x is not None])
    if len(r) < 5:
        continue
    print(f"{h:>5}分 {jst:>16} {np.median(r)*100:+8.3f} {np.mean(r > 0)*100:6.1f} "
          f"{np.median(r)*PX*JPY:7.0f}")
