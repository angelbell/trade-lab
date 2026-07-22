"""H17 (gold 15m Asia-box ORB + 1H EMA80 gate, no-TP, cost 2.8p) improvement screen.
Bleed anatomy + two death-reverse-engineered levers, post-hoc labels on the full-history
trade set (no split -- this is diagnosis, not selection yet):
  L1 open-drive filter: long breaks ALSO above PDH (new-high air) vs inside yesterday's
     range; shorts mirror below PDL. (market-profile open-drive vs rotation)
  L2 asymmetric daily regime: long & daily SMA150-up / short & daily SMA80-falling
     (= merge the H17-S lesson into both sides; the 1h gate stays).
Report: per-side x per-label pips/PF/N + per-year of the winners."""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_mt5_csv
import research.scalp_lab as SL

p = SimpleNamespace(asia_start_h=0, asia_end_h=7, bo_start_h=7, bo_end_h=11, force_exit_h=20,
                    rr=1.0, buf_atr=0.0, sl_buf_atr=0.0, max_range_atr=0.0, min_range_atr=0.0,
                    sl_frac=1.0, rsi_max=100.0, box_trend_max=1.0, no_tp=True, fade=False,
                    dir="both", cost=2.8, htf_tf="1h", htf_ema=80, htf_slope_k=0, stop_slip=0.0,
                    daily_sma=0, daily_slope_k=0,
                    htf_sr=False, trendline=False, htf_level=False, piv_k=6)

d = load_mt5_csv("data/vantage_xauusd_m15.csv")
dir_, sl_px, tp_px = SL.orb_signals(d, p)
# 1h trend gate (same as scalp_lab's htf gate)
chtf = d["close"].resample("1h", label="left", closed="left").last().dropna()
ema = chtf.ewm(span=80, adjust=False).mean()
up = (chtf > ema).shift(1).reindex(d.index, method="ffill").fillna(False).values
dn = (chtf < ema).shift(1).reindex(d.index, method="ffill").fillna(False).values
dir_ = np.where((dir_ == 1) & up, 1, np.where((dir_ == -1) & dn, -1, 0)).astype(np.int8)
tr = SL.backtest(d, dir_, sl_px, tp_px, p)
span = (tr.t_in.max() - tr.t_in.min()).days / 365.25
c15 = d["close"]
pdh = d["high"].resample("1D").max().dropna().shift(1).reindex(d.index, method="ffill")
pdl = d["low"].resample("1D").min().dropna().shift(1).reindex(d.index, method="ffill")
dc = d["close"].resample("1D").last().dropna()
s150 = dc.rolling(150).mean(); s80 = dc.rolling(80).mean()
d_up = ((dc > s150) & (s150 > s150.shift(10))).shift(1).reindex(d.index, method="ffill").fillna(False)
d_dn = (s80 < s80.shift(10)).shift(1).reindex(d.index, method="ffill").fillna(False)

e_px = c15.reindex(tr.t_in).values
air = np.where(tr.dir.values == 1, e_px > pdh.reindex(tr.t_in).values,
               e_px < pdl.reindex(tr.t_in).values)
reg = np.where(tr.dir.values == 1, d_up.reindex(tr.t_in).values, d_dn.reindex(tr.t_in).values)
def card(tag, m):
    if m.sum() < 15: print(f"  {tag:<34} n={m.sum()} few"); return
    x = tr.pips.values[m]
    pf = x[x>0].sum()/abs(x[x<=0].sum())
    print(f"  {tag:<34} N/yr={m.sum()/span:5.1f}  PF={pf:4.2f}  pips/tr={x.mean():+6.1f}  tot/yr={x.sum()/span:+7.0f}")
print(f"H17 full-history {tr.t_in.min().date()}->{tr.t_in.max().date()} ({span:.1f}yr) n={len(tr)}")
for side, sname in [(1, "LONG"), (-1, "SHORT")]:
    s = tr.dir.values == side
    card(f"{sname} 全体", s)
    card(f"{sname} 空中戦（PDH/PDL外）", s & air)
    card(f"{sname} レンジ内", s & ~air)
    card(f"{sname} ∩ 日足レジーム順", s & reg)
    card(f"{sname} 空中戦∩レジーム順", s & air & reg)
print("\nper-year pips (LONG全体 / LONG空中戦 / SHORT全体 / SHORT空中戦):")
yr = tr.t_in.dt.year.values
for y in np.unique(yr):
    m = yr == y
    L, La = m & (tr.dir.values==1), m & (tr.dir.values==1) & air
    S, Sa = m & (tr.dir.values==-1), m & (tr.dir.values==-1) & air
    print(f"  {y}: {tr.pips.values[L].sum():+6.0f}({L.sum():3d}) / {tr.pips.values[La].sum():+6.0f}({La.sum():3d})"
          f" / {tr.pips.values[S].sum():+6.0f}({S.sum():3d}) / {tr.pips.values[Sa].sum():+6.0f}({Sa.sum():3d})")

# ---- N-lever: a SECOND daily box (Europe box 7-13h -> NY break 13-17h), same 1h gate ----
print("\n===== second session: Europe box[7,13) -> NY break[13,17), exit 22h =====")
p2 = SimpleNamespace(**{**vars(p), "asia_start_h": 7, "asia_end_h": 13,
                        "bo_start_h": 13, "bo_end_h": 17, "force_exit_h": 22})
dir2, sl2, tp2 = SL.orb_signals(d, p2)
dir2 = np.where((dir2 == 1) & up, 1, np.where((dir2 == -1) & dn, -1, 0)).astype(np.int8)
tr2 = SL.backtest(d, dir2, sl2, tp2, p2)
for side, sname in [(1, "LONG"), (-1, "SHORT")]:
    s = tr2.dir.values == side
    x = tr2.pips.values[s]
    if s.sum() >= 15:
        pf = x[x>0].sum()/abs(x[x<=0].sum())
        print(f"  NY {sname:<6} N/yr={s.sum()/span:5.1f}  PF={pf:4.2f}  pips/tr={x.mean():+6.1f}  tot/yr={x.sum()/span:+7.0f}")
yr2 = tr2.t_in.dt.year.values
print("  per-year: " + "  ".join(f"{y}:{tr2.pips.values[yr2==y].sum():+.0f}" for y in np.unique(yr2)))
# overlap with the Asia leg (same-day double exposure)
d1 = set(tr.t_in.dt.date); d2 = set(tr2.t_in.dt.date)
print(f"  same-day overlap with Asia-ORB: {len(d1&d2)}/{len(d2)} ({len(d1&d2)/max(len(d2),1)*100:.0f}%)")
