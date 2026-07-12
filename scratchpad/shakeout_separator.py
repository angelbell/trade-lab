"""Search the trodden ground: split gold 15m gated-uptrend pullback-limit STOPOUTS by their
CHARACTER (all causal at/before the stopout bar), and see if the forward recovery separates.
Separators:
  closed_below : stopout bar CLOSED below pL2 (real break) vs wicked-and-held (liquidity sweep)
  wick_depth   : (pL2 - low[xj]) / ATR  (how far below the stop the spike went)
  buffer_pL0   : (pL2 - pL0) / ATR      (structural room down to the deeper prior low; entry-time)
  speed        : bars held from fill to stopout
Forward recovery over K bars after the stopout close (the thing we're trying to separate):
  resumed  : high reached the ORIGINAL target within K
  broke_pL0: low broke pL0 within K  (structural failure = the large trend really broke)
  mfe/mae  : (maxH-c)/ATR vs (c-minL)/ATR
  redeploy : re-enter pullback-limit at first higher close within K (stop=pL2, orig tgt) -> R
No lookahead: separators known by close of xj; recovery is strictly forward. Descriptive first."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas_ta as ta
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}
BO, FWD, FRAC, SP, K = 20, 500, 0.25, 0.6, 50

d = load_mt5_csv("data/vantage_xauusd_m5.csv").resample("15min").agg(AGG).dropna()
o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
atr = ta.atr(d["high"], d["low"], d["close"], 14).values
es = d["close"].ewm(span=80, adjust=False).mean().values
dc = d["close"].resample("1D").last().dropna(); sma = dc.rolling(150).mean()
up = ((dc > sma) & (sma > sma.shift(10))).shift(1)
reg = up.reindex(d.index, method="ffill").fillna(False).values
ext = ((dc - sma) / sma * 100.0).shift(1); ea = ext.reindex(d.index, method="ffill").values
sw = swings_zigzag(h, l, atr, 2.0)


def fb(level, after):
    for j in range(after, min(after + BO, len(c))):
        if c[j] > level: return j
    return None


E = []
for t in range(2, len(sw)):
    (cL2, iL2, pL2, kL2), (cH1, iH1, pH1, kH1), (cL0, iL0, pL0, kL0) = sw[t], sw[t-1], sw[t-2]
    if not (kL2 == -1 and kH1 == +1 and kL0 == -1): continue
    if pL2 <= pL0 or pH1 - pL0 <= 0: continue
    if not np.isnan(es[cL2]) and pH1 < es[cL2]: continue
    e_i = fb(pH1, cL2 + 1)
    if e_i is None: continue
    if not reg[e_i]: continue
    if not np.isnan(ea[e_i]) and ea[e_i] > 8: continue
    e = c[e_i]; risk = e - pL2
    if risk <= 0: continue
    E.append((e_i, e, pL2, e + 4.0 * risk, pH1, pL0))
E.sort(key=lambda x: x[0]); seen = set(); U = []
for en in E:
    if en[0] in seen: continue
    seen.add(en[0]); U.append(en)
E = U


def play_limit(e_i, e, stop, tgt):
    lim = e - FRAC * (e - stop)
    if lim <= stop: return None
    fill_j = None
    for j in range(e_i + 1, min(e_i + 1 + FWD, len(c))):
        if h[j] >= tgt: return None
        if l[j] <= lim: fill_j = j; break
    if fill_j is None: return None
    u = lim - stop; rew = tgt - lim
    if l[fill_j] <= stop: return (-1.0 - SP / u, fill_j, True, fill_j)
    xj = min(fill_j + FWD, len(c) - 1); R = None; ws = False
    for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
        if l[j] <= stop: R = -1.0; xj = j; ws = True; break
        if h[j] >= tgt: R = rew / u; xj = j; break
    if R is None: R = (c[xj] - lim) / u
    return (R - SP / u, xj, ws, fill_j)


# ---- collect stopouts of the primary leg with separators + forward recovery ----
busy = -1; rows = []
for (e_i, e, pL2, tgt, pH1, pL0) in E:
    if e_i <= busy: continue
    r = play_limit(e_i, e, pL2, tgt)
    if r is None: continue
    R, xj, was_stop, fill_j = r; busy = xj
    if not was_stop: continue
    a = atr[xj] if atr[xj] > 0 else np.nan
    closed_below = c[xj] < pL2
    wick_depth = (pL2 - l[xj]) / a
    buffer_pL0 = (pL2 - pL0) / a
    speed = xj - fill_j
    # forward recovery over K bars
    end = min(xj + 1 + K, len(c))
    resumed = broke = False; y = None
    for j in range(xj + 1, end):
        if h[j] >= tgt: resumed = True
        if l[j] <= pL0: broke = True
        if y is None and c[j] > pH1: y = j
    seg_h = h[xj + 1:end]; seg_l = l[xj + 1:end]
    mfe = (seg_h.max() - c[xj]) / a if len(seg_h) else np.nan
    mae = (c[xj] - seg_l.min()) / a if len(seg_l) else np.nan
    redeploy = None
    if y is not None:
        r2 = play_limit(y, c[y], pL2, tgt)
        if r2 is not None: redeploy = r2[0]
    rows.append(dict(closed_below=closed_below, wick_depth=wick_depth, buffer_pL0=buffer_pL0,
                     speed=speed, resumed=resumed, broke=broke, mfe=mfe, mae=mae, redeploy=redeploy))

import pandas as pd
df = pd.DataFrame(rows)
print(f"gold 15m gated-uptrend primary STOPOUTS: n={len(df)}  (K={K} bars forward)")
print(f"  overall: resumed_to_tgt={df.resumed.mean()*100:.0f}%  broke_pL0={df.broke.mean()*100:.0f}%  "
      f"mfe/mae={df.mfe.median()/max(df.mae.median(),1e-9):.2f}  "
      f"redeploy meanR={df.redeploy.dropna().mean():+.3f} (n{df.redeploy.notna().sum()})")


def show(name, mask_hi, label_hi, label_lo):
    for lab, m in ((label_hi, mask_hi), (label_lo, ~mask_hi)):
        g = df[m]
        rd = g.redeploy.dropna()
        print(f"    {lab:<26} n={len(g):>4}  resumed={g.resumed.mean()*100:>3.0f}%  "
              f"broke_pL0={g.broke.mean()*100:>3.0f}%  mfe/mae={g.mfe.median()/max(g.mae.median(),1e-9):>4.2f}  "
              f"redeploy meanR={rd.mean():+.3f}(n{len(rd)})")


print("\n  [1] CLOSED below pL2 (real break) vs wicked-and-held (sweep):")
show("closed_below", df.closed_below, "closed<pL2 (break)", "wick-held (sweep)")
print("\n  [2] WICK depth below pL2 (median split):")
wd = df.wick_depth.median()
show("wick_depth", df.wick_depth >= wd, f"deep wick (>= {wd:.2f}ATR)", f"shallow (< {wd:.2f}ATR)")
print("\n  [3] structural BUFFER pL2->pL0 (median split):")
bf = df.buffer_pL0.median()
show("buffer", df.buffer_pL0 >= bf, f"far pL0 (>= {bf:.2f}ATR)", f"near pL0 (< {bf:.2f}ATR)")
print("\n  [4] SPEED to stopout (median split):")
sp_ = df.speed.median()
show("speed", df.speed >= sp_, f"slow (>= {sp_:.0f} bars)", f"fast (< {sp_:.0f} bars)")
