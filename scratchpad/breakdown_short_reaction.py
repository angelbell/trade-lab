"""#1 SHORT the failed breakout. When a gated-uptrend pullback-limit LONG stops out AND then
price CLOSES below pL0 (the deeper prior low = the higher-low structure is truly broken), that
is a failed-bullish-structure = candidate SHORT. Bounce-verification ORDER (do NOT jump to RR):
  STEP 1 = does price actually TRAVEL DOWN after the breakdown, or is it a bear-trap that snaps
           back up? Measure downside MFE / upside MAE (ATR units) + excursion distribution.
  STEP 2 = only if it travels: RR feasibility for two stops (reclaim pL0 = tight / pL2 = wide),
           % reaching 3R down, and cost/risk (the tight-stop tax that killed gold-15m bounces).
Causal: short entry = next open after the confirmed close < pL0. Descriptive; no PF/win yet."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag, kama_adaptive

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}
BO, FWD, W, K = 20, 500, 100, 50


def build(csv, gate_kind, start=None):
    d = load_mt5_csv(csv)
    if start: d = d.loc[start:]
    if csv.endswith("m5.csv"): d = d.resample("15min").agg(AGG).dropna()
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    a = ta.atr(d["high"], d["low"], d["close"], 14).values
    es = d["close"].ewm(span=80, adjust=False).mean().values
    if gate_kind == "sma":
        dc = d["close"].resample("1D").last().dropna(); sma = dc.rolling(150).mean()
        up = ((dc > sma) & (sma > sma.shift(10))).shift(1)
        reg = up.reindex(d.index, method="ffill").fillna(False).values
        ext = ((dc - sma) / sma * 100.0).shift(1); ea = ext.reindex(d.index, method="ffill").values
    else:
        dck = d["close"].resample("1D").last().dropna(); kmg = kama_adaptive(dck, 14)
        reg = ((kmg > kmg.shift(1)).shift(1)).reindex(d.index, method="ffill").fillna(False).values
        ea = None
    sw = swings_zigzag(h, l, a, 2.0)

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
        if ea is not None and not np.isnan(ea[e_i]) and ea[e_i] > 8: continue
        e = c[e_i]; risk = e - pL2
        if risk <= 0: continue
        E.append((e_i, e, pL2, pL0, e + 4.0 * risk))
    E.sort(key=lambda x: x[0]); seen = set(); U = []
    for en in E:
        if en[0] in seen: continue
        seen.add(en[0]); U.append(en)
    return U, d.index, o, h, l, c, a


def analyze(name, csv, gate_kind, cost, start=None):
    E, idx, o, h, l, c, atr = build(csv, gate_kind, start)
    frac = 0.25 if gate_kind == "sma" else 0.30
    # primary long walk -> stopouts
    busy = -1; shorts = []
    for (e_i, e, pL2, pL0, tgt) in E:
        if e_i <= busy: continue
        lim = e - frac * (e - pL2)
        if lim <= pL2: continue
        fill_j = None
        for j in range(e_i + 1, min(e_i + 1 + FWD, len(c))):
            if h[j] >= tgt: fill_j = None; break
            if l[j] <= lim: fill_j = j; break
        if fill_j is None: continue
        # resolve long
        xj = None; ws = False
        if l[fill_j] <= pL2: xj = fill_j; ws = True
        else:
            for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
                if l[j] <= pL2: xj = j; ws = True; break
                if h[j] >= tgt: xj = j; break
            if xj is None: xj = min(fill_j + FWD, len(c) - 1)
        busy = xj
        if not ws: continue
        # find confirmed breakdown: first close < pL0 within W bars after the stopout
        bd = None
        for j in range(xj + 1, min(xj + 1 + W, len(c) - 1)):
            if c[j] < pL0: bd = j; break
        if bd is None or bd + 1 >= len(c): continue
        ep = o[bd + 1]; a = atr[bd] if atr[bd] > 0 else np.nan
        end = min(bd + 2 + K, len(c))
        seg_l = l[bd + 1:end]; seg_h = h[bd + 1:end]
        if len(seg_l) < 5: continue
        mfe_dn = (ep - seg_l.min()) / a          # downside travel (ATR)
        mae_up = (seg_h.max() - ep) / a          # adverse up (ATR)
        shorts.append(dict(t=idx[bd + 1], ep=ep, pL0=pL0, pL2=pL2, a=a,
                           mfe_dn=mfe_dn, mae_up=mae_up,
                           risk_tight=pL0 - ep, risk_wide=pL2 - ep))
    df = pd.DataFrame(shorts)
    print(f"\n===== {name}: failed-breakout SHORT candidates (close<pL0 after a long stopout) =====")
    print(f"  n={len(df)}  (K={K} bars downside reaction)")
    if len(df) < 8:
        print("  too few."); return
    # STEP 1: reaction
    r = df.mfe_dn.median() / max(df.mae_up.median(), 1e-9)
    print(f"  STEP1 REACTION: median MFE_down={df.mfe_dn.median():.2f}ATR  MAE_up={df.mae_up.median():.2f}ATR  "
          f"down/up ratio={r:.2f}   (>1 travels down, <1 = bear-trap snaps up)")
    print(f"                  mean down/up={df.mfe_dn.mean()/max(df.mae_up.mean(),1e-9):.2f}  "
          f"%with MFE_down>MAE_up={(df.mfe_dn>df.mae_up).mean()*100:.0f}%")
    yr = df.t.dt.year
    print("  per-year down/up: " + " ".join(
        f"{y}:{df[yr==y].mfe_dn.median()/max(df[yr==y].mae_up.median(),1e-9):.2f}(n{(yr==y).sum()})"
        for y in sorted(set(yr))))
    # STEP 2: RR feasibility (only meaningful if it travels)
    for tag, rk in (("tight stop=reclaim pL0", df.risk_tight), ("wide stop=pL2", df.risk_wide)):
        rk = rk.where(rk > 0)
        rmax = (df.ep - (df.ep - df.mfe_dn * df.a)) / rk    # = mfe_dn*a / risk = downside R reached
        rmax = (df.mfe_dn * df.a) / rk
        pct3 = (rmax >= 3).mean() * 100
        cr = (cost / rk)
        print(f"  STEP2 {tag:<22}: median risk={rk.median():.1f}  %reach 3R_down={pct3:.0f}%  "
              f"cost/risk med={cr.median():.3f}R (tax)")


if __name__ == "__main__":
    analyze("GOLD 15m", "data/vantage_xauusd_m5.csv", "sma", 0.6)
    analyze("BTC 15m", "data/vantage_btcusd_m15.csv", "kama", 15.0, start="2018-10-01")
