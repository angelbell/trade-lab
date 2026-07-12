"""FAITHFUL replica of pine/btc_15m_swing_breakout_zigzag.pine as implemented (2026-07-12),
run to (a) fix a LOOKAHEAD BUG found while porting and (b) re-measure the flow exit honestly.

*** THE BUG (found 2026-07-12, invalidates the earlier flow numbers) ***
Vantage MT5 bar labels are BROKER time (EET/EEST = UTC+2 winter / UTC+3 summer; verified by
return-correlation vs Binance: corr 0.99-1.00 at lag +8 bars in Jan, +12 bars in Jul).
The earlier flow work indexed the Binance (UTC) series with Vantage labels directly, so the
"trailing 4h" window actually covered (t_true - 2h, t_true + 2h] -- HALF THE WINDOW WAS FUTURE
DATA. Any separation it showed is contaminated. Fix: convert the Binance UTC index to broker
wall-clock (tz_convert Europe/Riga -> drop tz) so it lines up with the Vantage labels, then the
trailing window is genuinely trailing.

REPLICA (matches the Pine line by line):
  entry   ZigZag(2xATR) L0->H1->L2 (L2>L0, H1>L0), H1 > EMA80, close > H1 within 20 bars of L2,
          daily KAMA(14) rising [prior completed day], limit at lim = e - 0.30*(e-L2),
          cancelled if price reaches the target first or after fillWin=200 bars.
  size    w = 1.0 if break close > prior-day high else 0.5 (PDH soft sizing) -> R is w-weighted.
  exits   stop = L2, target = e + 4*(e-L2) (price-fixed); optional ratchet: once high >= lim+3.5u
          stop -> lim+2.0u; optional FLOW exit: after touching lim+2u, on the FIRST later bar
          whose low <= lim+1u, if (4h up-vol / down-vol from Binance 5m bars) < 0.90 -> exit at
          the NEXT bar's open. Bar order: stop/target intrabar first, then the close-based flow
          decision, then the ratchet arm (effective next bar).
  re-arm  a new signal can enter as soon as the previous position is closed (as in the Pine).
  cost    $15 round trip.  risk 1% of equity per trade (w-scaled).
Run: .venv/bin/python scratchpad/pine_replica_btc15m.py
"""
import sys, os
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta, io, contextlib
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag, kama_adaptive

ROOT = "/home/angelbell/dev/auto-trade"
START = "2018-10-01"
BO, FILLWIN, FWD = 20, 200, 500
FRAC, RR, COST = 0.30, 4.0, 15.0
RATCH = (3.5, 2.0)
FLOW_THR, FLOW_HRS = 0.90, 4
PDH_W = 0.5
RNG = np.random.default_rng(20260712)


def flow_ratio_series(idx):
    """4h up-vol/down-vol from Binance 5m bars, expressed on the BROKER clock so that it lines
    up with the Vantage bar labels (the fix for the lookahead bug)."""
    f = pd.read_csv(os.path.join(ROOT, "data/ext_btc_5m_flow.csv"), index_col=0, parse_dates=[0])
    f.index = f.index.tz_convert("Europe/Riga").tz_localize(None).tz_localize("UTC")  # broker wall clock
    g = f[["up_vol", "dn_vol"]].resample("15min").sum()
    up = g["up_vol"].rolling(f"{FLOW_HRS}h").sum()
    dn = g["dn_vol"].rolling(f"{FLOW_HRS}h").sum()
    r = (up / dn).where(dn > 0)
    return r.reindex(idx)          # NaN before 2020-09 -> flow exit simply never fires there


def build_entries(df):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    a = ta.atr(df["high"], df["low"], df["close"], 14).values
    es = df["close"].ewm(span=80, adjust=False).mean().values
    dck = df["close"].resample("1D").last().dropna()
    kmg = kama_adaptive(dck, 14)
    kreg = ((kmg > kmg.shift(1)).shift(1)).reindex(df.index, method="ffill").fillna(False).values
    pdh = df["high"].resample("1D").max().shift(1).reindex(df.index, method="ffill").values
    sw = swings_zigzag(h, l, a, 2.0)
    E = []
    for t in range(2, len(sw)):
        (cL2, iL2, pL2, kL2), (cH1, iH1, pH1, kH1), (cL0, iL0, pL0, kL0) = sw[t], sw[t-1], sw[t-2]
        if not (kL2 == -1 and kH1 == +1 and kL0 == -1): continue
        if pL2 <= pL0 or pH1 - pL0 <= 0: continue
        if not np.isnan(es[cL2]) and pH1 < es[cL2]: continue
        e_i = None
        for j in range(cL2 + 1, min(cL2 + 1 + BO, len(c))):
            if c[j] > pH1: e_i = j; break
        if e_i is None or not kreg[e_i]: continue
        e = c[e_i]; stop = pL2
        if e - stop <= 0: continue
        w = 1.0 if (np.isfinite(pdh[e_i]) and e > pdh[e_i]) else PDH_W
        E.append((e_i, e, stop, e + RR * (e - stop), w))
    E.sort(key=lambda x: x[0])
    seen = set(); U = []
    for en in E:
        if en[0] in seen: continue
        seen.add(en[0]); U.append(en)
    return U


def walk(df, E, use_flow, use_ratchet, flow, thr=FLOW_THR):
    h, l, c, o = (df[k].values for k in ("high", "low", "close", "open"))
    fv = flow.values
    busy = -1; out = []
    for (i, e, stop0, tgt, w) in E:
        if i <= busy: continue
        lim = e - FRAC * (e - stop0)
        if lim <= stop0 or lim >= e: continue
        fill = None
        for j in range(i + 1, min(i + 1 + FILLWIN, len(c))):
            if h[j] >= tgt: break                      # ran away -> limit cancelled (missed)
            if l[j] <= lim: fill = j; break
        if fill is None: continue
        u = lim - stop0; reward = tgt - lim
        lv2, lv1 = lim + 2 * u, lim + 1 * u
        if l[fill] <= stop0:
            out.append((df.index[fill], w * (-1.0 - COST / u), w)); busy = fill; continue
        cur = stop0; ratched = False; hit2 = False; done = False; pend = False
        R = None; exit_j = min(fill + FWD, len(c) - 1)
        for j in range(fill + 1, min(fill + 1 + FWD, len(c))):
            if pend: R = (o[j] - lim) / u; exit_j = j; break        # flow exit fills here
            if l[j] <= cur: R = (cur - lim) / u; exit_j = j; break  # stop (incl. ratcheted)
            if h[j] >= tgt: R = reward / u; exit_j = j; break       # target
            if use_flow and not done and hit2 and l[j] <= lv1:      # flow decision (once)
                done = True
                v = fv[j]
                if np.isfinite(v) and v < thr and j + 1 < len(c): pend = True
            if h[j] >= lv2: hit2 = True
            if use_ratchet and not ratched and h[j] >= lim + RATCH[0] * u:
                cur = max(cur, lim + RATCH[1] * u); ratched = True
        if R is None: R = (c[exit_j] - lim) / u
        out.append((df.index[fill], w * (R - COST / u), w)); busy = exit_j
    return out


def stats(tr, span, risk=0.01):
    R = np.array([x[1] for x in tr]); t = [x[0] for x in tr]
    cum = np.cumsum(R); ddR = (np.maximum.accumulate(cum) - cum).max()
    eq = np.cumprod(1 + risk * R); pk = np.maximum.accumulate(eq)
    ddp = ((pk - eq) / pk).max() * 100
    yrs = np.array([x.year for x in t]); uy = np.unique(yrs); half = uy[len(uy) // 2]
    grn = np.mean([R[yrs == y].sum() > 0 for y in uy]) * 100
    return dict(n=len(R), npy=len(R) / span, pf=R[R > 0].sum() / abs(R[R <= 0].sum()),
                totyr=R.sum() / span, ddR=ddR, ddp=ddp, retdd=R.sum() / ddR, meanR=R.mean(),
                IS=R[yrs < half].sum(), OOS=R[yrs >= half].sum(), grn=grn,
                cagr=(eq[-1] ** (1 / span) - 1) * 100)


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    span = (df.index[-1] - df.index[0]).days / 365.25
    flow = flow_ratio_series(df.index)
    E = build_entries(df)
    print(f"replica: {len(E)} armed breaks, span {span:.1f}yr, flow coverage "
          f"{100*flow.notna().mean():.0f}% of bars\n")

    print(f"{'variant (as implemented in the Pine)':<38}{'n':>5}{'PF':>6}{'totR/yr':>9}"
          f"{'maxDD%':>8}{'tot/DD':>8}{'meanR':>8}{'grn':>6}{'IS/OOS':>13}")
    cells = {}
    for name, uf, ur in (("base (RR4 only)", False, False),
                         ("+ ratchet only", False, True),
                         ("+ flow exit only", True, False),
                         ("+ flow + ratchet (Pine default)", True, True)):
        s = stats(walk(df, E, uf, ur, flow), span)
        cells[name] = s
        print(f"{name:<38}{s['n']:>5}{s['pf']:>6.2f}{s['totyr']:>9.2f}{s['ddp']:>7.1f}%"
              f"{s['retdd']:>8.2f}{s['meanR']:>+8.3f}{s['grn']:>5.0f}%"
              f"{f'{s[chr(73)+chr(83)]:+.0f}/{s[chr(79)+chr(79)+chr(83)]:+.0f}':>13}")

    # ---- is the flow exit still better than a RANDOM exit of the same count? ----
    base_tr = walk(df, E, False, False, flow)
    flow_tr = walk(df, E, True, False, flow)
    n_int = sum(1 for a, b in zip(base_tr, flow_tr) if abs(a[1] - b[1]) > 1e-9)
    print(f"\nflow exit intervened on {n_int} of {len(base_tr)} trades")
    if n_int:
        rnd = []
        for _ in range(400):
            sh = pd.Series(RNG.random(len(df)), index=df.index)      # random 'flow' with the
            thr_r = np.nanquantile(sh.values, np.nanmean(flow.values < FLOW_THR))
            s = stats(walk(df, E, True, False, sh, thr_r), span)
            rnd.append(s["retdd"])
        f_only = cells["+ flow exit only"]["retdd"]
        print(f"random-flow control: totR/DD median={np.median(rnd):.2f}  "
              f"real flow={f_only:.2f}  %ile={(f_only > np.array(rnd)).mean()*100:.0f}")

    print("\nequal-DD comparison (scale the bet so every variant runs the base's maxDD%):")
    base_dd = cells["base (RR4 only)"]["ddp"]
    for name, s in cells.items():
        tr = walk(df, E, "flow" in name, "ratchet" in name, flow)
        R = np.array([x[1] for x in tr])
        lo, hi = 0.001, 0.30
        for _ in range(60):
            mid = (lo + hi) / 2
            eq = np.cumprod(1 + mid * R); pk = np.maximum.accumulate(eq)
            if ((pk - eq) / pk).max() * 100 > base_dd: hi = mid
            else: lo = mid
        eq = np.cumprod(1 + lo * R)
        print(f"  {name:<38} f={100*lo:>5.2f}%  CAGR={(eq[-1]**(1/span)-1)*100:>6.1f}%  "
              f"final={eq[-1]:>6.1f}x")


if __name__ == "__main__":
    main()
