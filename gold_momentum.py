"""gold_momentum.py -- mechanise Gemini's "Gold Momentum Follower" (XAUUSD M5).

v1 (all-pullback): HTF(1h+15m)>200EMA, perfect order rising, pull to fast EMA,
   MACD trigger, bullish close -> next open. REFUTED (PF 0.66, 7yr all red).

v2 (Gemini's revision, --first-only): only the FIRST pullback after a fresh
   perfect-order establishment; MACD dropped; ATR-variable TP/SL; 15m-only HTF;
   optional time-of-day + ATR-squeeze gates.
     establish : perfect order (20>75>200) forms this bar (wasn't last bar).
     1st pull  : while armed, the first bar that drops >=`pull-atr`*ATR from the
                 post-establish running high AND touches the fast EMA. Later
                 pullbacks are IGNORED until PO re-establishes (counter reset).
     confirm   : first bullish bar within `confirm-win` bars of the touch -> next open.
     exit (atr): SL=`sl-atr`*ATR_entry, TP=`tp-atr`*ATR_entry, move stop to BE
                 once +`be-atr`*ATR reached (Gemini: RR1:1.5, BE@+1ATR).

Gold pip = 0.1 USD. Cost charged once/trade (round trip), default 1.4 pips.
No lookahead: HTF uses completed bars; entries fill next-bar open; ATR_entry uses
the entry bar's ATR; intrabar SL/TP/BE on subsequent bars.

Run (Gemini v2 full):
  .venv/bin/python gold_momentum.py --csv data/vantage_xauusd_m5.csv --first-only \
     --exit atr --sl-atr 1.0 --tp-atr 1.5 --be-atr 1.0 --htf-tf 15min --byyear
"""

import argparse
import numpy as np
import pandas as pd
import pandas_ta as ta

from src.data_loader import load_mt5_csv

PIP = 0.1  # XAUUSD: 1 pip = 0.1 USD


def htf_above_ema(close: pd.Series, rule: str, period: int) -> pd.Series:
    r = close.resample(rule, label="right", closed="right").last().dropna()
    ema = r.ewm(span=period, adjust=False).mean()
    ok = (r > ema).shift(1)
    return ok.reindex(close.index, method="ffill").fillna(False).astype(bool)


def run(d: pd.DataFrame, args) -> None:
    c, o, hi, lo = d["close"], d["open"], d["high"], d["low"]
    cv, ov, hv, lv = c.values, o.values, hi.values, lo.values
    n = len(cv)

    e1 = c.ewm(span=args.ema_fast, adjust=False).mean().values
    e2 = c.ewm(span=args.ema_mid, adjust=False).mean().values
    e3 = c.ewm(span=args.ema_slow, adjust=False).mean().values
    atr = ta.atr(hi, lo, c, length=args.atr_len).values
    atr_ma = pd.Series(atr).rolling(288, min_periods=50).mean().values   # 24h avg (M5)
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    line, hist, sig = macd.iloc[:, 0].values, macd.iloc[:, 1].values, macd.iloc[:, 2].values

    # HTF regime gate(s)
    htfs = [t.strip() for t in args.htf_tf.split(",") if t.strip()]
    up = np.ones(n, bool); dn = np.ones(n, bool)
    for rule in htfs:
        u = htf_above_ema(c, rule, args.htf_period).values
        up &= u; dn &= ~u

    # time-of-day gate (JST windows -> UTC): London 15:30-18:30 JST = 06:30-09:30 UTC;
    # NY 21:00-23:30 JST = 12:00-14:30 UTC. index is tz-aware UTC.
    minute_utc = (d.index.hour * 60 + d.index.minute).values
    time_ok = np.ones(n, bool)
    if args.time_filter:
        lon = (minute_utc >= 6 * 60 + 30) & (minute_utc <= 9 * 60 + 30)
        ny = (minute_utc >= 12 * 60) & (minute_utc <= 14 * 60 + 30)
        time_ok = lon | ny
    # ATR-squeeze gate: skip dead tape (ATR < 50% of its 24h average)
    sq_ok = np.ones(n, bool)
    if args.atr_squeeze:
        sq_ok = atr >= 0.5 * atr_ma

    long_sig = np.zeros(n, bool)
    short_sig = np.zeros(n, bool)
    sb = args.slope_lb
    warm = max(args.ema_slow, sb, 30)

    if not args.first_only:
        # ----- v1: every pullback (+ MACD) -----
        for i in range(warm, n):
            if np.isnan(e3[i]) or np.isnan(atr[i]) or np.isnan(hist[i]) or np.isnan(hist[i - 2]):
                continue
            po_up = e1[i] > e2[i] > e3[i]
            rising = e1[i] > e1[i - sb] and e2[i] > e2[i - sb] and e3[i] > e3[i - sb]
            pull_up = lv[i] <= e1[i] + args.pull_atr * atr[i]
            macd_up = (not args.no_macd) and ((hist[i] > hist[i - 1] and hist[i - 1] <= hist[i - 2] and line[i] > 0)
                                              or (line[i] > sig[i] and line[i - 1] <= sig[i - 1]))
            macd_up = macd_up or args.no_macd
            bull = cv[i] > ov[i]
            if args.dir in ("both", "long") and up[i] and po_up and rising and pull_up and macd_up and bull \
                    and time_ok[i] and sq_ok[i]:
                long_sig[i] = True
            po_dn = e1[i] < e2[i] < e3[i]
            falling = e1[i] < e1[i - sb] and e2[i] < e2[i - sb] and e3[i] < e3[i - sb]
            pull_dn = hv[i] >= e1[i] - args.pull_atr * atr[i]
            macd_dn = (not args.no_macd) and ((hist[i] < hist[i - 1] and hist[i - 1] >= hist[i - 2] and line[i] < 0)
                                              or (line[i] < sig[i] and line[i - 1] >= sig[i - 1]))
            macd_dn = macd_dn or args.no_macd
            bear = cv[i] < ov[i]
            if args.dir in ("both", "short") and dn[i] and po_dn and falling and pull_dn and macd_dn and bear \
                    and time_ok[i] and sq_ok[i]:
                short_sig[i] = True
    else:
        # ----- v2: FIRST pullback only (state machine) -----
        cw = args.confirm_win
        # long state
        arm_l = pulled_l = False; ran_hi = 0.0; pull_bar_l = -1
        # short state
        arm_s = pulled_s = False; ran_lo = 0.0; pull_bar_s = -1
        po_up_p = po_dn_p = False
        for i in range(warm, n):
            if np.isnan(e3[i]) or np.isnan(atr[i]):
                po_up_p = po_dn_p = False
                continue
            po_up = e1[i] > e2[i] > e3[i]
            po_dn = e1[i] < e2[i] < e3[i]
            # ----- LONG -----
            if args.dir in ("both", "long"):
                if po_up and not po_up_p:                 # fresh establishment
                    arm_l, pulled_l, ran_hi = True, False, hv[i]
                if not po_up:
                    arm_l = pulled_l = False
                if arm_l:
                    ran_hi = max(ran_hi, hv[i])
                    if not pulled_l:
                        drop = ran_hi - lv[i]
                        touch = lv[i] <= e1[i] or lv[i] <= e2[i]
                        if drop >= args.pull_atr * atr[i] and touch:
                            pulled_l, pull_bar_l = True, i
                    if pulled_l:
                        if i - pull_bar_l > cw:
                            arm_l = pulled_l = False       # missed window -> wait re-establish
                        elif cv[i] > ov[i] and up[i] and time_ok[i] and sq_ok[i]:
                            long_sig[i] = True
                            arm_l = pulled_l = False       # consume the 1st pullback
            # ----- SHORT -----
            if args.dir in ("both", "short"):
                if po_dn and not po_dn_p:
                    arm_s, pulled_s, ran_lo = True, False, lv[i]
                if not po_dn:
                    arm_s = pulled_s = False
                if arm_s:
                    ran_lo = min(ran_lo, lv[i])
                    if not pulled_s:
                        rise = hv[i] - ran_lo
                        touch = hv[i] >= e1[i] or hv[i] >= e2[i]
                        if rise >= args.pull_atr * atr[i] and touch:
                            pulled_s, pull_bar_s = True, i
                    if pulled_s:
                        if i - pull_bar_s > cw:
                            arm_s = pulled_s = False
                        elif cv[i] < ov[i] and dn[i] and time_ok[i] and sq_ok[i]:
                            short_sig[i] = True
                            arm_s = pulled_s = False
            po_up_p, po_dn_p = po_up, po_dn

    # ---------------- trade loop ----------------
    tp_d = args.tp * PIP; sl_d = args.sl * PIP; be_d = args.be * PIP; cost = args.cost * PIP
    idx = d.index
    trades = []
    pos = 0; ei = 0; e_px = stop = tp = be_lvl = 0.0; be_done = False; trail_anchor = 0.0

    def exit_trade(j, px):
        nonlocal pos
        gross = (px - e_px) if pos > 0 else (e_px - px)
        seg_h = hv[ei:j + 1].max(); seg_l = lv[ei:j + 1].min()
        if pos > 0:
            mfe, mae = (seg_h - e_px) / PIP, (e_px - seg_l) / PIP
        else:
            mfe, mae = (e_px - seg_l) / PIP, (seg_h - e_px) / PIP
        trades.append((idx[ei], pos, gross / PIP - cost / PIP, j - ei, mfe, mae))
        pos = 0

    for i in range(warm, n - 1):
        if pos != 0:
            if pos > 0:
                if lv[i] <= stop:
                    exit_trade(i, stop)
                elif hv[i] >= tp:
                    exit_trade(i, tp)
                else:
                    if args.exit == "atr" and not be_done and hv[i] >= be_lvl:
                        stop = max(stop, e_px); be_done = True
                    if args.exit in ("trail", "betrail"):
                        trail_anchor = max(trail_anchor, hv[i])
                        ns = trail_anchor - args.trail_atr * atr[i]
                        if args.exit == "betrail" and trail_anchor - e_px >= be_d:
                            ns = max(ns, e_px)
                        stop = max(stop, ns)
            else:
                if hv[i] >= stop:
                    exit_trade(i, stop)
                elif lv[i] <= tp:
                    exit_trade(i, tp)
                else:
                    if args.exit == "atr" and not be_done and lv[i] <= be_lvl:
                        stop = min(stop, e_px); be_done = True
                    if args.exit in ("trail", "betrail"):
                        trail_anchor = min(trail_anchor, lv[i])
                        ns = trail_anchor + args.trail_atr * atr[i]
                        if args.exit == "betrail" and e_px - trail_anchor >= be_d:
                            ns = min(ns, e_px)
                        stop = min(stop, ns)
        if pos != 0:
            continue
        want = 1 if long_sig[i] else (-1 if short_sig[i] else 0)
        if want == 0:
            continue
        e_px = ov[i + 1]; ei = i + 1; pos = want; trail_anchor = e_px; be_done = False
        ae = atr[i + 1] if not np.isnan(atr[i + 1]) else atr[i]
        if args.exit == "atr":
            sld = args.sl_atr * ae; tpd = args.tp_atr * ae
            be_lvl = e_px + (args.be_atr * ae if want > 0 else -args.be_atr * ae)
        else:
            sld = sl_d; tpd = (tp_d if args.exit == "fixed" else args.rr * sl_d)
        if want > 0:
            stop = e_px - sld; tp = e_px + tpd
        else:
            stop = e_px + sld; tp = e_px - tpd

    if not trades:
        print("  no trades"); return
    t = pd.DataFrame(trades, columns=["t_in", "dir", "pips", "bars", "mfe", "mae"])
    t["y"] = t["t_in"].dt.tz_localize(None).dt.year
    wins = t[t["pips"] > 0]["pips"]; loss = t[t["pips"] < 0]["pips"]
    pf = wins.sum() / abs(loss.sum()) if len(loss) and loss.sum() else float("inf")
    # constant-risk DD%: risk_pct on each trade's own SL distance (in pips)
    sl_pips = t["pips"].where(t["pips"] >= 0)  # placeholder; use realized risk model below
    risk_unit = args.sl_atr if args.exit == "atr" else (args.sl if args.exit == "fixed" else args.sl)
    # For DD%, normalise each trade by a representative 1R: use median loss magnitude.
    one_r = abs(loss.mean()) if len(loss) else args.sl
    rets = (t["pips"].values / one_r) * (args.risk_pct / 100.0)
    eqc = np.cumprod(1 + rets); peak = np.maximum.accumulate(eqc)
    ddp = ((peak - eqc) / peak).max() * 100.0
    tot = (eqc[-1] - 1) * 100.0
    tag = "first" if args.first_only else "all"
    print(f"  setup={tag:<5} exit={args.exit:<7} dir={args.dir:<5} trades={len(t):>5}  "
          f"net={t['pips'].sum():+8.0f}p  win={(t['pips']>0).mean()*100:>3.0f}%  PF={pf:4.2f}  "
          f"avgW={wins.mean() if len(wins) else 0:+5.1f} avgL={loss.mean() if len(loss) else 0:+6.1f}  "
          f"avgBars={t['bars'].mean():4.0f}  maxDD={ddp:4.1f}%  ret={tot:+6.0f}%")
    if args.byyear:
        print("      by year: " + "  ".join(
            f"{int(y)}:{g['pips'].sum():+.0f}({len(g)})" for y, g in t.groupby("y")))


def main():
    p = argparse.ArgumentParser(description="Gold Momentum Follower (XAUUSD M5) backtest")
    p.add_argument("--csv", required=True)
    p.add_argument("--first-only", action="store_true", help="v2: only the 1st pullback after PO establishment")
    p.add_argument("--no-macd", action="store_true", help="drop the MACD trigger (v2 default behaviour)")
    p.add_argument("--exit", default="fixed", choices=["fixed", "rr", "trail", "betrail", "atr"])
    p.add_argument("--dir", default="both", choices=["both", "long", "short"])
    p.add_argument("--tp", type=float, default=12.0)
    p.add_argument("--sl", type=float, default=10.0)
    p.add_argument("--rr", type=float, default=2.0)
    p.add_argument("--trail-atr", type=float, default=2.0)
    p.add_argument("--be", type=float, default=10.0)
    p.add_argument("--sl-atr", type=float, default=1.0, help="ATR exit: SL = this*ATR_entry")
    p.add_argument("--tp-atr", type=float, default=1.5, help="ATR exit: TP = this*ATR_entry")
    p.add_argument("--be-atr", type=float, default=1.0, help="ATR exit: move to BE at +this*ATR")
    p.add_argument("--ema-fast", type=int, default=20)
    p.add_argument("--ema-mid", type=int, default=75)
    p.add_argument("--ema-slow", type=int, default=200)
    p.add_argument("--htf-tf", default="1h,15min", help="HTF gate TFs (comma-sep), e.g. '15min' or '1h,15min'")
    p.add_argument("--htf-period", type=int, default=200)
    p.add_argument("--slope-lb", type=int, default=3)
    p.add_argument("--pull-atr", type=float, default=0.5, help="pullback depth/zone in ATR (v2 uses 0.7)")
    p.add_argument("--confirm-win", type=int, default=5, help="bars allowed between 1st-pullback touch and bullish confirm")
    p.add_argument("--time-filter", action="store_true", help="only London/NY-open JST windows")
    p.add_argument("--atr-squeeze", action="store_true", help="skip when ATR < 50%% of 24h avg")
    p.add_argument("--atr-len", type=int, default=14)
    p.add_argument("--cost", type=float, default=1.4)
    p.add_argument("--risk-pct", type=float, default=1.0)
    p.add_argument("--byyear", action="store_true")
    p.add_argument("--start", default="2018-06-01")
    p.add_argument("--end", default=None)
    args = p.parse_args()

    d = load_mt5_csv(args.csv).loc[args.start:args.end]
    print(f"\n=== Gold Momentum  {args.csv}  EMA({args.ema_fast}/{args.ema_mid}/{args.ema_slow}) "
          f"HTF{args.htf_period}({args.htf_tf})  pull{args.pull_atr}ATR  cost{args.cost}p"
          f"{'  [first-pullback]' if args.first_only else ''}"
          f"{'  [time]' if args.time_filter else ''}{'  [squeeze]' if args.atr_squeeze else ''} ===")
    print(f"  {len(d):,} M5 bars  {d.index[0]} -> {d.index[-1]}")
    run(d, args)


if __name__ == "__main__":
    main()
