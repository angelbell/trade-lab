"""eurusd_yf_prescreen.py -- PRE-SCREEN only: does EURUSD have session structure + any Asian-range
breakout/fade edge, on the yfinance feed? Vantage is the arbiter (pull via the mt5-mcp bridge); this is the
instrument_screen.py-style cheap pre-check while the bridge is down.

USDJPY's session breakout AND fade were both DEAD (-0.24 / -0.32, no directional edge gross + cost wall).
EURUSD is London-centric and less BoJ-managed -- maybe its session carries directional structure USDJPY's
doesn't. Test the SAME falsifier before spending the Vantage pull:
  PASS(pre) = clear VOL hump at London/NY AND breakout OR fade shows cost-after meanR>0 AND beats a QUIET
  control window AND window choice plateaus. yfinance feed has NO tick volume (range=vol proxy) and is NOT
  the trade feed -> a pre-screen PASS only earns a Vantage re-test; a clean FAIL kills it cheaply.

Caveats baked in: tz = Europe/London (so hr = London local), weekends dropped, cost in price units (0.00003
~0.3pip EURUSD spread). In-sample; Vantage arbitrates.
  .venv/bin/python research/eurusd_yf_prescreen.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf


def load_eurusd():
    df = yf.download("EURUSD=X", period="730d", interval="1h", progress=False, auto_adjust=False)
    df.columns = [c[0].lower() for c in df.columns]            # flatten multiindex
    d = df[["open", "high", "low", "close"]].copy()
    d = d[(d.index.dayofweek < 5)]                              # drop weekends
    d = d[d["high"] >= d["low"]].dropna()
    return d


def atr(d, n=14):
    h, l, c = d["high"], d["low"], d["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def screen(d):
    c = d["close"]; ret = np.log(c / c.shift(1)); rng = (d["high"] - d["low"]) / c
    hh = d.index.hour
    print(f"== EURUSD hour-of-day (London tz; {d.index.min().date()}..{d.index.max().date()}, n={len(d)}) ==")
    print(f"  {'hr':>3} {'n':>5} {'meanRet(bps)':>13} {'|ret|(bps)':>11} {'range(bps)':>11} {'up%':>6}")
    g = pd.DataFrame({"h": hh, "ret": ret.values, "rng": rng.values}).dropna()
    for h in range(24):
        s = g[g.h == h]
        if len(s) < 50:
            continue
        print(f"  {h:>3} {len(s):>5} {s.ret.mean()*1e4:>+13.2f} {s.ret.abs().mean()*1e4:>11.1f} "
              f"{s.rng.mean()*1e4:>11.1f} {(s.ret>0).mean()*100:>5.1f}%")
    print(f"\n  day-of-week:  {'dow':>4} {'n':>5} {'meanRet(bps)':>13} {'up%':>6}")
    dn = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    daily = np.log(c.resample("1D").last()).diff()
    dd = pd.DataFrame({"d": daily.index.dayofweek, "r": daily.values}).dropna()
    for k in range(5):
        s = dd[dd.d == k]
        if len(s) < 10:
            continue
        print(f"               {dn[k]:>4} {len(s):>5} {s.r.mean()*1e4:>+13.2f} {(s.r>0).mean()*100:>5.1f}%")


def run(d, asia=(0, 7), bo=(8, 16), eod=20, k=1.0, rr=2.0, cost=0.00003, fade=False):
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    a = atr(d).values
    hr = d.index.hour.values
    day = d.index.normalize()
    in_asia = (hr >= asia[0]) & (hr < asia[1])
    dfA = pd.DataFrame({"day": day, "h": h, "l": l, "in": in_asia})
    aH = dfA[dfA["in"]].groupby("day")["h"].max()
    aL = dfA[dfA["in"]].groupby("day")["l"].min()
    in_bo = (hr >= bo[0]) & (hr < bo[1])
    traded = set(); rows = []; i = 0; n = len(d)
    while i < n - 1:
        dd = day[i]
        if in_bo[i] and dd not in traded and dd in aH.index and np.isfinite(a[i]) and a[i] > 0:
            rh, rl = aH[dd], aL[dd]
            brokeUp, brokeDn = c[i] > rh, c[i] < rl
            isL = brokeDn if fade else brokeUp
            isS = brokeUp if fade else brokeDn
            if isL or isS:
                e = c[i]
                stop = e - k * a[i] if isL else e + k * a[i]
                risk = abs(e - stop)
                tgt = e + rr * risk if isL else e - rr * risk
                R = None; j = i + 1
                while j < n and (day[j] == dd) and hr[j] < eod:
                    if (l[j] <= stop) if isL else (h[j] >= stop):
                        R = -1; break
                    if (h[j] >= tgt) if isL else (l[j] <= tgt):
                        R = rr; break
                    j += 1
                if R is None:
                    jj = min(j, n - 1)
                    R = ((c[jj] - e) if isL else (e - c[jj])) / risk
                R -= cost / risk
                rows.append((d.index[i], "L" if isL else "S", R))
                traded.add(dd)
        i += 1
    return pd.DataFrame(rows, columns=["time", "side", "R"])


def line(tag, t, rr=2.0):
    if len(t) < 20:
        print(f"  {tag:<22} n={len(t)} (too few)"); return
    be = 100 / (1 + rr)
    w, ll = t[t.R > 0].R.sum(), -t[t.R < 0].R.sum()
    print(f"  {tag:<22} n={len(t):>4} win%={(t.R>0).mean()*100:>3.0f}(BE{be:.0f}) meanR={t.R.mean():+5.2f} "
          f"totR={t.R.sum():>+6.1f} PF={w/max(ll,1e-9):4.2f}")


def main():
    d = load_eurusd()
    screen(d)
    rr = 2.0
    print(f"\n== Asian-range -> session BREAKOUT (asia 0-7, bo 8-16 London, confirmed close, RR{rr}, cost~0.3pip) ==")
    line("breakout 8-16", run(d, bo=(8, 16), rr=rr), rr)
    line("London 8-12", run(d, bo=(8, 12), rr=rr), rr)
    line("NY 13-17", run(d, bo=(13, 17), rr=rr), rr)
    print(" FADE (false-break reversal):")
    line("fade 8-16", run(d, bo=(8, 16), rr=rr, fade=True), rr)
    line("fade London 8-12", run(d, bo=(8, 12), rr=rr, fade=True), rr)
    print(" CONTROL (quiet window -- if breakout/fade as good here, session doesn't matter):")
    line("quiet 20-24 bo", run(d, asia=(13, 20), bo=(20, 24), eod=7, rr=rr), rr)
    line("quiet 20-24 fade", run(d, asia=(13, 20), bo=(20, 24), eod=7, rr=rr, fade=True), rr)

    print("\n== window-start plateau (breakout) ==")
    for s in (7, 8, 9, 10):
        line(f"  bo {s}-16", run(d, bo=(s, 16), rr=rr), rr)
    print("== RR sweep (breakout 8-16) ==")
    for r in (1.5, 2.0, 2.5, 3.0):
        line(f"  RR{r}", run(d, bo=(8, 16), rr=r), r)
    print("== cost stress (breakout 8-16) ==")
    for cc in (0.00003, 0.00006, 0.0001):
        line(f"  cost~{cc/0.0001:.0f}pip", run(d, bo=(8, 16), rr=rr, cost=cc), rr)
    print("\n  PRE-SCREEN verdict: a clean negative on the Yahoo feed kills it before the Vantage pull;")
    print("  any window with cost-after meanR>0 that beats the quiet control + plateaus earns a Vantage re-test.")


if __name__ == "__main__":
    main()
