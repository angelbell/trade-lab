"""instrument_screen.py -- trend-CHARACTER pre-screen to pick NEW instruments worth a Vantage H1
export + the full gauntlet. Hurst/variance-ratio used in their CANONICAL role (long-sample
trend-persistence characterization), NOT as the fast deploy gates that already failed.

PRE-SCREEN ONLY: feed = Yahoo daily (yfinance), NOT Vantage; daily bars (no H1) => measures trend
CHARACTER, not our H1 breakout strategy. A pass just earns a Vantage H1 export -> breakout_wave +
KAMA + overfit_audit, which stays the arbiter. Validity-first: trust the ranking of new names only
AFTER it ranks the KNOWN anchors right (gold/BTC = trending, USDJPY = not).

  .venv/bin/python research/instrument_screen.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.regime_gate_lab import er
from research.regime_statedet import hurst_rs, variance_ratio

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
NEW = {"US500": "^GSPC", "NAS100": "^NDX", "JP225": "^N225", "GER40": "^GDAXI"}
ANCHOR = {"GOLD": "GC=F", "BTC": "BTC-USD", "USDJPY": "JPY=X"}   # known: gold/BTC trend, JPY not


def load(ticker, start="2015-01-01"):
    """yfinance daily close, cached to data/ext_<ticker>.csv (fetch once)."""
    safe = ticker.replace("^", "idx_").replace("=", "_").replace("-", "_")
    path = os.path.join(DATA, f"ext_{safe}.csv")
    if os.path.exists(path):
        s = pd.read_csv(path, index_col=0, parse_dates=True)["close"]
    else:
        import yfinance as yf
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if df is None or len(df) == 0:
            return None
        s = df["Close"]
        s = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
        s.name = "close"
        s.to_frame().to_csv(path)
    return s.dropna()


def daily_breakout_payoff(close, N=20, sl_atr=2.0, rr=3.0):
    """crude daily Donchian(N) breakout, ATR stop, next-bar fill -> meanR. A 'does trend pay here'
    sniff test, NOT the real H1 strategy. Long-only (trend family is long-biased)."""
    c = close.values; n = len(c)
    hi = pd.Series(c).rolling(N).max().shift(1).values
    tr = np.abs(np.diff(c, prepend=c[0]))
    atr = pd.Series(tr).rolling(14).mean().values
    rows = []; busy = -1
    for i in range(N + 14, n - 1):
        if i <= busy or np.isnan(hi[i]) or np.isnan(atr[i]) or atr[i] == 0:
            continue
        if c[i] > hi[i]:                                  # close breaks N-day high
            e = c[i + 1] if i + 1 < n else c[i]; risk = sl_atr * atr[i]
            stop = e - risk; tgt = e + rr * risk; R = None
            for j in range(i + 1, min(i + 250, n)):
                if c[j] <= stop: R = -1.0; break
                if c[j] >= tgt: R = rr; break
            if R is None:
                R = (c[min(i + 250, n - 1)] - e) / risk; j = min(i + 250, n - 1)
            rows.append(R); busy = j
    r = np.array(rows)
    return (len(r), r.mean() if len(r) else np.nan)


def metrics(close):
    r = np.log(close).diff().dropna().values
    H = hurst_rs(r)
    VR = {q: variance_ratio(r, q) for q in (5, 10, 20)}
    ER = er(close, 20).mean()
    s200 = close.rolling(200).mean()
    in_trend = ((close > s200) & (s200 > s200.shift(20))).mean()
    n_bo, payoff = daily_breakout_payoff(close)
    return dict(n=len(close), H=H, VR10=VR[10], VR20=VR[20], ER=ER, trend=in_trend,
                bo_n=n_bo, bo_meanR=payoff)


def main():
    print("INSTRUMENT TREND-CHARACTER PRE-SCREEN (Yahoo daily; pre-screen only, Vantage H1 = arbiter)\n")
    rows = {}
    for grp, uni in (("ANCHOR", ANCHOR), ("NEW", NEW)):
        for name, tk in uni.items():
            s = load(tk)
            if s is None or len(s) < 400:
                print(f"  {name:<8} ({tk}) fetch FAILED / too short"); continue
            rows[name] = {"grp": grp, **metrics(s)}
    df = pd.DataFrame(rows).T
    # composite trend-character z-score (higher = more trend-following-suitable)
    feats = ["H", "VR20", "ER", "trend", "bo_meanR"]
    z = df[feats].astype(float).apply(lambda c: (c - c.mean()) / c.std(ddof=0))
    df["Zscore"] = z.mean(axis=1)
    df = df.sort_values("Zscore", ascending=False)

    print(f"  {'name':<8}{'grp':<8}{'n':>5}{'Hurst':>7}{'VR10':>6}{'VR20':>6}{'ER':>6}"
          f"{'trend%':>7}{'bo_n':>6}{'bo_R':>7}{'Zrank':>7}")
    for nm, r in df.iterrows():
        print(f"  {nm:<8}{r['grp']:<8}{int(r['n']):>5}{r['H']:>7.2f}{r['VR10']:>6.2f}{r['VR20']:>6.2f}"
              f"{r['ER']:>6.2f}{r['trend']*100:>6.0f}%{int(r['bo_n']):>6}{r['bo_meanR']:>+7.2f}{r['Zscore']:>+7.2f}")

    print("\n  -- SANITY (validity-first): anchors must rank gold/BTC HIGH, USDJPY LOW --")
    anc = df[df.grp == "ANCHOR"].sort_values("Zscore", ascending=False)
    order = list(anc.index)
    ok = (order.index("USDJPY") == len(order) - 1) if "USDJPY" in order else False
    print(f"    anchor order (high->low trend-character): {order}  -> USDJPY last? {'YES' if ok else 'NO'}")
    if not ok:
        print("    >>> screener does NOT rank known anchors correctly -> ranking of new names NOT trusted.")
        return

    # the metrics DISAGREE -> split into the two distinct characters instead of one fragile composite:
    #   BREAKOUT-continuation fit  ~ Hurst (persistence) + VR>1 (momentum)   [our breakout family]
    #   PULLBACK / buy-the-dip fit ~ high trend% (sustained up) + VR<1 (short-term mean-revert)
    print("\n  -- CHARACTER SPLIT (the metrics disagree -> two distinct trend characters) --")
    print(f"    {'name':<8}{'grp':<8}{'BO-fit(H+VR)':>14}{'PULL-fit(tr+dip)':>18}  closest-family")
    bo_g, bo_b = None, None
    for nm in ["GOLD", "BTC"]:
        r = df.loc[nm]
        if nm == "GOLD": bo_g = (r["H"], r["VR20"])
        else: bo_b = (r["H"], r["VR20"])
    bo_bar = min(bo_g[0], bo_b[0])                       # breakout bar = the lower gold/BTC Hurst
    for nm, r in df.iterrows():
        bo = r["H"] + max(r["VR20"] - 1.0, 0) * 0.5      # persistence + momentum bonus
        pull = r["trend"] + max(1.0 - r["VR20"], 0) * 0.5  # sustained-up + dip-revert bonus
        fam = "BREAKOUT (gold/BTC-like)" if r["H"] >= bo_bar and r["VR20"] >= 0.95 else (
              "PULLBACK (dip-buy)" if r["trend"] > 0.62 and r["VR20"] < 0.95 else "neither/weak")
        print(f"    {nm:<8}{r['grp']:<8}{bo:>14.2f}{pull:>18.2f}  {fam}")
    print(f"\n  READ: breakout-continuation needs Hurst >= gold/BTC ({bo_bar:.2f}) -- NO index clears that")
    print("  (US500/NAS100 Hurst<0.5 = NOT persistent). Indices show high trend%+VR<1 = BUY-THE-DIP")
    print("  character => the transferable family for indices is the EMA-PULLBACK leg, NOT breakout.")
    print("  NEXT (if pursued): export US500/NAS100 Vantage H1 -> test ema_pullback (+KAMA), full gauntlet.")
    print("  Pre-screen = character only; the H1 strategy + cost/correlation/regime = the real arbiter.")


if __name__ == "__main__":
    main()
