"""USDJPY SAR fade-to-mean (user pivot 2026-07-05).

Entry: price STRETCHED from EMA200 AND Parabolic-SAR FLIPS -> fade toward the mean.
  SHORT fade: close-EMA200 > k*ATR (stretched UP)   AND PSAR flips to short (dot->above)
  LONG  fade: EMA200-close > k*ATR (stretched DOWN)  AND PSAR flips to long  (dot->below)
Exit: fade-to-mean (harness exit_mode="mean" = 20-SMA target, katr*ATR adverse stop).
Falsify order: all-signals BASE first (sweep k), GROSS vs NET, THEN radar-weak HTF gate.
Causal: SAR/EMA on closed bar i; HTF radar via merge_asof backward (no lookahead).
"""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from pandas.tseries.frequencies import to_offset
import research.edge_harness as EH
from research.edge_harness import evaluate, check_causal, sweep, random_drop_null, LADDERS, AGG
from src.data_loader import load_mt5_csv

_m5 = load_mt5_csv("data/vantage_usdjpy_m5.csv"); _m5 = _m5[_m5.index >= "2018-01-01"]
_real = load_mt5_csv
EH.load_mt5_csv = lambda p, *a, **k: _m5.copy() if p == "MEM_M5c" else _real(p, *a, **k)
LADDERS["UJ_M5c"] = ("MEM_M5c", 0.015, [("5m", "5min")])
LADDERS["UJ_M15"] = ("data/vantage_usdjpy_m15.csv", 0.015, [("15m", "15min")])

def _psar_flip(df):
    ps = ta.psar(df["high"], df["low"], df["close"], af0=0.02, af=0.02, max_af=0.2)
    ld = ps.filter(like="PSARl").iloc[:, 0].notna().values   # dot below = up
    sd = ps.filter(like="PSARs").iloc[:, 0].notna().values   # dot above = down
    fl = np.zeros(len(df), bool); fs = np.zeros(len(df), bool)
    fl[1:] = ld[1:] & ~ld[:-1]     # flip to long
    fs[1:] = sd[1:] & ~sd[:-1]     # flip to short
    return fl, fs

def _range_ref(df, ref):
    """deviation-width reference = a PERIOD price-range (user spec: 当日/一定期間の変動幅)."""
    if ref == "atr":                                   # baseline: 14-bar entry-TF ATR
        return ta.atr(df["high"], df["low"], df["close"], 14).values
    if ref == "win":                                   # rolling ~1-day high/low range on entry TF
        P = max(1, int(round(1440 / ((df.index[1] - df.index[0]).seconds / 60))))
        return (df["high"].rolling(P).max() - df["low"].rolling(P).min()).values
    if ref == "day":                                   # DAILY ATR14 (当日ボラ), causal ffill to entry bars
        d = df.resample("1440min").agg(AGG).dropna()
        da = ta.atr(d["high"], d["low"], d["close"], 14)
        avail = d.index + to_offset("1440min")         # yesterday's daily range, known at day close
        m = pd.merge_asof(pd.DataFrame({"t": df.index.values}),
                          pd.DataFrame({"t": avail.values, "r": da.values}).sort_values("t"),
                          on="t", direction="backward")
        return m["r"].values
    raise ValueError(ref)

def make_fade(k, ref="day"):
    def fade(df):
        c = df["close"].values
        e200 = df["close"].ewm(span=200, adjust=False).mean().values
        rr = _range_ref(df, ref)
        fl, fs = _psar_flip(df)
        dev = c - e200
        sig = np.zeros(len(c))
        sig[(dev < -k * rr) & fl] = 1    # stretched down + SAR flips up -> fade LONG to mean
        sig[(dev > k * rr) & fs] = -1    # stretched up   + SAR flips dn -> fade SHORT to mean
        sig[:210] = 0
        return sig
    return fade

def meter_dir_strength(df):
    c = df["close"]; a = ta.atr(df["high"], df["low"], df["close"], 14)
    er = ((c - c.shift(20)).abs() / (c - c.shift(1)).abs().rolling(20).sum()).clip(0, 1)
    adxN = ((ta.adx(df["high"], df["low"], df["close"], 14)["ADX_14"] - 15) / 25).clip(0, 1)
    eF = c.ewm(span=20, adjust=False).mean(); eS = c.ewm(span=50, adjust=False).mean()
    st = ((c > eF).astype(int) * 2 - 1) + ((eF > eS).astype(int) * 2 - 1) + ((eS > eS.shift(10)).astype(int) * 2 - 1)
    align = st.abs() / 3.0
    slopeN = ((eS - eS.shift(10)).abs() / (a * 1.5)).clip(0, 1)
    atrexpN = ((a / a.rolling(100).mean() - 0.8) / 0.7).clip(0, 1)
    return (10.0 * (er + adxN + align + slopeN + atrexpN) / 5.0).values, st.values

def make_fade_gated(k, gate_frs, thr=4.0, ref="day"):
    base = make_fade(k, ref)
    def sig(df):
        s0 = base(df); weak = np.ones(len(df), bool)
        for fr in gate_frs:
            h = df.resample(fr).agg(AGG).dropna()
            s, _ = meter_dir_strength(h)
            avail = h.index + to_offset(fr)
            m = pd.merge_asof(pd.DataFrame({"t": df.index.values}),
                              pd.DataFrame({"t": avail.values, "s": s}).sort_values("t"),
                              on="t", direction="backward")
            weak &= (m["s"].values < thr)     # fade only when HTF is NOT strongly trending
        out = s0.copy(); out[~weak] = 0
        return out
    return sig

if __name__ == "__main__":
    print("### dev-width = DAILY range (当日ATR14). BASE fade-to-mean, k-sweep @ 15m -- GROSS ###")
    sweep("UJ_M15", lambda k: make_fade(k, "day"), [0.5, 0.75, 1.0, 1.5, 2.0], tf="15m", pname="k", exit_mode="mean", cost=0.0)
    print("### -- NET ###")
    sweep("UJ_M15", lambda k: make_fade(k, "day"), [0.5, 0.75, 1.0, 1.5, 2.0], tf="15m", pname="k", exit_mode="mean")
    print("### dev-width = rolling 1-DAY high/low range. GROSS ###")
    sweep("UJ_M15", lambda k: make_fade(k, "win"), [0.25, 0.5, 0.75, 1.0], tf="15m", pname="k", exit_mode="mean", cost=0.0)
    K = 1.0
    print(f"\n### full card @ k={K} (daily-range ref), 15m + 5m ###")
    evaluate("UJ_M15", make_fade(K, "day"), exit_mode="mean", stop_slip=0.5)
    evaluate("UJ_M5c", make_fade(K, "day"), exit_mode="mean", stop_slip=0.5)
    print("\n### + radar-WEAK HTF gate (2h,4h radar<4 = avoid fading a strong trend), 15m ###")
    rG = evaluate("UJ_M15", make_fade_gated(K, ["120min", "240min"], ref="day"), exit_mode="mean", stop_slip=0.5, _return=True)
    print("\n### causal + random-drop null (gate vs random trim) ###")
    check_causal(make_fade_gated(K, ["120min", "240min"], ref="day"), EH.load_mt5_csv(LADDERS["UJ_M15"][0]).resample("15min").agg(AGG).dropna())
    bG = evaluate("UJ_M15", make_fade(K, "day"), exit_mode="mean", stop_slip=0.5, quiet=True, _return=True, beta_trials=0)
    if "15m" in rG and "15m" in bG:
        random_drop_null(bG["15m"][0], rG["15m"][0], bG["15m"][1]["ny"])
