"""SAR x 200EMA + RADAR multi-TF alignment gate (user spec 2026-07-05).

Gate = RADAR trend-strength meter (trend_strength_meter.pine, 0-10) on the HTFs
ABOVE the entry TF, all same direction (MA-stack sign) AND each strength >=4.
  5m  entry -> gate 15m + 2h + 4h
  15m entry -> gate 2h + 4h        (drop self-TF; entry-TF own strength not gated)
Entry = faithful SAR x EMA10/25-cross x EMA200 (experiments/sar_ema200.py).
Causal: HTF meter uses only CLOSED HTF bars (merge_asof backward on close-time).
"""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from pandas.tseries.frequencies import to_offset
import research.edge_harness as EH
from research.edge_harness import evaluate, check_causal, random_drop_null, LADDERS, AGG
from src.data_loader import load_mt5_csv

# trimmed clean M5 (the raw file has sparse pre-2018 history that distorts N/yr)
_m5 = load_mt5_csv("data/vantage_usdjpy_m5.csv")
_m5 = _m5[_m5.index >= "2018-01-01"]
print(f"M5 trimmed: {len(_m5)} bars {_m5.index[0].date()}->{_m5.index[-1].date()}")
_real_load = load_mt5_csv
def _patched(path, *a, **k):
    return _m5.copy() if path == "MEM_M5c" else _real_load(path, *a, **k)
EH.load_mt5_csv = _patched
LADDERS["UJ_M5c"] = ("MEM_M5c", 0.015, [("5m", "5min")])
LADDERS["UJ_M15"] = ("data/vantage_usdjpy_m15.csv", 0.015, [("15m", "15min")])

def sar_ema(df):
    c = df["close"].values
    e10 = df["close"].ewm(span=10, adjust=False).mean().values
    e25 = df["close"].ewm(span=25, adjust=False).mean().values
    e200 = df["close"].ewm(span=200, adjust=False).mean().values
    ps = ta.psar(df["high"], df["low"], df["close"], af0=0.02, af=0.02, max_af=0.2)
    long_dot = ps.filter(like="PSARl").iloc[:, 0].notna().values
    short_dot = ps.filter(like="PSARs").iloc[:, 0].notna().values
    cu = np.zeros(len(c), bool); cd = np.zeros(len(c), bool)
    cu[1:] = (e10[1:] > e25[1:]) & (e10[:-1] <= e25[:-1])
    cd[1:] = (e10[1:] < e25[1:]) & (e10[:-1] >= e25[:-1])
    sig = np.zeros(len(c))
    sig[cu & (c > e200) & long_dot] = 1
    sig[cd & (c < e200) & short_dot] = -1
    sig[:210] = 0
    return sig

def meter(df):
    """RADAR strength(0-10) + MA-stack dir(-3..3), faithful to trend_strength_meter.pine."""
    c = df["close"]
    a = ta.atr(df["high"], df["low"], df["close"], 14)
    er = ((c - c.shift(20)).abs() / (c - c.shift(1)).abs().rolling(20).sum()).clip(0, 1)
    adx = ta.adx(df["high"], df["low"], df["close"], 14)["ADX_14"]
    adxN = ((adx - 15.0) / 25.0).clip(0, 1)
    eF = c.ewm(span=20, adjust=False).mean(); eS = c.ewm(span=50, adjust=False).mean()
    st = ((c > eF).astype(int) * 2 - 1) + ((eF > eS).astype(int) * 2 - 1) + ((eS > eS.shift(10)).astype(int) * 2 - 1)
    align = st.abs() / 3.0
    slopeN = ((eS - eS.shift(10)).abs() / (a * 1.5)).clip(0, 1)
    atrexpN = ((a / a.rolling(100).mean() - 0.8) / 0.7).clip(0, 1)
    strength = 10.0 * (er + adxN + align + slopeN + atrexpN) / 5.0
    return strength.values, st.values

def make_gated(gate_frs, thr=4.0):
    def sig(df):
        base = sar_ema(df)
        okL = np.ones(len(df), bool); okS = np.ones(len(df), bool)
        for fr in gate_frs:
            h = df.resample(fr).agg(AGG).dropna()
            s, st = meter(h)
            avail = h.index + to_offset(fr)                      # value known only at HTF close
            left = pd.DataFrame({"t": df.index.values})
            right = pd.DataFrame({"t": avail.values, "s": s, "d": st}).sort_values("t")
            m = pd.merge_asof(left, right, on="t", direction="backward")
            sv = m["s"].values; dv = m["d"].values
            okL &= (sv >= thr) & (dv >= 2)
            okS &= (sv >= thr) & (dv <= -2)
        out = base.copy()
        out[(base > 0) & ~okL] = 0
        out[(base < 0) & ~okS] = 0
        return out
    return sig

g5 = make_gated(["15min", "120min", "240min"])   # 5m entry -> 15m/2h/4h gate
g15 = make_gated(["120min", "240min"])           # 15m entry -> 2h/4h gate

if __name__ == "__main__":
    print("\n########## NET (cost=1.5pip, sslip 0.5) ##########")
    r5 = evaluate("UJ_M5c", g5, rr=2.0, stop_slip=0.5, _return=True)
    r15 = evaluate("UJ_M15", g15, rr=2.0, stop_slip=0.5, _return=True)
    print("\n########## GROSS (cost=0) — does the gate create entry edge? ##########")
    evaluate("UJ_M5c", g5, rr=2.0, cost=0.0)
    evaluate("UJ_M15", g15, rr=2.0, cost=0.0)
    print("\n########## causal checks ##########")
    check_causal(g5, EH.load_mt5_csv(LADDERS["UJ_M5c"][0]).resample("5min").agg(AGG).dropna())
    check_causal(g15, EH.load_mt5_csv(LADDERS["UJ_M15"][0]).resample("15min").agg(AGG).dropna())
    print("\n########## random-drop null: does the gate beat trimming N at random? ##########")
    b5 = evaluate("UJ_M5c", sar_ema, rr=2.0, stop_slip=0.5, quiet=True, _return=True, beta_trials=0)
    b15 = evaluate("UJ_M15", sar_ema, rr=2.0, stop_slip=0.5, quiet=True, _return=True, beta_trials=0)
    for tf, base, kept, nm in [("5m", b5, r5, "5m"), ("15m", b15, r15, "15m")]:
        if tf in base and tf in kept:
            yrs = base[tf][1]["ny"]
            print(f"  [{nm}] base N={base[tf][1]['N']} kept N={kept[tf][1]['N']}")
            random_drop_null(base[tf][0], kept[tf][0], yrs)
