"""radar_gate_race.py -- HEAD-TO-HEAD: existing HTF gates vs the radar, DECOMPOSED by component.

Question (user): as the deploy gate for the 15m legs, which is better -- the existing gates
(daily SMA150+slope for gold / KAMA-rising for BTC) or the trend-strength radar, and which
radar COMPONENT (ER / ADX / EMA-stack / slope / ATR-expansion) carries the weight?

Method: regenerate each 15m leg UNGATED via breakout_wave.run (gate off, ext-cap off), then
post-hoc label entries with each causal gate (HTF confirmed bar shift(1) -> ffill). To keep
the horse race fair, every "direction & component>=q" gate has its threshold q chosen so its
bar-level ON%% MATCHES the instrument's existing gate ON%% (selectivity-matched; a tighter
gate always looks prettier per-trade). Natural-threshold cells (s>=5, stack==3) also shown.
Net absolute cost post-hoc via risk column: gold $0.6, BTC $15. Gold frac=0.25, BTC frac=0.3.
NOTE: post-hoc labeling ignores the no-overlap re-arm difference (same caveat for all cells;
comparison is internally consistent). Gold cells exclude the ext-cap8 add-on except the
canon reference row (ext-cap is an entry filter, not a regime gate).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample, kama_adaptive

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}
BASE = dict(pattern="B", sl_mode="line", sl_buf=0.25, swing="zigzag", zz_k=2.0,
            pivot_n=5, renko_k=2.0, mom_fast=12, mom_slow=26, trend_ema=80,
            bo_window=20, tp_mode="rr", rr=4.0, atr=14, cost=0.0, swap_pct=0.0,
            fwd=500, peryear=False, start=None, end=None, daily_sma=0,
            daily_slope_k=0, risk=0.01, gate_kama=0, tf="15min", csv="")


def comps_tf(d15, rule):
    """Radar components on rule-TF, confirmed (shift 1), ffilled to the 15m index."""
    d = d15.resample(rule).agg(AGG).dropna()
    c = d["close"]
    atr = ta.atr(d["high"], d["low"], d["close"], 14)
    er = (c.diff(20).abs() / c.diff().abs().rolling(20).sum()).clip(0, 1)
    adxN = ((ta.adx(d["high"], d["low"], d["close"], 14)["ADX_14"] - 15) / 25).clip(0, 1)
    emaF = c.ewm(span=20, adjust=False).mean()
    emaS = c.ewm(span=50, adjust=False).mean()
    stack = np.sign(c - emaF) + np.sign(emaF - emaS) + np.sign(emaS - emaS.shift(10))
    align = stack.abs() / 3.0
    slopeN = ((emaS - emaS.shift(10)).abs() / (atr * 1.5)).clip(0, 1)
    atrexpN = ((atr / atr.rolling(100).mean() - 0.8) / 0.7).clip(0, 1)
    s10 = 10.0 * (er.fillna(0) + adxN.fillna(0) + align.fillna(0)
                  + slopeN.fillna(0) + atrexpN.fillna(0)) / 5.0
    out = {}
    for k, v in [("er", er), ("adx", adxN), ("slope", slopeN), ("atrexp", atrexpN),
                 ("s10", s10), ("stack", stack)]:
        out[k] = v.shift(1).reindex(d15.index, method="ffill").values
    return out


def kama_up(d15, rule):
    dc = d15["close"].resample(rule).last().dropna()
    km = kama_adaptive(dc, 14)
    return ((km > km.shift(1)).shift(1)
            .reindex(d15.index, method="ffill").fillna(False).values.astype(bool))


def sma_gate(d15, n=150, k=10, cap=None):
    dc = d15["close"].resample("1D").last().dropna()
    sma = dc.rolling(n).mean()
    up = (dc > sma) & (sma > sma.shift(k))
    g = up.shift(1).reindex(d15.index, method="ffill").fillna(False).values
    if cap is not None:
        ext = ((dc - sma) / sma * 100.0).shift(1).reindex(d15.index, method="ffill").values
        g = g & (np.isnan(ext) | (ext <= cap))
    return g.astype(bool)


def matched(cond, comp, target_on):
    """direction cond AND comp>=q, q set so bar-level ON%% == target_on (if possible)."""
    ok = cond & ~np.isnan(comp)
    if ok.mean() <= target_on:
        return ok
    keep = target_on / ok.mean()
    q = np.nanquantile(comp[ok], 1.0 - keep)
    return ok & (comp >= q)


def cell(tag, Rn, yr, span):
    if len(Rn) < 5:
        print(f"  {tag:>22}: n={len(Rn)} (too few)")
        return
    pf = Rn[Rn > 0].sum() / abs(Rn[Rn <= 0].sum()) if (Rn <= 0).any() else float("inf")
    eq = np.cumsum(Rn)
    dd = (np.maximum.accumulate(eq) - eq).max()
    ys = np.unique(yr)
    green = sum(Rn[yr == y].sum() > 0 for y in ys)
    print(f"  {tag:>22}: N/yr={len(Rn)/span:5.1f}  win={(Rn>0).mean()*100:4.1f}%  meanR={Rn.mean():+.3f}"
          f"  PF={pf:4.2f}  totR/yr={Rn.sum()/span:+6.1f}  maxDD={dd:5.1f}R  ret/DD={Rn.sum()/dd:5.2f}"
          f"  green={green}/{len(ys)}")


def race(name, csv, frac, rt, ref_gate_builder, extra_rows):
    d = load_mt5_csv(csv)
    if name == "BTC":
        d = d[d.index >= "2018-10-01"]
    d15 = resample(d, "15min")
    span = (d15.index[-1] - d15.index[0]).days / 365.25
    t = run(d15, SimpleNamespace(**BASE, pullback_frac=frac))
    Rn = t["R"].values - rt / t["risk"].values
    yr = t["time"].dt.year.values
    pos = d15.index.get_indexer(t["time"])

    ref = ref_gate_builder(d15)                    # the instrument's existing gate
    target = ref.mean()
    C4 = comps_tf(d15, "240min")
    up4 = C4["stack"] > 0

    gates = {}
    for tag, g in extra_rows(d15, ref):
        gates[tag] = g
    gates["stack4h>0 (向きのみ)"] = up4
    gates["stack4h==3 (完全整列)"] = C4["stack"] >= 3
    for ck, lab in [("er", "ER"), ("adx", "ADX"), ("slope", "slope"), ("atrexp", "ATRexp"),
                    ("s10", "合成s")]:
        gates[f"up4h&{lab} (ON%一致)"] = matched(up4, C4[ck], target)
    gates["radar4h s>=5 (素)"] = up4 & (C4["s10"] >= 5.0)

    print(f"\n===== {name} 15m leg (frac{frac}, net ${rt}), {span:.1f}yr, "
          f"既存ゲートON%={target*100:.0f}% (選別率をここに一致) =====")
    cell("UNGATED base", Rn, yr, span)
    for tag, g in gates.items():
        m = g[pos]
        cell(f"{tag} [ON{g.mean()*100:3.0f}%]", Rn[m], yr[m], span)


def main():
    def gold_ref(d15):
        return sma_gate(d15, 150, 10, cap=None)

    def gold_rows(d15, ref):
        return [("SMA150d+slope (既存)", ref),
                ("既存+extcap8 (正典)", sma_gate(d15, 150, 10, cap=8.0)),
                ("kama1d", kama_up(d15, "1D")),
                ("kama4h", kama_up(d15, "240min"))]

    def btc_ref(d15):
        return kama_up(d15, "240min")

    def btc_rows(d15, ref):
        k1 = kama_up(d15, "1D")
        return [("kama1d (正典)", k1),
                ("kama4h (C1)", ref),
                ("kama4h&1d (C1)", ref & k1)]

    race("GOLD", "data/vantage_xauusd_m15.csv", 0.25, 0.6, gold_ref, gold_rows)
    race("BTC", "data/vantage_btcusd_m15.csv", 0.3, 15.0, btc_ref, btc_rows)


if __name__ == "__main__":
    main()
