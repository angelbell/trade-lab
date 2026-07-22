"""STEP2 実行ドライバ。atr_spike_transplant_step2.py の部品を使い、STEP1 を通った 銘柄×方向 だけに
凍結仕様をそのまま当てる。コスト（往復・価格に対する割合換算、根拠明記）は下の COST_SPEC で
銘柄ごとに定義する。FX/gold/BTC は正典値、指数・その他コモディティは仮定を明記した梯子(1x/2x/4x)。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os
import argparse
import json

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

from src.data_loader import load_mt5_csv                                    # noqa: E402
from src.engine.mirror import invert                                        # noqa: E402
from atr_spike_transplant_step2 import (atr_prev_of, raw_triggers, build_pdh_dist,  # noqa: E402
                                         build_entries, run_cell, stats, per_year_rows,
                                         fmt, drop_null, pf_of)
from atr_spike_transplant_step1 import INSTR                                # noqa: E402

PATHS = {name: (path, start) for name, path, start, _note in INSTR}

# ---------------------------------------------------------------- コスト仕様（往復・価格%換算）
# ("fx", pip_size) / ("fixed", 往復$) / ("ladder", 往復の仮定額・1x)
COST_SPEC = {
    "eurusd": ("fx", 0.0001), "gbpusd": ("fx", 0.0001), "audusd": ("fx", 0.0001),
    "nzdusd": ("fx", 0.0001), "usdcad": ("fx", 0.0001), "usdjpy": ("fx", 0.01),
    "xauusd": ("fixed", 0.6),          # 正典 $0.6/oz（実勢0.15-0.35の保守側）
    "btcusd": ("fixed", 15.0),         # 正典 $15
    "usdx.r": ("ladder", 0.05),        # ドル指数CFD、往復0.05pt(スプレッド1-2pt相当)を仮置き
    "nas100.r": ("ladder", 2.0),       # 指数CFD、往復2.0pt(スプレッド1-2pt)を仮置き
    "ger40.r": ("ladder", 2.0),
    "us2000.r": ("ladder", 1.0),
    "spx": ("ladder", 0.5),
    "usousd": ("ladder", 0.05),        # 原油、往復$0.05(5セント)を仮置き
    "xagusd": ("ladder", 0.03),        # 銀、往復$0.03を仮置き
    "xptusd.r": ("ladder", 0.5),       # プラチナ、往復$0.5を仮置き
    "copper-cr": ("ladder", 0.002),    # 銅、往復$0.002を仮置き
}


def cost_frac_for(name, median_price, mult=1.0):
    kind, val = COST_SPEC[name]
    if kind == "fx":
        cost_abs = 0.9 * val
    elif kind == "fixed":
        cost_abs = val
    else:  # ladder
        cost_abs = val * mult
    return cost_abs / median_price, cost_abs


def load_side(name, direction):
    path, start = PATHS[name]
    df = load_mt5_csv(path)
    if start:
        df = df.loc[start:]
    if direction == "long":
        return df, None
    inv = invert(df)
    C = 2 * df["high"].max()
    return inv, C


def run_one(name, direction, k, mult=1.0, verbose=True):
    d, C = load_side(name, direction)
    span = (d.index[-1] - d.index[0]).days / 365.25
    atr_prev = atr_prev_of(d)
    s_idx = raw_triggers(d, atr_prev, k)
    if direction == "long":
        pdh = build_pdh_dist(d, atr_prev)
        s_idx = s_idx[pdh[s_idx] > 0.0]
        pf = 0.0
    else:
        pf = 0.5
    entries = build_entries(d, atr_prev, s_idx, rr=1000.0)
    price_ref = (C - d["close"]) if C is not None else d["close"]
    median_price = float(price_ref.median())
    cost_frac, cost_abs = cost_frac_for(name, median_price, mult)
    t = run_cell(d, entries, pf, cost_frac, C=C)
    if t is None or len(t) < 5:
        if verbose:
            print(f"  {name:>10} {direction:>5} k={k:.1f} mult={mult:g}: トレード数不足 (N={0 if t is None else len(t)})")
        return None
    s = stats(t, span)
    if verbose:
        print(f"  {name:>10} {direction:>5} k={k:.1f} mult={mult:g} cost_abs={cost_abs:.4g} "
              f"median_px={median_price:.4g} cost%={cost_frac*100:.4f}%  " + fmt(s))
    return dict(name=name, direction=direction, k=k, mult=mult, cost_abs=cost_abs,
                median_price=median_price, cost_frac=cost_frac, **s), t


def main(smoke, pairs, ks, mults):
    results = []
    for name, direction in pairs:
        print(f"\n{'=' * 100}\n### {name} / {direction}")
        for k in ks:
            for mult in mults:
                out = run_one(name, direction, k, mult)
                if out is not None:
                    s, t = out
                    results.append(s)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pairs", default="")     # "name:dir,name:dir"
    ap.add_argument("--ks", default="1.5,2.0,2.5")
    ap.add_argument("--mults", default="1.0")  # ladder銘柄のコスト梯子
    args = ap.parse_args()
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if p] or [("btcusd", "long"), ("btcusd", "short")]
    ks = [float(x) for x in args.ks.split(",")]
    mults = [float(x) for x in args.mults.split(",")]
    res = main(args.smoke, pairs, ks, mults)

    # ---- 検算1: BTC h1 ロング k=2.0（凍結仕様どおり=ATR3トレール・fwd20・pdh>0フィルタ）が
    #      既知のアンカー N/年42.0・勝率51.3%・PF2.12・平均+0.821%・maxDD13.2% を再現すること
    btc_l = [r for r in res if r["name"] == "btcusd" and r["direction"] == "long" and r["k"] == 2.0]
    if btc_l:
        b = btc_l[0]
        assert 41.0 < b["N_yr"] < 43.0, b["N_yr"]
        assert 50.5 < b["win"] < 52.0, b["win"]
        assert 2.05 < b["PF"] < 2.20, b["PF"]
        assert 0.80 < b["mean_pct"] < 0.85, b["mean_pct"]
        assert 12.5 < b["maxDD_pct"] < 14.0, b["maxDD_pct"]
        print(f"\nOK: BTC h1 ロング k=2.0 凍結仕様アンカーを再現: N/年={b['N_yr']:.1f} 勝率={b['win']:.1f}% "
              f"PF={b['PF']:.2f} 平均%={b['mean_pct']:+.3f} maxDD%={b['maxDD_pct']:.1f}")

    # ---- 検算2: BTC h1 ショート k=2.0 pf=0.5 のトレール実測を報告（凍結仕様どおり）。
    #      仕様カードの「短アンカー N/年50.0 PF1.27 平均+0.192%」は別の出口(固定RR3)で測られた値と
    #      判明（atr5c_trail_canon.py の検算=N457/PF1.27はRR3、トレールではない）。この不一致は
    #      報告で明示的にフラグする。
    btc_s = [r for r in res if r["name"] == "btcusd" and r["direction"] == "short" and r["k"] == 2.0]
    if btc_s:
        s = btc_s[0]
        print(f"\n[参考] BTC h1 ショート k=2.0 pf=0.5 実測(凍結仕様=ATR3トレール): N/年={s['N_yr']:.1f} "
              f"PF={s['PF']:.2f} 平均%={s['mean_pct']:+.3f} maxDD%={s['maxDD_pct']:.1f}")

    out_path = "scratchpad/out_atr_transplant_step2_results.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"\n結果を {out_path} に保存")
