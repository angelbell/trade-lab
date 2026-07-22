"""【仕様カード第6段】ATR拡大足機構の全銘柄横展開 ― STEP1: 巡行幅スクリーン。

引き金・走査は BTC h1 で凍結した定義をそのまま使う（自前で書き直さない）。
実体 > ATR(14)[s-1] * k、k in {0, 1.5, 2.0, 2.5}（k=0 = 素の全陽線/全陰線）。
前方20本・R単位=ATR×1・時間帯一致ランダム建て帰無200回。ratio = MFE中央値 / |MAE中央値|。

先読み無し: ATR は shift(1) 済み、走査は e=s+1 の始値から。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# (name, path, start_or_None, note)
INSTR = [
    ("eurusd", "data/vantage_eurusd_h1.csv", None, ""),
    ("gbpusd", "data/vantage_gbpusd_h1.csv", None, ""),
    ("audusd", "data/vantage_audusd_h1.csv", None, ""),
    ("nzdusd", "data/vantage_nzdusd_h1.csv", None, ""),
    ("usdcad", "data/vantage_usdcad_h1.csv", None, ""),
    ("usdjpy", "data/vantage_usdjpy_h1.csv", None, ""),
    ("usdx.r", "data/vantage_usdx.r_h1.csv", "2024-04-01", "2024-04以前は月20-25本の疎データ(罠)＝切る"),
    ("nas100.r", "data/vantage_nas100.r_h1.csv", "2018-03-01", "2018-03以前は月20-25本の疎データ(罠)＝切る"),
    ("ger40.r", "data/vantage_ger40.r_h1.csv", "2017-11-01", "2017-11以前は月20-25本の疎データ(罠)＝切る"),
    ("us2000.r", "data/vantage_us2000.r_h1.csv", None, "2020-04開始が実データ開始(自然な部分年、罠ではない)"),
    ("spx", "data/vantage_spx_h1.csv", None, "2022-03開始が実データ開始(自然な部分年)。全期間4.3年と短い"),
    ("usousd", "data/vantage_usousd_h1.csv", "2018-03-01", "2018-03以前は月20-25本の疎データ(罠)＝切る"),
    ("xagusd", "data/vantage_xagusd_h1.csv", "2018-03-01", "2018-03以前は月20-25本の疎データ(罠)＝切る"),
    ("xptusd.r", "data/vantage_xptusd.r_h1.csv", None, "2022-02開始が実データ開始(自然な部分年)。全期間4.4年と短い"),
    ("copper-cr", "data/vantage_copper-cr_h1.csv", "2018-03-01", "2018-03以前は月20-25本の疎データ(罠)＝切る"),
    ("xauusd", "data/vantage_xauusd_h1.csv", "2018-01-01", "既定の運用規約どおり(真の密化は2018-03、規約に合わせる)"),
    ("btcusd", "data/vantage_btcusd_h1.csv", None, "アンカー。2017-05〜2018-03は月20-90本と疎(同型の罠)だが"
                                                     "確立済みアンカー値がこの未トリム全期間で定義されているため、"
                                                     "tie-back優先で切らない(下で別途トリガー数を確認)"),
]


def wilder_atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _rolling_extrema(df, fwd):
    """事前計算: roll_hi[j] = max(high[j-fwd+1 : j+1])、roll_lo[j] = min(low[...])。
    s の走査幅 high[s+1:s+1+fwd] は j=s+fwd の位置に一致する（先読み無し、pandas rolling は
    確定足のみを見る標準機能）。ベクトル化のための1回だけの前計算＝Pythonループを避ける高速化で、
    ロジックは元の逐次版（各 s ごとに h[e:e+fwd].max() を取る）と数学的に同一。"""
    roll_hi = df["high"].rolling(fwd).max().to_numpy()
    roll_lo = df["low"].rolling(fwd).min().to_numpy()
    return roll_hi, roll_lo


def excursions(df, idx, direction, fwd, roll_hi=None, roll_lo=None):
    o = df["open"].to_numpy()
    atr = wilder_atr(df).shift(1).to_numpy()
    if roll_hi is None or roll_lo is None:
        roll_hi, roll_lo = _rolling_extrema(df, fwd)
    n = len(df)
    idx = idx[idx + fwd < n]
    if len(idx) == 0:
        return np.array([]), np.array([])
    e = idx + 1
    R = atr[idx]
    valid = np.isfinite(R) & (R > 0)
    idx, e, R = idx[valid], e[valid], R[valid]
    pe = o[e]
    up = roll_hi[idx + fwd] - pe
    dn = roll_lo[idx + fwd] - pe
    if direction > 0:
        f, a = up, dn
    else:
        f, a = -dn, -up
    return f / R, a / R


def ratio_of(mfe, mae):
    if len(mfe) == 0 or np.median(mae) == 0:
        return float("nan")
    return float(np.median(mfe) / abs(np.median(mae)))


def screen_one(df, k, direction, fwd=20, reps=200, seed=7):
    atr_prev = wilder_atr(df).shift(1)
    body = (df["close"] - df["open"]).abs()
    side = (df["close"] > df["open"]) if direction > 0 else (df["close"] < df["open"])
    if k == 0.0:
        hit = side & atr_prev.notna()
    else:
        hit = (body > atr_prev * k) & side & atr_prev.notna()
    idx = np.flatnonzero(hit.to_numpy())
    idx = idx[idx + fwd + 1 < len(df)]

    mfe, mae = excursions(df, idx, direction, fwd)
    r = ratio_of(mfe, mae)
    if len(mfe) < 20:
        span = (df.index[-1] - df.index[0]).days / 365.25
        return {"n": len(mfe), "per_year": len(mfe) / span if span > 0 else float("nan"),
                "mfe_med": float(np.median(mfe)) if len(mfe) else float("nan"),
                "mae_med": float(np.median(mae)) if len(mae) else float("nan"),
                "ratio": r, "null_med": float("nan"), "null_std": float("nan"), "pctile": float("nan")}

    hours = df.index.hour.to_numpy()
    trig_hours = hours[idx]
    valid = np.flatnonzero(np.isfinite(atr_prev.to_numpy()) & (np.arange(len(df)) < len(df) - fwd - 1))
    pool = {hh: valid[hours[valid] == hh] for hh in np.unique(trig_hours)}
    rng = np.random.default_rng(seed)
    null = []
    cnt = pd.Series(trig_hours).value_counts()
    for _ in range(reps):
        pieces = []
        for hh, c in cnt.items():
            avail = pool[hh]
            if len(avail) < c:
                pieces.append(rng.choice(avail, size=int(c), replace=True))
            else:
                pieces.append(rng.choice(avail, size=int(c), replace=False))
        pick = np.concatenate(pieces)
        nf, na = excursions(df, np.sort(pick), direction, fwd)
        null.append(ratio_of(nf, na))
    null = np.array(null)
    pct = float((null < r).mean() * 100)

    span = (df.index[-1] - df.index[0]).days / 365.25
    return {"n": len(mfe), "per_year": len(mfe) / span if span > 0 else float("nan"),
            "mfe_med": float(np.median(mfe)), "mae_med": float(np.median(mae)), "ratio": r,
            "null_med": float(np.median(null)), "null_std": float(np.std(null, ddof=1)), "pctile": pct}


def main(smoke):
    Ks = [0.0, 1.5, 2.0, 2.5]
    reps = 40 if smoke else 200
    results = []
    for name, path, start, note in INSTR:
        df = load_mt5_csv(path)
        if start:
            df = df.loc[start:]
        if smoke:
            df = df.iloc[-20000:]  # 部分データでスモーク
        span = (df.index[-1] - df.index[0]).days / 365.25
        n_bars = len(df)
        print(f"\n{'=' * 100}\n### {name}  期間={df.index[0].date()}..{df.index[-1].date()} "
              f"span={span:.2f}年 本数={n_bars} 本/年={n_bars / span:.0f}  {note}")
        for direction, dname in ((+1, "long"), (-1, "short")):
            print(f"\n  --- {dname}")
            print(f"  {'k':>5} {'n':>6} {'N/年':>7} {'MFE中央値':>9} {'MAE中央値':>9} {'比':>6} "
                  f"{'帰無中央値':>9} {'帰無std':>7} {'%ile':>6}")
            for k in Ks:
                r = screen_one(df, k, direction, fwd=20, reps=reps)
                print(f"  {k:5.1f} {r['n']:6d} {r['per_year']:7.1f} {r['mfe_med']:9.3f} "
                      f"{r['mae_med']:9.3f} {r['ratio']:6.3f} {r['null_med']:9.3f} "
                      f"{r['null_std']:7.3f} {r['pctile']:6.1f}")
                results.append(dict(instr=name, direction=dname, k=k, **r))
    out_path = "scratchpad/out_atr_transplant_step1_results.json" if not smoke else \
        "scratchpad/out_atr_transplant_step1_results_smoke.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"\n結果を {out_path} に保存")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    res = main(args.smoke)

    # 検算: BTC h1 ロング k=2.0 が既知の水準(比1.545・帰無1.02付近・%ile 100)を再現すること
    btc_long_k2 = [r for r in res if r["instr"] == "btcusd" and r["direction"] == "long" and r["k"] == 2.0][0]
    if not args.smoke:
        assert 600 <= btc_long_k2["n"] <= 800, btc_long_k2["n"]
        assert 1.30 <= btc_long_k2["ratio"] <= 1.80, btc_long_k2["ratio"]
        assert btc_long_k2["pctile"] >= 95, btc_long_k2["pctile"]
        # gold h1 (2018-) がこの機構で死んでいること(比<1.2)を確認
        gold_long_k2 = [r for r in res if r["instr"] == "xauusd" and r["direction"] == "long" and r["k"] == 2.0][0]
        assert gold_long_k2["ratio"] < 1.2, gold_long_k2["ratio"]
        print(f"\nOK: BTC h1 ロング k=2.0 比={btc_long_k2['ratio']:.3f} (%ile={btc_long_k2['pctile']:.1f}) 再現、"
              f"gold h1 比={gold_long_k2['ratio']:.3f} <1.2 で死亡を確認")
    else:
        print(f"\n[smoke] BTC k=2.0 ロング n={btc_long_k2['n']} 比={btc_long_k2['ratio']:.3f}")
