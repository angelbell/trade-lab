"""measure が返した ATR拡大足スクリーンの独立照合。

同じスクリプトを回し直しても同じ実装ミスが再現するだけなので、ATR も走査も帰無も
ここで書き直す（pandas_ta を使わず Wilder RMA を自前実装）。照合対象は BTC h1 body k=2.0 long
の見出し数字と、両方向で立つという主張。
"""
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv  # noqa: E402


def wilder_atr(df, n=14):
    """pandas_ta を使わない Wilder ATR。"""
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def excursions(df, idx, direction, fwd):
    """idx = 引き金足の位置。入口 = idx+1 の始値。走査 = idx+1 .. idx+fwd。"""
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    atr = wilder_atr(df).shift(1).to_numpy()  # s-1 までで確定した ATR
    mfe, mae = [], []
    for s in idx:
        e = s + 1
        if e + fwd > len(df):
            continue
        R = atr[s]
        if not np.isfinite(R) or R <= 0:
            continue
        pe = o[e]
        up = h[e:e + fwd].max() - pe
        dn = l[e:e + fwd].min() - pe
        f, a = (up, dn) if direction > 0 else (-dn, -up)
        mfe.append(f / R)
        mae.append(a / R)
    return np.array(mfe), np.array(mae)


def ratio_of(mfe, mae):
    return float(np.median(mfe) / abs(np.median(mae)))


def run(path, start, k, width, direction, fwd, reps=200, seed=7):
    df = load_mt5_csv(path)
    if start:
        df = df.loc[start:]
    atr_prev = wilder_atr(df).shift(1)
    body = (df["close"] - df["open"]).abs()
    full = df["high"] - df["low"]
    w = body if width == "body" else full
    side = (df["close"] > df["open"]) if direction > 0 else (df["close"] < df["open"])
    hit = (w > atr_prev * k) & side & atr_prev.notna()
    idx = np.flatnonzero(hit.to_numpy())

    mfe, mae = excursions(df, idx, direction, fwd)
    r = ratio_of(mfe, mae)

    # 帰無: 同数・同じ時間帯分布・同方向のランダム建て
    hours = df.index.hour.to_numpy()
    trig_hours = hours[idx]
    valid = np.flatnonzero(np.isfinite(atr_prev.to_numpy()) & (np.arange(len(df)) < len(df) - fwd - 1))
    pool = {h: valid[hours[valid] == h] for h in np.unique(trig_hours)}
    rng = np.random.default_rng(seed)
    null = []
    cnt = pd.Series(trig_hours).value_counts()
    for _ in range(reps):
        pick = np.concatenate([rng.choice(pool[h], size=int(c), replace=False) for h, c in cnt.items()])
        nf, na = excursions(df, np.sort(pick), direction, fwd)
        null.append(ratio_of(nf, na))
    null = np.array(null)
    pct = float((null < r).mean() * 100)

    span = (df.index[-1] - df.index[0]).days / 365.25
    return {"n": len(mfe), "per_year": len(mfe) / span, "mfe_med": float(np.median(mfe)),
            "mae_med": float(np.median(mae)), "ratio": r, "mfe_std": float(mfe.std(ddof=1)),
            "stop_hit_1R": float((mae <= -1.0).mean()),
            "reach": {x: float((mfe >= x).mean()) for x in (1.0, 2.0, 3.0)},
            "null_med": float(np.median(null)), "null_std": float(null.std(ddof=1)), "pctile": pct}


BTC = "data/vantage_btcusd_h1.csv"

print("=== BTC h1 body k=2.0 long fwd=20 （measure 報告: n=701 N/年=76.7 比=1.542 %ile=100）")
a = run(BTC, None, 2.0, "body", +1, 20)
print(a)

print("=== BTC h1 body k=2.0 short fwd=20 （measure 報告: n=721 比=1.307 %ile=100）")
b = run(BTC, None, 2.0, "body", -1, 20)
print(b)

print("=== BTC h1 body k=1.5 long fwd=20 （measure 報告: n=1372 比=1.289）")
c = run(BTC, None, 1.5, "body", +1, 20)
print(c)

print("=== BTC h1 body k=2.5 long fwd=20 （measure 報告: n=388 比=1.492）")
d = run(BTC, None, 2.5, "body", +1, 20)
print(d)

# 検算: 既知の値への数値 assert（印字して目視は検算ではない）
assert 600 <= a["n"] <= 800, a["n"]
assert 1.30 <= a["ratio"] <= 1.80, a["ratio"]
assert a["pctile"] >= 95, a["pctile"]
assert 0.90 <= a["null_med"] <= 1.10, a["null_med"]
assert 1.10 <= b["ratio"] <= 1.55, b["ratio"]
assert b["pctile"] >= 95, b["pctile"]
# 用量反応（k を上げると比が上がる）が独立実装でも出るか
assert c["ratio"] < d["ratio"], (c["ratio"], d["ratio"])
print("\nOK: 独立実装で見出し数字を再現")
