"""対立仮説の検定: 「拡大足が情報を持つ」vs「BTC h1 は素のバー間継続性を持つだけ」。

measure の報告は k=1.5/2.0/2.5 の用量反応を示したが、k=0（＝方向条件だけの全陽線/全陰線）と
比べていない。チェックリスト1「素の全信号をまず見る — フィルタはエッジを濃縮するだけで、
無からは作らない」の対照がそこ。あわせて年別の広がり（チェックリスト5）も出す。
"""
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv  # noqa: E402


def wilder_atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def prep(path, start, fwd):
    df = load_mt5_csv(path)
    if start:
        df = df.loc[start:]
    atr_prev = wilder_atr(df).shift(1)
    # 入口 e から fwd 本（e .. e+fwd-1）の最高値/最安値をベクトルで
    fmax = df["high"][::-1].rolling(fwd).max()[::-1]
    lmin = df["low"][::-1].rolling(fwd).min()[::-1]
    return df, atr_prev.to_numpy(), fmax.to_numpy(), lmin.to_numpy()


def cell(df, atr_prev, fmax, lmin, k, direction, fwd, width="body"):
    o = df["open"].to_numpy()
    body = (df["close"] - df["open"]).abs().to_numpy()
    full = (df["high"] - df["low"]).to_numpy()
    w = body if width == "body" else full
    up = (df["close"] > df["open"]).to_numpy()
    side = up if direction > 0 else ~up
    ok = side & np.isfinite(atr_prev) & (atr_prev > 0)
    if k > 0:
        ok &= w > atr_prev * k
    s = np.flatnonzero(ok)
    s = s[(s + 1 + fwd) <= len(df) - 1]
    e = s + 1
    R = atr_prev[s]
    pe = o[e]
    u, d = fmax[e] - pe, lmin[e] - pe
    f, a = (u, d) if direction > 0 else (-d, -u)
    mfe, mae = f / R, a / R
    good = np.isfinite(mfe) & np.isfinite(mae)
    return s[good], mfe[good], mae[good]


def ratio(mfe, mae):
    m = np.median(mae)
    return float(np.median(mfe) / abs(m)) if m != 0 else float("nan")


FWD = 20
for path, start, label in [("data/vantage_btcusd_h1.csv", None, "BTC h1"),
                           ("data/vantage_btcusd_m15.csv", "2018-10-01", "BTC m15")]:
    df, atr_prev, fmax, lmin = prep(path, start, FWD)
    span = (df.index[-1] - df.index[0]).days / 365.25
    print(f"\n===== {label} body / fwd={FWD}本  （k=0 は方向条件のみ＝素の母集団）")
    print(f"{'k':>5} {'方向':>4} {'N':>7} {'N/年':>7} {'MFE中央':>8} {'MAE中央':>8} {'比':>6}")
    store = {}
    for direction, dname in ((+1, "long"), (-1, "short")):
        for k in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
            s, mfe, mae = cell(df, atr_prev, fmax, lmin, k, direction, FWD)
            r = ratio(mfe, mae)
            store[(k, direction)] = (s, mfe, mae, r)
            print(f"{k:5.1f} {dname:>5} {len(mfe):7d} {len(mfe)/span:7.1f} "
                  f"{np.median(mfe):8.3f} {np.median(mae):8.3f} {r:6.3f}")

    # 年別の広がり（k=2.0 と、その素の母集団 k=0 を並べる）
    print(f"\n----- {label} 年別の比（long）: k=2.0 と素の母集団 k=0")
    yrs = sorted(set(df.index.year))
    print(f"{'年':>6} {'N(k2)':>6} {'比(k2)':>7} {'比(k0)':>7}")
    for y in yrs:
        row = [y]
        for k in (2.0, 0.0):
            s, mfe, mae, _ = store[(k, +1)]
            m = df.index[s].year == y
            row.append((m.sum(), ratio(mfe[m], mae[m]) if m.sum() >= 10 else float("nan")))
        print(f"{row[0]:6d} {row[1][0]:6d} {row[1][1]:7.3f} {row[2][1]:7.3f}")

    if label == "BTC h1":
        # 検算: 既に独立照合済みの値と一致すること（ベクトル化が壊れていないこと）
        assert store[(2.0, +1)][3] > 1.50 and store[(2.0, +1)][3] < 1.59, store[(2.0, +1)][3]
        assert len(store[(2.0, +1)][1]) == 701, len(store[(2.0, +1)][1])
        print("\nOK: k=2.0 long がループ版の n=701 / 比1.5446 と一致")
