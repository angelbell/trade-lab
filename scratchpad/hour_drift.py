"""「時間帯そのものに方向情報があるか」の直接テスト（2026-07-14）。
これまでのプラセボ窓は間接テスト（窓をずらしても悪くならない）だった。ここでは直接:
  NY時刻の各時間について、15分足リターンの平均（bps）・t値・時代別の符号安定性を測る。
方向情報があるなら「その時間は平均して上がる/下がる」が時代をまたいで残るはず。
比較のため「どれだけ動くか」(値幅/ATR) も併記する。
Run: .venv/bin/python scratchpad/hour_drift.py"""
import sys, io, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ict_killzone import load_ny, SYMS

ERAS = [(2000, 2008), (2009, 2017), (2018, 2026)]

for name in ("gold", "eurusd", "gbpusd", "usdjpy", "btcusd"):
    with contextlib.redirect_stderr(io.StringIO()):
        df, _ = load_ny(SYMS[name], cut2000=(name == "usdjpy"))
    df["ret"] = df["close"].pct_change() * 1e4          # bps per 15m bar
    df["yr"] = pd.to_datetime(df["broker_dt"]).dt.year
    df["rng"] = (df["high"] - df["low"]) / df["atr14"]
    d = df.dropna(subset=["ret", "atr14"])
    d = d[np.isfinite(d["ret"])]
    print(f"\n=== {name} ===  (NY時刻・15分足リターン bps / t値 / 時代別の符号)  n={len(d)}")
    print(f"  {'NYh':>4} {'meanRet(bps)':>13} {'t':>6} {'n':>7} {'値幅/ATR':>9}  時代別平均(bps)")
    for h in range(24):
        s = d[d["ny_hour"] == h]
        if len(s) < 200:
            continue
        r = s["ret"].values
        t = r.mean() / (r.std() / np.sqrt(len(r))) if r.std() > 0 else np.nan
        eras = []
        for a, b in ERAS:
            e = s[(s["yr"] >= a) & (s["yr"] <= b)]["ret"]
            eras.append(f"{e.mean():+6.2f}" if len(e) > 200 else "   n/a")
        tag = " <-KZ" if 7 <= h <= 9 else (" <-LON" if 2 <= h <= 5 else "")
        star = " *" if abs(t) > 2.5 else ""
        print(f"  {h:4d} {r.mean():+13.3f} {t:+6.2f} {len(r):7d} {s['rng'].mean():9.3f}  "
              f"{' '.join(eras)}{tag}{star}")
