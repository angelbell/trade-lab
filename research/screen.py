"""巡行幅（MFE/MAE）の一次スクリーン — 新しい入口は必ずここを最初に通す。

`mfe_mae.py` は --entry が breakout/swing/meanrev/fvg/fvgfill の5択で、イベント起点の入口が
書けない。そこが今回 FOMC の巡行幅を最後まで測らなかった実務上の理由なので、ここでは
「入口の一覧（時刻・方向・建値・損切り値）」だけを受け取り、入口の作り方には関与しない。

成果物 `research/screens/<name>.json` を書く。フック（.claude/hooks/screen_gate.py）が
この成果物の存在を見て、トレード統計を出すスクリプトの実行可否を判定する。

判定線（`mfe_mae.py` と同じ）:
    比 < 1.0   入口の方向にエッジ無し。どんな出口でも救えない
    1.0-1.2   境界。出口とサイジングが余程でなければ非推奨
    > 1.2     方向のエッジあり。深い検証に進む価値がある

使い方（スクリプトから）:
    from research.screen import run_screen
    run_screen("fomc_scalp_m1", df, entries, windows=[5, 10, 15])
      entries: [(t_entry, direction, entry_price, stop_price), ...]
      direction: +1 ロング / -1 ショート
      stop_price: 1R を定義する価格。None なら R 単位の出力を省く
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

SCREEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screens")


def run_screen(name, df, entries, windows=(5, 10, 15), quiet=False):
    """巡行幅を測り research/screens/<name>.json を書く。結果 dict を返す。"""
    if not entries:
        raise ValueError("entries が空です")
    os.makedirs(SCREEN_DIR, exist_ok=True)
    t_end = df.index.max()
    out = {"name": name, "n_entries": len(entries),
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "windows": {}}

    for W in windows:
        mfe, mae, mfeR, maeR = [], [], [], []
        for t, d, pe, sl in entries:
            t_out = t + pd.Timedelta(minutes=W)
            if t_out > t_end or pe is None or not np.isfinite(pe) or pe <= 0:
                continue
            b = df.loc[t:t_out]
            if len(b) < 2:
                continue
            up, dn = b["high"].max() - pe, b["low"].min() - pe
            f, a = (up, dn) if d > 0 else (-dn, -up)
            mfe.append(f / pe)
            mae.append(a / pe)
            if sl is not None and np.isfinite(sl) and abs(pe - sl) > 0:
                R = abs(pe - sl)
                mfeR.append(f / R)
                maeR.append(a / R)
        if len(mfe) < 5:
            continue
        mfe, mae = np.array(mfe), np.array(mae)
        ratio = float(np.median(mfe) / abs(np.median(mae))) if np.median(mae) != 0 else float("nan")
        w = {"n": len(mfe),
             "mfe_median_pct": float(np.median(mfe) * 100),
             "mfe_std_pct": float(mfe.std(ddof=1) * 100),
             "mae_median_pct": float(np.median(mae) * 100),
             "mae_std_pct": float(mae.std(ddof=1) * 100),
             "ratio_median": ratio,
             "verdict": "死(<1.0)" if ratio < 1.0 else ("境界(1.0-1.2)" if ratio < 1.2 else "深掘り可(>1.2)")}
        if mfeR:
            mfeR, maeR = np.array(mfeR), np.array(maeR)
            w.update({"mfe_median_R": float(np.median(mfeR)), "mfe_std_R": float(mfeR.std(ddof=1)),
                      "mae_median_R": float(np.median(maeR)), "mae_std_R": float(maeR.std(ddof=1)),
                      "reach": {f"{x}R": float(np.mean(mfeR >= x)) for x in (0.5, 1.0, 1.5, 2.0, 3.0)},
                      "stop_hit": float(np.mean(maeR <= -1.0))})
        out["windows"][str(W)] = w

    path = os.path.join(SCREEN_DIR, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    if not quiet:
        print(f"[screen] {name}  n={out['n_entries']}  -> {path}")
        for W, w in out["windows"].items():
            line = (f"  窓{W:>4}分 n={w['n']:>4} | MFE 中央値 {w['mfe_median_pct']:+.3f}% "
                    f"(σ{w['mfe_std_pct']:.3f}) | MAE 中央値 {w['mae_median_pct']:+.3f}% "
                    f"(σ{w['mae_std_pct']:.3f}) | 比 {w['ratio_median']:.2f} {w['verdict']}")
            if "mfe_median_R" in w:
                line += (f" | R: MFE {w['mfe_median_R']:+.2f} / MAE {w['mae_median_R']:+.2f} "
                         f"/ 損切り到達 {w['stop_hit']*100:.0f}%")
            print(line)
    return out


def screen_exists(name):
    return os.path.exists(os.path.join(SCREEN_DIR, f"{name}.json"))
