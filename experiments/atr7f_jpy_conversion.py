"""【第7段・締め】円換算。10万円口座・0.01ロット刻み切り下げ。
(a) 現行どおり: 1トレードのリスク3%上限で採用ロットを決める版。
(b) 追加: ゲート版のロットを「ゲート無し版と同じmaxDD%になるまで」引き上げた版（法則7.5点検＝
    ばらつきを下げる操作が自動でレバレッジを買っていないかを見る）。玉固定版(a)と並べて報告する。
    (b)がリスク3%上限を超える場合はその超過率も表示する（3%ルールを破ってよいという意味ではない、
    「DD改善の正体がレバレッジかどうか」を測るための比較値）。
"""
SCREEN = "atr_spike_btc_h1"
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atr7_common import (load_frames, atr_prev_of, raw_triggers, build_entries, run_cell, stats,
                          span_years, weekly_up_regime, build_pdh_dist_series)  # noqa: E402

ACCOUNT = 100000.0
MAXRISK = 0.03
UNITS001 = 1000.0   # 0.01ロット=1,000通貨（USDJPY）
FX = 1.0             # 円建てなので換算不要


def build(df, k, system, stopk, val, fwd, gate):
    atr_prev = atr_prev_of(df)
    s_idx = raw_triggers(df, atr_prev, k)
    pdh = build_pdh_dist_series(df, atr_prev)
    mask = pdh[s_idx] > 0.0
    if gate:
        wu = weekly_up_regime(df)
        mask &= wu[s_idx]
    s_sel = s_idx[mask]
    ent = build_entries(df, atr_prev, s_sel, system, 0.0, stopk=stopk, trail=True)
    return run_cell(df, ent, fill_win=200, fwd=fwd, trail_atr=val)


def jpy_row(name, t, span, lots_override=None):
    """risk はpnl_px計算に使ったのと同じ価格単位(円)。t["risk"]=ストップ距離(円)、
    t["R"]*t["risk"]-cost が1ロット(=1通貨)あたりの円損益 -> ×1000×mult で0.01ロット単位に換算。"""
    risk = t["risk"].to_numpy()
    pnl_unit = t["pnl_px"].to_numpy()          # 1通貨あたりの円損益（コスト差引後）
    loss001 = np.median(risk) * UNITS001 * FX  # 0.01ロットの中央損失(円)
    if lots_override is not None:
        lots = lots_override
    else:
        lots = max(0.01, np.floor(ACCOUNT * MAXRISK / loss001) * 0.01)
    mult = lots / 0.01
    per_trade = pnl_unit.mean() * UNITS001 * FX * mult
    n_yr = len(t) / span
    yearly = per_trade * n_yr
    yr_jpy = pd.Series(pnl_unit * UNITS001 * FX * mult).groupby(t["y"].to_numpy()).sum()
    dd = float((np.maximum.accumulate(np.cumsum(pnl_unit * UNITS001 * FX * mult)) -
                np.cumsum(pnl_unit * UNITS001 * FX * mult)).max())
    dd_pct_of_acct = dd / ACCOUNT * 100
    risk_pct_per_trade = loss001 * mult / ACCOUNT * 100
    print(f"{name:<40} 採用ロット={lots:6.2f} 1本リスク=¥{loss001*mult:9,.0f}({risk_pct_per_trade:4.1f}%) "
          f"1本期待値=¥{per_trade:8,.0f} 年間=¥{yearly:10,.0f} 口座比={yearly/ACCOUNT*100:6.2f}% "
          f"最悪年=¥{yr_jpy.min():9,.0f} maxDD=¥{dd:8,.0f}({dd_pct_of_acct:5.2f}%)")
    return dict(lots=lots, dd_pct=dd_pct_of_acct, yearly=yearly, risk_pct=risk_pct_per_trade,
                worst_year=float(yr_jpy.min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    df, inv, C = load_frames()
    if cli.smoke:
        df = df.loc[:"2015-12-31"]
    span = span_years(df)

    CANDS = [
        ("候補1", dict(k=2.0, system="B", stopk=2.0, val=2.0, fwd=20)),
        ("候補2", dict(k=2.5, system="A", stopk=2.0, val=3.0, fwd=20)),
    ]

    for label, cfg in CANDS:
        print("\n" + "=" * 128)
        print(f"##### {label}: k={cfg['k']} {cfg['system']}系 TR{cfg['val']} fwd{cfg['fwd']} #####")
        print("=" * 128)
        t_gate = build(df, gate=True, **cfg)
        t_nogate = build(df, gate=False, **cfg)
        if t_gate is None or t_nogate is None:
            print("  約定不足"); continue

        print("\n-- (a) リスク3%固定ロット --")
        r_gate = jpy_row(f"{label}(ゲート:週足30MA上)", t_gate, span)
        r_nogate = jpy_row(f"{label}(ゲート無し)", t_nogate, span)

        print("\n-- (b) ゲート版のロットを『ゲート無し版と同じmaxDD%』まで引き上げ --")
        # 線形近似: DD%はロット倍率にほぼ比例するので、比率で目標ロットを逆算してから実測で確認
        target_dd = r_nogate["dd_pct"]
        lots_a = r_gate["lots"]
        scale = target_dd / max(r_gate["dd_pct"], 1e-9)
        lots_b = lots_a * scale
        r_matched = jpy_row(f"{label}(ゲート・DD一致={target_dd:.2f}%目標)", t_gate, span,
                             lots_override=lots_b)
        print(f"    (参考: 3%上限に対する超過率 = {r_matched['risk_pct']/3.0:.2f}倍"
              f"{' ※3%上限超過' if r_matched['risk_pct'] > 3.0 else ''})")

    print(f"\n実行コマンド: .venv/bin/python experiments/atr7f_jpy_conversion.py{' --smoke' if cli.smoke else ''}")


if __name__ == "__main__":
    main()
