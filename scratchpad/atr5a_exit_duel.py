"""(a) 出口の公平な再対決 ― 損切り2種(A/B) x 出口(固定RR vs ATRトレール) x 保有上限(fwd) の格子。

前回の比較は不公平だった: A系(拡大足の反対端=2ATR超)の損切りに対し RR3 の目標は6ATR先にあり、
保有上限20本では届かない＝「トレール vs 時間切れ」の比較になっていた。B系（stop=e-2.0*ATR[s-1]）で
目標が届く形にしたときに固定RRが勝つか、が事前登録の予言（構造法則4/9: 遠い固定目標が最適）。

固定RR出口: 目標RRを {2,3,4.5,6} で振る（trail_atr=0）。
ATRトレール出口: 目標(tgt)は「従来のRR」（方向設定ごとの凍結アンカーRR = k2.0→3.0, k1.5→4.5）に
  固定したまま併存させ、trail_atr（ATR倍率）だけを {2,3,4,6} で振る（カードの指示どおり
  「目標は従来どおり併存」＝先に触れた方で決着、trail_atrはwalk()の実装＝カード(c)で検証済み）。

母集団の警告: トレールは早期退出で max_pos=1 のスロットが空き、次シグナルを拾えるぶん固定RRと
Nが変わりうる（早く降りると玉が回るため）。全セルにNを併記し、N差が大きいセルは別母集団の
比較として扱う。

実行: 損益は入口価格に対する% ／ walk()はcost=0で呼び外側でコストを引く ／ ロング成行・ショート指値。

SCREEN = "atr_spike_btc_h1"
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atr5_common import (load_frames, atr_prev_of, raw_triggers, build_entries, run_cell,
                          stats, span_years, fmt_row, DIRECTIONS)  # noqa: E402

RR_LADDER = [2.0, 3.0, 4.5, 6.0]
TRAIL_LADDER = [2.0, 3.0, 4.0, 6.0]
FWDS = [20, 60, 200]
SYSTEMS = ["A", "B"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    df, inv, C = load_frames()
    if cli.smoke:
        df = df.loc[:"2021-12-31"]
        inv = inv.loc[:"2021-12-31"]
    years = span_years(df)

    rows = []
    for dcfg in DIRECTIONS:
        k, side, anchor_rr, pf = dcfg["k"], dcfg["side"], dcfg["rr"], dcfg["pf"]
        d = df if side == "long" else inv
        Cx = None if side == "long" else C
        atr_prev = atr_prev_of(d)
        s_idx = raw_triggers(d, atr_prev, k)

        print("\n" + "=" * 112)
        print(f"##### {dcfg['name']}  (side={side} k={k} アンカーRR={anchor_rr} pf={pf}) #####")
        print("=" * 112)

        for system in SYSTEMS:
            # RRラダーぶんentriesを事前構築（アンカーRRはこの中に含まれる＝トレール実験で再利用）
            ent_by_rr = {rr: build_entries(d, atr_prev, s_idx, system, rr) for rr in RR_LADDER}
            print(f"\n--- 損切り系={system} "
                  f"({'拡大足の反対端' if system == 'A' else 'e-2.0*ATR[s-1]（目標が届く形）'}) ---")
            for fwd in FWDS:
                print(f"  [fwd={fwd}]")
                exits = [("固定RR", rr, ent_by_rr[rr], 0.0) for rr in RR_LADDER]
                exits += [("ATRトレール", m, ent_by_rr[anchor_rr], m) for m in TRAIL_LADDER]
                for kind, val, ent, trail_atr in exits:
                    t = run_cell(d, ent, pf=pf, fill_win=200, fwd=fwd, trail_atr=trail_atr, C=Cx)
                    label = f"{kind}={val}"
                    if t is None or len(t) == 0:
                        print(f"    {label:<16} (約定0件)")
                        continue
                    s = stats(t, years)
                    print("    " + fmt_row(f"{label:<16}", s))
                    rows.append(dict(direction=dcfg["name"], side=side, k=k, system=system,
                                      fwd=fwd, exit_kind=kind, exit_val=val, **s))

    grid = pd.DataFrame(rows)
    out_csv = os.path.join(os.path.dirname(__file__),
                            "out_atr5a_exit_duel_smoke.csv" if cli.smoke else "out_atr5a_exit_duel.csv")
    grid.to_csv(out_csv, index=False)
    print(f"\n[csv] {out_csv} ({len(grid)} 行)")

    # ================================================================
    # 結論行: 各(direction, system, fwd)内で固定RR最良 vs トレール最良 を比べる
    # ================================================================
    print("\n" + "=" * 112)
    print("結論表: (方向設定, 損切り系, fwd) ごとに 固定RR最良セル vs ATRトレール最良セル (総%で比較)")
    print("=" * 112)
    header = (f"  {'方向':<24}{'系':>3}{'fwd':>6} | {'固定RR最良':>10}{'総%':>9}{'N':>5}"
              f" | {'トレール最良':>10}{'総%':>9}{'N':>5} | 勝者")
    print(header)
    win_summary = []
    for (direction, system, fwd), g in grid.groupby(["direction", "system", "fwd"]):
        rr_g = g[g.exit_kind == "固定RR"]
        tr_g = g[g.exit_kind == "ATRトレール"]
        if len(rr_g) == 0 or len(tr_g) == 0:
            continue
        best_rr = rr_g.loc[rr_g.tot_pct.idxmax()]
        best_tr = tr_g.loc[tr_g.tot_pct.idxmax()]
        winner = "固定RR" if best_rr.tot_pct >= best_tr.tot_pct else "トレール"
        win_summary.append(dict(direction=direction, system=system, fwd=fwd, winner=winner,
                                 rr_val=best_rr.exit_val, rr_tot=best_rr.tot_pct, rr_n=best_rr.N,
                                 tr_val=best_tr.exit_val, tr_tot=best_tr.tot_pct, tr_n=best_tr.N))
        print(f"  {direction:<24}{system:>3}{fwd:>6} | "
              f"RR{best_rr.exit_val:>4.1f}{best_rr.tot_pct:>+9.1f}{int(best_rr.N):>5} | "
              f"x{best_tr.exit_val:>4.1f}{best_tr.tot_pct:>+9.1f}{int(best_tr.N):>5} | {winner}")

    ws = pd.DataFrame(win_summary)
    if len(ws):
        b_wins = ws[ws.system == "B"]
        n_rr_win_b = (b_wins.winner == "固定RR").sum()
        n_tr_win_b = (b_wins.winner == "トレール").sum()
        a_wins = ws[ws.system == "A"]
        n_rr_win_a = (a_wins.winner == "固定RR").sum()
        n_tr_win_a = (a_wins.winner == "トレール").sum()
        print(f"\n[結論(a)] B系(目標が届く形)で固定RRが勝ったセル数: {n_rr_win_b}/{len(b_wins)}"
              f"  トレールが勝ったセル数: {n_tr_win_b}/{len(b_wins)}")
        print(f"          A系(従来の反対端)で固定RRが勝ったセル数: {n_rr_win_a}/{len(a_wins)}"
              f"  トレールが勝ったセル数: {n_tr_win_a}/{len(a_wins)}")
        if n_rr_win_b > n_tr_win_b:
            print("          → B系では固定RRが優勢＝事前登録の予言(構造法則4/9: 遠い固定目標が最適)と整合")
        elif n_rr_win_b < n_tr_win_b:
            print("          → B系でもトレールが優勢＝事前登録の予言(構造法則4/9)と矛盾する結果")
        else:
            print("          → B系は拮抗＝どちらとも言えない")

    print(f"\n実行コマンド: .venv/bin/python scratchpad/atr5a_exit_duel.py{' --smoke' if cli.smoke else ''}")


if __name__ == "__main__":
    main()
