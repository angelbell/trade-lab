"""follow-up 1: `ict_condition_ablation.py` は C0-C3 を【全史】と【2024-26】でしか出していなかった。
`ict_ground_panel.py` で EURUSD の C0(E3xS2) が 2018-20:0.81 / 2021-23:0.82 / 2024-26:0.79 と
ほぼ平らだと分かった --- 全史の C0=0.95 は 2000-2017 に引っ張られたプーリング汚染であり、
「2024-26 に地面が沈んだ」という物語そのものが崩れる。ここで条件4段 x 入口2 x 損切り2 = 16セルを
2018-20/2021-23/2024-26 の3年代（+全史を参考併記）に分けて出し、リフト(C3-C0)が年代でどう動くかを
確定させる。

流用（車輪の再発明禁止）: ict_condition_ablation.{calibrate, build_c0, make_off_lim_fn}、
ict_population.{build, canonical_setups, load_prepped}、ict_exec.{walk, F_CANON, RR_CANON}、
ict_extliq_target.{make_ext_tgt_fn, EURUSD_LIM_FN, EURUSD_MA}、ict_dxy_smt.cost_tiers、
ict_capture_decomp.{cell_stats, filter_era, S1_BUF, S2_BUF}。新規実装は era ループと lift 集計のみ。

Run: .venv/bin/python experiments/ict_ground_lift_by_era.py [--smoke] 2>&1 | tee experiments/out_ict_ground_lift_by_era.txt
"""
import sys, io, argparse, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np

from ict_exec import F_CANON, RR_CANON, walk
from ict_population import canonical_setups, load_prepped, build
from ict_extliq_target import make_ext_tgt_fn, EURUSD_LIM_FN, EURUSD_MA
from ict_dxy_smt import cost_tiers
from ict_capture_decomp import cell_stats, filter_era, S1_BUF, S2_BUF
from ict_condition_ablation import calibrate, build_c0, make_off_lim_fn

ERAS = [(2018, 2020, "2018-20"), (2021, 2023, "2021-23"), (2024, 2026, "2024-26"), (2000, 2026, "全史(参考)")]


def fmt(cs):
    if cs is None:
        return "n=0"
    return f"n={cs['n']:4d} win%={cs['win']:5.1f} PF={cs['pf']:5.2f} meanR={cs['meanR']:+.3f} totR={cs['totR']:+7.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    with contextlib.redirect_stderr(io.StringIO()):
        df, tarr, dates, span = load_prepped("eurusd")
    if args.smoke:
        dates = dates[-int(len(dates) * 0.3):]

    sp, cost = cost_tiers("eurusd")["realistic"]
    tgt_fn = make_ext_tgt_fn("pdh", 5, "eurusd", "long")

    S3 = canonical_setups(df, tarr, dates, 0, use_fvg=True, fvg_min_atr=EURUSD_MA, use_liq=True, liq_ns=(20, 40))
    atr_by_date = {rec["date"]: rec["long"]["atr"] for rec in S3 if rec["long"] is not None}
    calib = calibrate(df, S3, atr_by_date, sp, cost, tgt_fn)
    off_E1, off_E3 = calib["E1"]["off_med"], calib["E3"]["off_med"]
    depth_E1, depth_E3 = calib["E1"]["depth_med"], calib["E3"]["depth_med"]
    print(f"較正: off_med[E1]={off_E1:+.3f} depth_med[E1]={depth_E1:+.3f}  "
          f"off_med[E3]={off_E3:+.3f} depth_med[E3]={depth_E3:+.3f}\n")

    S1_pop = build(df, tarr, dates, use_sweep=True, use_mss=False, leg="lonend", shift=0,
                   use_fvg=False, use_liq=True, liq_ns=(20, 40))
    S2_pop = build(df, tarr, dates, use_sweep=True, use_mss=True, leg="mss", shift=0,
                   use_fvg=False, use_liq=True, liq_ns=(20, 40))
    C0_forE1 = build_c0(df, tarr, dates, off_E1, depth_E1, "E1")
    C0_forE3 = build_c0(df, tarr, dates, off_E3, depth_E3, "E3")

    conditions = {
        "C0_条件なし": {"E1": (C0_forE1, make_off_lim_fn(off_E1)), "E3": (C0_forE3, None)},
        "C1_狩りのみ": {"E1": (S1_pop, make_off_lim_fn(off_E1)), "E3": (S1_pop, None)},
        "C2_狩り+MSS": {"E1": (S2_pop, make_off_lim_fn(off_E1)), "E3": (S2_pop, None)},
        "C3_狩り+MSS+FVG(旗艦)": {"E1": (S3, EURUSD_LIM_FN), "E3": (S3, None)},
    }
    stop_bufs = {"S1": S1_BUF, "S2": S2_BUF}

    all_trades = {}
    for cname, emap in conditions.items():
        for ek in ("E1", "E3"):
            pop, lim_fn = emap[ek]
            for sk, buf in stop_bufs.items():
                trades = walk(df, pop, F_CANON, RR_CANON, buf, sp, cost, "long", lim_fn=lim_fn, tgt_fn=tgt_fn)
                all_trades[(cname, ek, sk)] = trades

    print("#" * 110)
    print("16セル x 年代別（EURUSD, C0/C1/C2/C3 x E1/E3 x S1/S2）")
    print("#" * 110)
    for ek in ("E1", "E3"):
        for sk in stop_bufs:
            print(f"\n  === {ek} x {sk} ===")
            for lo, hi, elabel in ERAS:
                row = []
                for cname in conditions:
                    cs = cell_stats(filter_era(all_trades[(cname, ek, sk)], lo, hi))
                    row.append(f"{cname}: {fmt(cs)}")
                print(f"    [{elabel:10s}]")
                for r in row:
                    print(f"        {r}")

    print("\n" + "#" * 110)
    print("リフト = C3(旗艦) − C0(条件なし)  年代別（PF差・win%差）")
    print("#" * 110)
    for ek in ("E1", "E3"):
        for sk in stop_bufs:
            print(f"\n  {ek} x {sk}:")
            for lo, hi, elabel in ERAS:
                c0 = cell_stats(filter_era(all_trades[("C0_条件なし", ek, sk)], lo, hi))
                c3 = cell_stats(filter_era(all_trades[("C3_狩り+MSS+FVG(旗艦)", ek, sk)], lo, hi))
                if c0 is None or c3 is None:
                    print(f"    {elabel:10s}: n不足")
                    continue
                print(f"    {elabel:10s}: C0[win{c0['win']:.1f}/PF{c0['pf']:.2f}] -> "
                      f"C3[win{c3['win']:.1f}/PF{c3['pf']:.2f}]   "
                      f"リフト: Δwin%={c3['win']-c0['win']:+.1f}pt  ΔPF={c3['pf']-c0['pf']:+.2f}")

    print("\n" + "#" * 110)
    print("判定: (i)リフトが縮んだ か (ii)地面が動いた か --- 上のリフト推移と、フォローアップ1冒頭の")
    print("C0年代別(ほぼ平ら)を突き合わせて判定すること。")
    print("#" * 110)


if __name__ == "__main__":
    main()
