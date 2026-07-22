"""ICT 忠実化・優先3 追走: 審判対象を「勝率最良(swingH20)」から「全軸でRR4を上回ったPDH」へ差し替え。

コーディネータからの訂正: 勝率だけを最大化する近接目標(swingH20)は既にPF/maxDD/保守コスト/直近時代で
崩れると判明した（ict_extliq_target.py の判定3）ため、審判すべきは PDH_fluff5/fluff3。

1. EURUSD-long-FVG-CE(mid,min0.15): ext_PDH_fluff5 / ext_PDH_fluff3 を RR4 と並べて審判
   （ブロックブートストラップ1/3/6/12・プラセボ窓+4/8/12h・時代別、realistic+conservative）。
2. 台地確認: PDH fluff∈{0,3,5} で上記の頑健性指標が単調/台地かlone spikeか。
3. 本命 usdjpy-long（入口=v4の0.25/min0.25、ict_extliq_target.CONTROL_MA・lim_fn=None）に
   ext_PDH_fluff5 を適用しRR4と比較（win%/PF/meanR/totR-DD/maxDD/skip% + ブロック1/3/6/12・時代別、2コスト段）。

すべて ict_extliq_target.py の judge_cell/run_cell/make_ext_tgt_fn をそのまま import して再利用
（車輪の再発明禁止）。PDHの先読み検査は事前に別途実施済み（前日の実高値と完全一致・shift(1)で確定済み）。

Run: .venv/bin/python experiments/ict_extliq_pdh_judge.py [--smoke] 2>&1 | tee experiments/out_ict_extliq_pdh_judge.txt
"""
import sys, io, argparse, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")

from ict_population import load_prepped
from ict_extliq_target import (EURUSD_LIM_FN, EURUSD_MA, CONTROL_MA,
                               make_ext_tgt_fn, judge_cell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print("#" * 110)
    print("1+2. EURUSD-long-FVG-CE(mid,min0.15): RR4 vs ext_PDH_fluff{0,3,5} 審判（台地確認込み）")
    print("#" * 110)
    with contextlib.redirect_stderr(io.StringIO()):
        df, tarr, dates, span = load_prepped("eurusd")
    if args.smoke:
        dates = dates[-int(len(dates) * 0.25):]

    for tier in ("realistic", "conservative"):
        judge_cell(df, tarr, dates, "eurusd", "long", span, EURUSD_MA, EURUSD_LIM_FN, tier, "RR4", rr=4.0)
        for fl in (0, 3, 5):
            tgt_fn = make_ext_tgt_fn("pdh", fl, "eurusd", "long")
            judge_cell(df, tarr, dates, "eurusd", "long", span, EURUSD_MA, EURUSD_LIM_FN, tier,
                      f"ext_PDH_fluff{fl}", tgt_fn=tgt_fn)

    print("\n" + "#" * 110)
    print("3. usdjpy-long（v4の0.25入口/min0.25）: RR4 vs ext_PDH_fluff5")
    print("#" * 110)
    with contextlib.redirect_stderr(io.StringIO()):
        dfj, tarrj, datesj, spanj = load_prepped("usdjpy")
    if args.smoke:
        datesj = datesj[-int(len(datesj) * 0.25):]

    for tier in ("realistic", "conservative"):
        judge_cell(dfj, tarrj, datesj, "usdjpy", "long", spanj, CONTROL_MA, None, tier, "RR4", rr=4.0)
        tgt_fn = make_ext_tgt_fn("pdh", 5, "usdjpy", "long")
        judge_cell(dfj, tarrj, datesj, "usdjpy", "long", spanj, CONTROL_MA, None, tier,
                  "ext_PDH_fluff5", tgt_fn=tgt_fn)


if __name__ == "__main__":
    main()
