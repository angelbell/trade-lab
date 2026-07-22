"""follow-up 2: 非打ち切りMAE を6ペア x 3年代でパネル化 --- H2(ダマシ/深い刺し)の正しい検定。

`ict_ground_panel.py` の MAE は損切りで 1.0 に打ち切る定義（scan_mfe_mae）だったため、損切り率が
85-100%と高いこの母集団では中央値が全セル1.00に張り付き、素材診断(`ict_material_decay.py`)が
見つけた「MAE -2.7R -> -3.5Rへの深化」を原理的に検出できなかった。ここでは
`ict_material_decay.mfe_mae_scan`（基準点=KZ開始バー始値・リスク単位=基準点-(L-0.1ATR)・
前進500本・損切りで止めない）をそのまま流用し、6ペア x 3年代（2018-20/2021-23/2024-26）で
MAE中央値・平均・分位を集計する。母集団は canonical_setups(shift=0)（狩り+MSS、FVG無し=
素材診断と同じ「原材料」レベル）。

流用（車輪の再発明禁止）: ict_material_decay.{mfe_mae_scan, long_diag, selfcheck_tieback}、
ict_population.{canonical_setups, load_prepped}、ict_exec.CUT2000。

Run: .venv/bin/python experiments/ict_mae_panel.py [--smoke] 2>&1 | tee experiments/out_ict_mae_panel.txt
"""
import sys, argparse
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_population import canonical_setups, load_prepped
from ict_material_decay import mfe_mae_scan, long_diag, selfcheck_tieback

FX6 = ["eurusd", "gbpusd", "usdjpy", "audusd", "nzdusd", "usdcad"]
ERAS = [(2018, 2020, "2018-20"), (2021, 2023, "2021-23"), (2024, 2026, "2024-26")]


def summarize(mae, mfe):
    n = len(mae)
    if n < 5:
        return None
    return dict(n=n, mae_med=np.median(mae), mae_mean=mae.mean(), mae_sd=mae.std(ddof=1) if n > 1 else 0.0,
                mae_q25=np.percentile(mae, 25), mae_q75=np.percentile(mae, 75),
                mfe_med=np.median(mfe), mfe_mean=mfe.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    panel = {}
    for name in FX6:
        df, tarr, dates, span = load_prepped(name)
        if args.smoke:
            dates = dates[-int(len(dates) * 0.3):]
        ok = selfcheck_tieback(df, tarr, dates, name)
        S0 = canonical_setups(df, tarr, dates, shift=0)
        diag = long_diag(df, tarr, dates)
        scan = mfe_mae_scan(df, S0, diag)
        print(f"\n  --- {name} (tie-back {'PASS' if ok else 'FAIL'}, n_scan={len(scan)}) ---")
        for a, b, elabel in ERAS:
            sub = [r for r in scan if a <= r["year"] <= b]
            mae = np.array([r["mae"] for r in sub])
            mfe = np.array([r["mfe"] for r in sub])
            s = summarize(mae, mfe)
            panel[(name, a, b)] = s
            if s is None:
                print(f"    {elabel:10s}: n<5 skip")
                continue
            print(f"    {elabel:10s}: n={s['n']:4d}  MAE中央={s['mae_med']:+.2f} 平均={s['mae_mean']:+.2f} "
                  f"sd={s['mae_sd']:.2f} [q25={s['mae_q25']:+.2f},q75={s['mae_q75']:+.2f}]  "
                  f"MFE中央={s['mfe_med']:.2f} 平均={s['mfe_mean']:.2f}")

    print("\n" + "#" * 110)
    print("EURUSD固有か市場全体か: MAE中央値/平均 の年代推移(6ペア並記)")
    print("#" * 110)
    for key in ("mae_med", "mae_mean"):
        print(f"\n  {key}:")
        for name in FX6:
            vals = []
            for a, b, elabel in ERAS:
                s = panel.get((name, a, b))
                vals.append(f"{elabel}:{s[key]:+.2f}" if s else f"{elabel}:n/a")
            print(f"    {name:<8}{'  '.join(vals)}")

    print("\n" + "#" * 110)
    print("判定: EURUSDだけMAEが深化(より負に)しているか、6ペア共通で深化しているか、どちらも無いか")
    print("#" * 110)


if __name__ == "__main__":
    main()
