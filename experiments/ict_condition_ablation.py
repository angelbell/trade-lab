"""ICT 改善案1（最優先）: 執行エッジのプラセボ = 条件アブレーション。

問い: 旗艦(EURUSD 15m long, 狩り+MSS+FVG／入口FVG-CE／損切りL-0.1ATR／目標PDH-5pip)のエッジが
「執行（安く買う）」なら、狩り+MSS+FVG という"条件"は何を足しているのか。C3(旗艦)しか無かった
6セル分解（ict_capture_decomp.py）に、「条件なし×指値」という対照(C0)を追加し、
C0/C1(狩りのみ)/C2(狩り+MSS)/C3(狩り+MSS+FVG) × E1(CE指値)/E3(0.25浅指値) × S1(-0.1ATR)/S2(-0.5ATR)
= 16セルで、条件が勝率・PFを押し上げているかを測る。

流用（車輪の再発明禁止）:
  - ict_population.build/canonical_setups  … C1(use_mss=False)/C2(use_mss=True,use_fvg=False)/
    C3(canonical_setups, 現行旗艦)をそのまま生成
  - ict_exec.walk/window_pos/KZ_HOURS       … 約定・前進走査（同足損切り優先など）はそのまま
  - ict_fvg_anchor.fvg_anchor_fn            … C3 の E1(FVG-CE) 入口
  - ict_extliq_target.make_ext_tgt_fn       … 目標 PDH-5pip
  - ict_dxy_smt.cost_tiers                  … realistic コスト
  - ict_capture_decomp.cell_stats/filter_era/run_cell … 統計・era切り出し・6セル分解の実行部を再利用
    （C3×{E1,E3}×{S1,S2} の実測値と、E1/E3 それぞれの「Hからの入口オフセット中央値(ATR単位)」
    「Lからの入口深さ中央値(ATR単位)」の較正はここから取る）

C0（条件なし）の作り方 --- 厳密一致が不可能なので採った近似を明記:
  各日、NYキルゾーン窓(k0,k1)を条件を一切課さず機械的に計算する（sweep/MSS/FVGを問わない）。
  「H」の代役 = キルゾーン開始足の始値 o[k0]（前のロンドン安値からの水準ではなく、その日その時点で
  唯一自然に取れる基準価格）。「ATR」は k0-1 で確定済みのATR14。
  入口オフセット(ATR単位, off_med[E])・Lからの深さ(ATR単位, depth_med[E]) は、
  実際の C3×E1 / C3×E3 の約定トレード群から (H_real-entry_real)/atr_real ・ (entry_real-L_real)/atr_real
  の中央値として較正する（stop(S1/S2)には依存しない量 --- entry/Lはstop選択と無関係に決まるため）。
  合成 H_C0 = o[k0]、合成 L_C0(E) = H_C0 - depth_med[E]*atr_ref とすることで、
  E3(0.25浅指値, lim_fn=None) は H-0.25*(H-L) の標準式がそのまま off_med[E3]*atr_ref を再現する
  （off_med[E3] は実測で 0.25*median[(H-L)/atr] と自動的に一致するため近似ではなく厳密な代数関係）。
  E1(FVG-CE) は C0 に FVG 帯が存在しないため、lim_fn = H_C0 - off_med[E1]*atr_ref という
  固定ATRオフセット指値で代用する（これが今回唯一の実質的な近似）。
  損切りは通常どおり L_C0(E) - buf*atr_ref（buf=S1/S2）--- Lを合成しているので、この式は
  「Lからbufぶん」という現行の掟をそのまま保つ。

C1(狩りのみ)/C2(狩り+MSS): build() が実物の H,L,atr,kz(,pdh) を返すので近似は不要 --- ただし
  FVG帯は存在しない(use_fvg=False)ため、E1列だけは C0 と同じ固定ATRオフセット代用
  (lim_fn = H - off_med[E1]*atr) を使う。E3列は標準の 0.25*(H-L) 式をそのまま使う(近似なし)。

目標=PDH-5pip固定・コスト=realistic固定・EURUSD long専用（変更禁止、旗艦分解と同じ）。
先読み: build()のsweep/MSS/FVG判定・PDHのshift(1)は既存のまま不変。C0のH_C0=o[k0]・ATR=atr[k0-1]は
いずれもキルゾーン開始足以前に確定済み。

自己検査: E1×S1×C3 が台帳 n=313/win34.5/PF1.41/totR-DD4.05/maxDD21.7 を再現するかを最初に確認。

Run: .venv/bin/python experiments/ict_condition_ablation.py [--smoke] 2>&1 | tee experiments/out_ict_condition_ablation.txt
"""
import sys, io, argparse, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import BUF, F_CANON, RR_CANON, walk, stats, window_pos, KZ_HOURS
from ict_population import canonical_setups, load_prepped, build, prev_day_extremes
from ict_fvg_anchor import fvg_anchor_fn
from ict_extliq_target import make_ext_tgt_fn, EURUSD_LIM_FN, EURUSD_MA
from ict_dxy_smt import cost_tiers
from ict_capture_decomp import cell_stats, filter_era, run_cell, S1_BUF, S2_BUF

ERAS = [(2000, 2026, "全史"), (2024, 2026, "2024-26")]


# ============================================================== 較正: off_med[E] / depth_med[E]
def calibrate(df, S_pop, atr_by_date, sp, cost, tgt_fn):
    """実際の C3×E1 / C3×E3 の約定群から、H起点オフセットとL起点深さの中央値(ATR単位)を較正する。
    stop(S1/S2)に依存しない量なので S1 セルの trade_log だけで十分（entry/Lはstopと無関係）。"""
    H_by_date = {rec["date"]: rec["long"]["H"] for rec in S_pop if rec["long"] is not None}
    L_by_date = {rec["date"]: rec["long"]["L"] for rec in S_pop if rec["long"] is not None}
    out = {}
    for ek, lim_fn in (("E1", EURUSD_LIM_FN), ("E3", None)):
        _, tlog = run_cell(df, S_pop, S_pop, atr_by_date, ek, "S1", sp, cost, tgt_fn)
        offs, depths = [], []
        for r in tlog:
            d = r["date"]
            A = atr_by_date.get(d)
            H = H_by_date.get(d)
            L = L_by_date.get(d)
            if A and A > 0 and H is not None and L is not None:
                offs.append((H - r["entry"]) / A)
                depths.append((r["entry"] - L) / A)
        out[ek] = dict(off_med=float(np.median(offs)), depth_med=float(np.median(depths)), n=len(offs))
    return out


# ============================================================== C0: 条件なしキルゾーン母集団
def build_c0(df, tarr, dates, off_med, depth_med, entry_key):
    """各日、キルゾーン窓を条件無しで機械的に作る。基準価格 ref=o[k0]・atr_ref=atr[k0-1]
    （確定済み・先読み無し）。pdhも既存の prev_day_extremes を流用。

    【2026-07-17 バグ修正】旧実装は L_C0 = ref - depth_med*A としていたが、depth_med は
    (entry-L)/A（入口から狩り安値までの深さ）であり、(H-L)/A（レンジ全体）ではない。
    その結果 E1 では入口が合成Lより下（S1で全弾スキップ=n0、S2で risk が実物の1/10 の人工物）、
    E3 では損切りが実物比~25%タイトになっていた。
    正: L_C0 = ref - (off_med + depth_med)*A（入口が ref から off_med 下・L から depth_med 上に来る）。
    さらに E3 は walk の f式 entry = (1-f)*H + f*L が entry = ref - off_med*A を返すよう H を逆算
    （H = ref + (f/(1-f)*depth_med - off_med)*A）。E1 は lim_fn = H - off_med*A なので H = ref のまま。
    これで両列とも、実測 C3 トレード群の入口オフセット・深さの中央値を厳密に再現する。"""
    o = df["open"].values
    atr = df["atr14"].values
    dates_arr = df["_t"].dt.normalize().values
    pdh, _ = prev_day_extremes(df, dates)
    out = []
    K0H, K1H = KZ_HOURS
    for d in dates:
        day = pd.Timestamp(d)
        k0, k1 = window_pos(tarr, day + pd.Timedelta(hours=K0H), day + pd.Timedelta(hours=K1H))
        rec = {"date": d, "long": None, "short": None}
        if (k1 - k0) < 2 or not np.isfinite(atr[k0 - 1]) or atr[k0 - 1] <= 0:
            out.append(rec); continue
        A = atr[k0 - 1]
        ref = o[k0]
        L_C0 = ref - (off_med + depth_med) * A
        if entry_key == "E3":
            # f式 entry=(1-f)H+fL が entry=ref-off_med*A になるよう H を逆算
            H_C0 = ref + (F_CANON / (1.0 - F_CANON) * depth_med - off_med) * A
        else:
            # E1 は lim_fn=H-off_med*A なので H=ref で entry=ref-off_med*A
            H_C0 = ref
        rec["long"] = dict(L=L_C0, H=H_C0, atr=A, kz=(k0, k1), pdh=pdh.get(d, np.nan))
        out.append(rec)
    return out


def make_off_lim_fn(off_med):
    def fn(s):
        return s["H"] - off_med * s["atr"]
    return fn


def n_pop(S):
    return sum(1 for rec in S if rec["long"] is not None)


def yr_span(dates):
    return max((pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25, 0.1)


def fmt_row(label, cs, span):
    if cs is None:
        return f"    {label:26s} n=0 (該当なし)"
    npy = cs['n'] / span
    rdd = cs['totR'] / cs['maxDD'] if cs['maxDD'] > 0 else float("inf")
    return (f"    {label:26s} n={cs['n']:4d} 本/年={npy:5.1f} win%={cs['win']:5.1f} PF={cs['pf']:5.2f} "
            f"meanR={cs['meanR']:+.3f} totR={cs['totR']:+7.1f} maxDD={cs['maxDD']:6.2f} totR/DD={rdd:6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    with contextlib.redirect_stderr(io.StringIO()):
        df, tarr, dates, span = load_prepped("eurusd")
    if args.smoke:
        dates = dates[-int(len(dates) * 0.25):]

    sp, cost = cost_tiers("eurusd")["realistic"]
    tgt_fn = make_ext_tgt_fn("pdh", 5, "eurusd", "long")

    print("#" * 110)
    print("0. tie-back: C3×E1×S1 が台帳 n=313/win34.5/PF1.41/totR-DD4.05/maxDD21.7 を再現するか")
    print("#" * 110)
    S3 = canonical_setups(df, tarr, dates, 0, use_fvg=True, fvg_min_atr=EURUSD_MA, use_liq=True, liq_ns=(20, 40))
    atr_by_date = {rec["date"]: rec["long"]["atr"] for rec in S3 if rec["long"] is not None}
    tr, _ = run_cell(df, S3, S3, atr_by_date, "E1", "S1", sp, cost, tgt_fn)
    st = stats(tr, span)
    if st:
        print(f"  再現値: n={st['n']} win%={st['win']:.1f} PF={st['pf']:.2f} totR/DD={st['rdd']:.2f} maxDD={st['dd']:.1f}")
        if not args.smoke:
            ok = (st['n'] == 313 and abs(st['win'] - 34.5) < 0.2 and abs(st['pf'] - 1.41) < 0.02)
            print(f"  {'PASS' if ok else 'FAIL --- 要確認'}")
    else:
        print("  n<10 (smokeでは想定内)")

    print("\n較正: off_med[E]・depth_med[E]（実測C3×E1/E3の中央値, ATR単位）")
    calib = calibrate(df, S3, atr_by_date, sp, cost, tgt_fn)
    for ek, v in calib.items():
        print(f"  {ek}: off_med={v['off_med']:+.3f}  depth_med={v['depth_med']:+.3f}  (n={v['n']})")

    # ---------------- 母集団4段 ----------------
    S1_pop = build(df, tarr, dates, use_sweep=True, use_mss=False, leg="lonend", shift=0,
                   use_fvg=False, use_liq=True, liq_ns=(20, 40))   # C1 狩りのみ
    S2_pop = build(df, tarr, dates, use_sweep=True, use_mss=True, leg="mss", shift=0,
                   use_fvg=False, use_liq=True, liq_ns=(20, 40))   # C2 狩り+MSS
    off_E1, depth_E1 = calib["E1"]["off_med"], calib["E1"]["depth_med"]
    off_E3, depth_E3 = calib["E3"]["off_med"], calib["E3"]["depth_med"]
    C0_forE1 = build_c0(df, tarr, dates, off_E1, depth_E1, "E1")
    C0_forE3 = build_c0(df, tarr, dates, off_E3, depth_E3, "E3")

    print(f"\n母集団サイズ: C0(全KZ日)={n_pop(C0_forE1)}  C1(狩りのみ)={n_pop(S1_pop)}  "
          f"C2(狩り+MSS)={n_pop(S2_pop)}  C3(狩り+MSS+FVG,旗艦)={n_pop(S3)}")

    conditions = {
        "C0_条件なし": {"E1": (C0_forE1, make_off_lim_fn(off_E1)), "E3": (C0_forE3, None)},
        "C1_狩りのみ": {"E1": (S1_pop, make_off_lim_fn(off_E1)), "E3": (S1_pop, None)},
        "C2_狩り+MSS": {"E1": (S2_pop, make_off_lim_fn(off_E1)), "E3": (S2_pop, None)},
        "C3_狩り+MSS+FVG(旗艦)": {"E1": (S3, EURUSD_LIM_FN), "E3": (S3, None)},
    }
    stop_bufs = {"S1": S1_BUF, "S2": S2_BUF}

    print("\n" + "#" * 110)
    print("1. 16セル（条件4 × 入口2 × 損切り2）: n/本年/win%/PF/meanR/totR/maxDD/totR-DD（全史・2024-26）")
    print("   E1列: C0/C1/C2 は FVG帯が無いため H-off_med[E1]*ATR の固定オフセット指値で代用（近似）。")
    print("   E3列: 全条件で標準の H-0.25*(H-L) 式（C0のみ H,L 自体が合成、近似はそこに限定）。")
    print("#" * 110)

    span_use = span if not args.smoke else yr_span(dates)
    results = {}
    for cname, emap in conditions.items():
        print(f"\n  === {cname} ===")
        for ek in ("E1", "E3"):
            pop, lim_fn = emap[ek]
            atr_map = {rec["date"]: rec["long"]["atr"] for rec in pop if rec["long"] is not None}
            for sk, buf in stop_bufs.items():
                trades = walk(df, pop, F_CANON, RR_CANON, buf, sp, cost, "long", lim_fn=lim_fn, tgt_fn=tgt_fn)
                label = f"{ek}x{sk}"
                for lo, hi, elabel in ERAS:
                    sub = filter_era(trades, lo, hi)
                    cs = cell_stats(sub)
                    print(f"  [{elabel:8s}] {fmt_row(label, cs, span_use if elabel == '全史' else 3.0)}")
                results[(cname, ek, sk)] = trades

    print("\n" + "#" * 110)
    print("2. 判定に必要な一言: C3(旗艦) は C0(条件なし) を win%・PF で明確に上回るか（同一 E×S で対比較）")
    print("#" * 110)
    for ek in ("E1", "E3"):
        for sk in stop_bufs:
            c0 = cell_stats(results[("C0_条件なし", ek, sk)])
            c1 = cell_stats(results[("C1_狩りのみ", ek, sk)])
            c2 = cell_stats(results[("C2_狩り+MSS", ek, sk)])
            c3 = cell_stats(results[("C3_狩り+MSS+FVG(旗艦)", ek, sk)])
            def s(cs):
                return f"win{cs['win']:.1f}/PF{cs['pf']:.2f}/n{cs['n']}" if cs else "n=0"
            print(f"  {ek}x{sk}:  C0={s(c0)}   C1={s(c1)}   C2={s(c2)}   C3={s(c3)}")


if __name__ == "__main__":
    main()
