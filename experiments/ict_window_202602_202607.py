"""ユーザー依頼: 凍結済みICT忠実版（狩り+MSS+FVG displacement+FVG-CE入口+外部流動性PDH出口）を
EURUSD 15m long-only で 2026-02-02〜2026-07-16 窓に限定して回し、TradingView Pine(OANDA feed)の
同窓の結果（6本・1勝5敗）と桁が合うかを検算する。

仕様（前回セッションで確定済み・全て既存の凍結スクリプトから再構成、新規実装なし）:
  - 銘柄/TF: EURUSD 15m, long-only
  - セッション(NY壁時計): Asia=前日19:00-02:00 / London=02:00-07:00 / KZ=07:00-10:00
  - 狩り: London安値L が Asia安値 or PDL を割る
  - MSS: L直前フラクタル高値を高値で上抜け
  - displacement: ブレイク脚内 bullish FVG（帯幅/ATR>=0.15）必須（AND条件）= fvg_min_atr=0.15
  - 入口: FVG-CE(50%中点)に買い指値、KZ内約定、来る前に走り抜けたら見送り
    = ict_fvg_anchor.fvg_anchor_fn("mid","long") を lim_fn として使用
  - 損切: 狩られた安値L - 0.10*ATR14（BUF=0.1, 既定値）
  - 利確: 前日高値PDH - 5pip（外部流動性ターゲット）= ict_extliq_target.make_ext_tgt_fn("pdh",5,"eurusd","long")
  - 最小レンジ: H-L >= 0.25*ATR（build()内に既にハードコード済み、変更不要）
  - コスト: cost_tiers("eurusd")["realistic"] = spread 0.3pip + commission 0.6pip = 0.9pip RT

これは ict_fvg_anchor.py の生存候補(EURUSD-long-FVG-CE(mid), min_atr=0.15) の入口に、
ict_extliq_pdh_judge.py が審判した出口(ext_PDH_fluff5)を組み合わせた「完成版」の窓評価。

重要: 先読み禁止のため、指標(ATR/フラクタル/PDH/FVG)は全期間のフル系列で計算してから
（canonical_setups を full dates で呼ぶ）、集計だけを窓でスライスする。窓の頭でPDH等が
NaNになる汚染を避けるため。

Run: .venv/bin/python experiments/ict_window_202602_202607.py 2>&1 | tee experiments/out_ict_window_202602_202607.txt
"""
import sys, io, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import BUF, F_CANON, RR_CANON, walk, stats
from ict_population import canonical_setups, load_prepped
from ict_extliq_target import EURUSD_LIM_FN, EURUSD_MA, make_ext_tgt_fn
from ict_dxy_smt import cost_tiers

WIN_START = pd.Timestamp("2026-02-02")
WIN_END = pd.Timestamp("2026-07-16")   # データは 2026-07-10 08:45 UTC までしか無い（下で明示）


def main():
    print("=" * 100)
    print(f"窓検算: EURUSD 15m long, FVG-CE(mid,min_atr=0.15) 入口 / stop=L-0.1ATR / target=PDH-5pip")
    print(f"窓 = [{WIN_START.date()}, {WIN_END.date()}]（NY暦日, rec['date']で判定）")
    print("=" * 100)

    with contextlib.redirect_stderr(io.StringIO()):
        df, tarr, dates, span = load_prepped("eurusd")
    print(f"データ範囲: broker_dt {df['broker_dt'].min()} 〜 {df['broker_dt'].max()}  (bars={len(df)}, span={span}年)")
    if df["broker_dt"].max() < WIN_END:
        print(f"** 注意: データの終端が {df['broker_dt'].max()} で、要求窓の終端 {WIN_END.date()} に届いていない。"
              f" 実際にカバーできる窓は [{WIN_START.date()}, {df['broker_dt'].max().date()}]。")

    # 指標はフル系列（全dates）で構築 -> 窓は集計時にのみ適用
    S = canonical_setups(df, tarr, dates, 0, use_fvg=True, fvg_min_atr=EURUSD_MA,
                         use_liq=True, liq_ns=(20, 40))
    n_pop_full = sum(1 for rec in S if rec["long"] is not None)
    n_pop_win = sum(1 for rec in S if rec["long"] is not None
                    and WIN_START <= pd.Timestamp(rec["date"]) <= WIN_END)
    print(f"母集団(狩り+MSS+FVG displacement 通過, KZ到達前に割れなし): 全期間 n_pop={n_pop_full} / 窓内 n_pop={n_pop_win}")

    sp, cost = cost_tiers("eurusd")["realistic"]
    print(f"コスト(realistic): spread={sp:.5f}({sp*1e4:.2f}pip) + commission={cost:.5f}({cost*1e4:.2f}pip) "
          f"= RT {(sp+cost)*1e4:.2f}pip 相当（spreadは約定条件にも使用、commissionはRのみ減算）")

    lim_fn = EURUSD_LIM_FN                              # FVG-CE mid
    tgt_fn = make_ext_tgt_fn("pdh", 5, "eurusd", "long")  # PDH - 5pip

    trade_log = []
    tr_full = walk(df, S, F_CANON, RR_CANON, BUF, sp, cost, "long",
                   lim_fn=lim_fn, tgt_fn=tgt_fn, trade_log=trade_log)
    # trade_log は約定した全トレード（全期間）。窓でフィルタ。
    win_trades = [t for t in trade_log if WIN_START <= pd.Timestamp(t["date"]) <= WIN_END]
    win_trades.sort(key=lambda t: t["fill_dt"])

    print(f"\n約定数: 全期間 n={len(trade_log)} / 窓内 n={len(win_trades)}")

    print("\n" + "-" * 100)
    print("1. 窓内トレード一覧（broker時刻[Vantage EET/EEST]・向き・入口・損切・利確・決済理由・R）")
    print("-" * 100)
    if not win_trades:
        print("  (窓内約定ゼロ)")
    else:
        print(f"  {'#':>3} {'entry_dt(broker)':19s} {'side':5s} {'entry':>9s} {'stop':>9s} {'tgt':>9s} "
              f"{'reason':7s} {'R(gross)':>9s} {'net(R)':>9s}")
        for i, t in enumerate(win_trades, 1):
            print(f"  {i:3d} {str(t['fill_dt']):19s} {t['side']:5s} {t['entry']:9.5f} {t['stop']:9.5f} "
                  f"{t['tgt']:9.5f} {t['reason']:7s} {t['R']:9.3f} {t['net']:9.3f}")

    print("\n" + "-" * 100)
    print("2. 窓内サマリ（参考値・本数が少ないためノイズ大）")
    print("-" * 100)
    n = len(win_trades)
    if n == 0:
        print("  n=0、統計不可")
    else:
        net = np.array([t["net"] for t in win_trades])
        g = np.array([t["R"] for t in win_trades])
        win_pct = 100 * (g > 0).mean()
        pos, neg = net[net > 0].sum(), -net[net < 0].sum()
        pf = pos / neg if neg > 0 else float("inf")
        cum = np.cumsum(net)
        dd = float((np.maximum.accumulate(cum) - cum).max()) if n > 0 else 0.0
        print(f"  n={n}  win%={win_pct:.1f}  PF={pf:.2f}  totR={net.sum():+.3f}  meanR={net.mean():+.3f}  "
              f"maxDD(R)={dd:.3f}")
        reasons = pd.Series([t["reason"] for t in win_trades]).value_counts().to_dict()
        print(f"  決済理由内訳: {reasons}")

    print("\n" + "-" * 100)
    print("3. 桁レンジ判定（Pine/OANDA同窓 = 6本・1勝5敗 が参照値）")
    print("-" * 100)
    if n == 0:
        print("  n=0（0〜3本のレンジ）: フィード差だけでは説明しづらい。母集団自体がこの窓で立っていない"
              "可能性が高い → 下の母集団内訳を見て、どのゲート（狩り/MSS/FVG/約定/PDH目標到達）で"
              "落ちているか調べる必要あり。")
    elif n <= 3:
        print(f"  n={n}（0〜3本のレンジ）: Pine根拠数6本の半分以下。フィード差にしては差が大きい可能性 "
              "→ 要調査（下の内訳参照）。")
    elif n > 10:
        print(f"  n={n}（10本超）: Pine根拠数6本の倍近い。フィルタ漏れ（狩り/MSS/FVG/PDH見送りのどれかが "
              "緩すぎる）の可能性 → 要調査。")
    else:
        print(f"  n={n}（5〜8本のレンジ）: 移植成功とみなせる範囲。フィード差(Vantage vs OANDA)を考えれば"
              "妥当なズレ。")

    # 内訳診断: 各段階でどれだけ落ちているか(窓内、狩り通過→MSS→FVG→約定→PDH目標妥当)を出す
    print("\n" + "-" * 100)
    print("4. 窓内の段階別内訳（診断用）")
    print("-" * 100)
    dates_win = [d for d in dates if WIN_START <= pd.Timestamp(d) <= WIN_END]
    print(f"  窓内の日数(NY暦日, 取引可能日ベース): {len(dates_win)}")
    # 狩りのみ / 狩り+MSS(FVG無し) / 狩り+MSS+FVG(displacement) の順に緩めて数える
    S_sweep_only = canonical_setups(df, tarr, dates, 0, use_fvg=False, fvg_min_atr=0.0,
                                     use_liq=False)
    with contextlib.redirect_stderr(io.StringIO()):
        from ict_population import build
    S_no_fvg = build(df, tarr, dates, True, True, "mss", 0, use_fvg=False)
    n_sweep_mss_win = sum(1 for rec in S_no_fvg if rec["long"] is not None
                          and WIN_START <= pd.Timestamp(rec["date"]) <= WIN_END)
    print(f"  狩り+MSS通過(FVG条件なし): 窓内 n={n_sweep_mss_win}")
    print(f"  狩り+MSS+FVG displacement(min_atr=0.15)通過: 窓内 n={n_pop_win}")
    skip_log = []
    tr_win_check = walk(df, S, F_CANON, RR_CANON, BUF, sp, cost, "long",
                        lim_fn=lim_fn, tgt_fn=tgt_fn, skip_log=skip_log)
    assert len(tr_win_check) == len(tr_full), "tie-back: 同じ引数で2回walkした結果本数が不一致"
    skip_win = [(d, r) for (d, r) in skip_log if WIN_START <= pd.Timestamp(d) <= WIN_END]
    filled_win = sum(1 for t in trade_log if WIN_START <= pd.Timestamp(t["date"]) <= WIN_END)
    no_fill_win = n_pop_win - filled_win - len(skip_win)   # KZ内に指値が来ず未約定（FVG通過はしたが不成立）
    print(f"  窓内 FVG displacement 通過 n_pop={n_pop_win} の内訳:"
          f" 約定(TP/SL/timeout)={filled_win} / tgt_fn見送り={len(skip_win)} / KZ内未約定(指値まで来ず)={no_fill_win}")
    if skip_win:
        print(f"  tgt_fn見送り内訳(窓内): {pd.Series([r for _,r in skip_win]).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
