"""ICT 旗艦(EURUSD 15m long) 2024-26 出血の解剖 — 「獲り方」でどこまで説明できるか。

背景: `ict_material_decay.py` の素材診断で「巡行幅(MFE中央値2.5-3.0R)は健在だが目標到達への変換率が
半減(~0.61->~0.33)・MAEが深化(-3.5R)」と出た。ここでは同じ母集団の上で入口3種×損切り2種=6セルに
分解し、出血が「入口の深さ(逆選択)」「損切りの狭さ(先に刺される)」のどちらで説明できるかを測る。
診断であって採用実験ではない — 掟(±1sweep・block boot・placebo等)の検定はまだ課さない。数字だけ返す。

母集団（変更禁止・固定）: EURUSD 15m long-only、狩り+MSS+FVG(fvg_min_atr=0.15)、
  use_liq=True で rec["long"]["pdh"] を保持（優先3の PDH ターゲット用）。
  ict_population.canonical_setups をそのまま呼ぶ（車輪の再発明禁止）。n_pop=600（自己検査で確認済み）。
目標（変更禁止・全セル共通）: PDH-5pip。ict_extliq_target.make_ext_tgt_fn("pdh", 5, "eurusd", "long")
  をそのまま使う（skip条件=objective_at_or_below_entry / too_close(<0.5R) / no_objective は現行のまま）。
コスト: cost_tiers("eurusd")["realistic"]（ict_dxy_smt.cost_tiers を流用）。

入口3種（ict_exec.walk の lim_fn 引数で切り替え。全て既存の関数・仕組みをそのまま使う）:
  E1 = FVG-CE(50%) 指値  … ict_fvg_anchor.fvg_anchor_fn("mid","long")（現行旗艦の入口。母集団は
       use_fvg=True で fvg_lo/fvg_hi を既に保持しているので追加実装なし）
  E2 = KZ開始バーの次足始値で成行（約定選別ゼロ）… 新規: market_lim_fn(s) が常に約定するほど大きい
       指値を返すことで walk() の指値サーチを「即約定」に退化させ、setups の kz を (k0+1, k1) に
       1本ずらす（shift_kz_next_bar）ことで fp が必ず k0+1 になる（=KZ開始バーの"次足"始値、ASK基準）。
       walk() 本体のロジック(SL優先・FWD_CAP・タイブレーク)は一切変更しない — 入口探索の起点をずらす
       だけで「成行」を表現する。全setupsが対象（狩り+MSS+FVGの条件を満たした母集団はそのまま）。
  E3 = 0.25 リトレース指値 … lim_fn=None で ict_exec.walk のデフォルト(f*(H-L)固定)をそのまま使う。
       F_CANON=0.25 が既にこれと同じ値（旧v2旗艦の浅い押し目定義）。

損切り2種（buf 引数、ict_exec.BUF=0.1 が S1）:
  S1 = L - 0.1*ATR14（現行）　S2 = L - 0.5*ATR14（広い版）

逆選択の直接測定（項目2）: E1/E3 それぞれについて「指値が母集団のどのdateで実際に約定したか」を
  ダミーRR(rr=4.0,tgt_fn=None)の walk() で検出する（tgt_fn有無に関わらずfp探索ロジックは同一なので、
  約定判定だけを固定RRの軽い呼び出しで抽出=既存関数の副作用のない流用）。その約定/非約定でグループ分けし、
  「そのまま KZ 成行(E2)で持っていたら」の R（=E2×S1セルの trades、tgt_fn=PDH-5pip・stop=S1固定）を
  比較する。E2の対象母集団は狩り+MSS+FVG条件を満たした全setups（=n_pop=600の全員）。

自己検査: E1×S1(=優先3の旗艦 ext_PDH_fluff5) が台帳 n=313/win34.5/PF1.41/totR-DD4.05/maxDD21.7 を
再現することを最初に確認してから本測定に進む。

Run: .venv/bin/python scratchpad/ict_capture_decomp.py [--smoke] 2>&1 | tee scratchpad/out_ict_capture_decomp.txt
"""
import sys, io, argparse, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import MODEL, BUF, F_CANON, RR_CANON, walk, stats
from ict_population import canonical_setups, load_prepped
from ict_fvg_anchor import fvg_anchor_fn
from ict_extliq_target import make_ext_tgt_fn, EURUSD_LIM_FN, EURUSD_MA
from ict_dxy_smt import cost_tiers

ERAS = [(2000, 2026, "全史"), (2018, 2023, "2018-23"), (2024, 2024, "2024"),
        (2025, 2025, "2025"), (2026, 2026, "2026")]
S1_BUF, S2_BUF = 0.1, 0.5     # BUF(ict_exec)=0.1 が S1 と一致


def market_lim_fn(s):
    """E2: walk()の指値サーチを「常に即約定」に退化させるためのダミー指値（常にH+十分大の定数）。
    実際の約定価格は entry=min(lim, o[fp]+spread) で lim が巨大なら o[fp]+spread に潰れる
    ＝実質「成行(ASK基準)」になる。fp は shift_kz_next_bar で (k0+1) に固定される。"""
    return s["H"] + 1e6


def shift_kz_next_bar(setups):
    """rec["long"]["kz"] を (k0+1, k1) に1本ずらした複製を返す（E2専用、他セルには影響しない）。
    母集団のビルド条件(k1-k0>=2)により k0+1 < k1 は常に成立＝全setupsが対象のまま。"""
    out = []
    for rec in setups:
        rec2 = dict(rec)
        s = rec.get("long")
        if s is not None:
            s2 = dict(s)
            k0, k1 = s["kz"]
            s2["kz"] = (k0 + 1, k1)
            rec2["long"] = s2 if (k0 + 1) < k1 else None
        out.append(rec2)
    return out


def cell_stats(trades):
    """era非依存の軽量統計（n・win%・PF・meanR(net)・totR・maxDD はセル内の経路で計算）。
    stats()と同じ定義(win%=gross>0, DD=cumsum(net)の山谷)だが、span/npy/IS-OOSはera分割に不要なため省く。"""
    if len(trades) == 0:
        return None
    net = np.array([t[1] for t in trades]); g = np.array([t[2] for t in trades])
    pos, neg = net[net > 0].sum(), -net[net < 0].sum()
    cum = np.cumsum(net); dd = float((np.maximum.accumulate(cum) - cum).max()) if len(cum) else 0.0
    return dict(n=len(net), win=100 * (g > 0).mean(), meanR=net.mean(),
                pf=(pos / neg if neg > 0 else np.inf), totR=net.sum(), maxDD=dd)


def filter_era(trades, lo, hi):
    return [t for t in trades if lo <= pd.Timestamp(t[0]).year <= hi]


def risk_rr_summary(trade_log_rows, atr_by_date, lo, hi):
    """era内のトレードについて risk/ATR(中央値) と 実現RR(中央値) を trade_log から集計。"""
    rows = [r for r in trade_log_rows if lo <= pd.Timestamp(r["date"]).year <= hi]
    if not rows:
        return None
    risk_atr = []
    rr = []
    for r in rows:
        d = r["date"]
        A = atr_by_date.get(d)
        if A is not None and A > 0:
            risk = r["entry"] - r["stop"]   # long
            risk_atr.append(risk / A)
        rr.append(r["r_rr"])
    return dict(n=len(rows),
                risk_atr_med=(float(np.median(risk_atr)) if risk_atr else float("nan")),
                rr_med=float(np.median(rr)))


def run_cell(df, S_pop, S_pop_e2, atr_by_date, entry_key, stop_key, sp, cost, tgt_fn):
    buf = S1_BUF if stop_key == "S1" else S2_BUF
    trade_log = []
    if entry_key == "E1":
        setups, lim_fn, f = S_pop, EURUSD_LIM_FN, F_CANON
    elif entry_key == "E2":
        setups, lim_fn, f = S_pop_e2, market_lim_fn, F_CANON
    else:  # E3
        setups, lim_fn, f = S_pop, None, F_CANON
    trades = walk(df, setups, f, RR_CANON, buf, sp, cost, "long",
                  lim_fn=lim_fn, tgt_fn=tgt_fn, trade_log=trade_log)
    return trades, trade_log


def fmt_cell_row(label, cs):
    if cs is None:
        return f"    {label:10s} n=0 (該当なし)"
    return (f"    {label:10s} n={cs['n']:4d} win%={cs['win']:5.1f} PF={cs['pf']:5.2f} "
            f"meanR={cs['meanR']:+.3f} totR={cs['totR']:+7.1f} maxDD(era内)={cs['maxDD']:6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2023-以降のみ")
    args = ap.parse_args()

    print("#" * 110)
    print("0. 自己検査: E1×S1 (=優先3旗艦 ext_PDH_fluff5) が台帳 n=313/win34.5/PF1.41/totR-DD4.05/maxDD21.7 を再現するか")
    print("#" * 110)
    with contextlib.redirect_stderr(io.StringIO()):
        df, tarr, dates, span = load_prepped("eurusd")
    if args.smoke:
        dates = dates[-int(len(dates) * 0.25):]

    S_pop = canonical_setups(df, tarr, dates, 0, use_fvg=True, fvg_min_atr=EURUSD_MA,
                              use_liq=True, liq_ns=(20, 40))
    n_pop = sum(1 for rec in S_pop if rec["long"] is not None)
    S_pop_e2 = shift_kz_next_bar(S_pop)
    n_pop_e2 = sum(1 for rec in S_pop_e2 if rec["long"] is not None)
    atr_by_date = {rec["date"]: rec["long"]["atr"] for rec in S_pop if rec["long"] is not None}
    print(f"  n_pop(狩り+MSS+FVG0.15) = {n_pop}  (E2用kz+1シフト後 有効 = {n_pop_e2})  span={span}年")

    sp, cost = cost_tiers("eurusd")["realistic"]
    tgt_fn = make_ext_tgt_fn("pdh", 5, "eurusd", "long")

    tr_anchor, _ = run_cell(df, S_pop, S_pop_e2, atr_by_date, "E1", "S1", sp, cost, tgt_fn)
    st = stats(tr_anchor, span)
    print(f"  再現値: n={st['n']} win%={st['win']:.1f} PF={st['pf']:.2f} meanR={st['net']:+.3f} "
          f"totR/DD={st['rdd']:.2f} maxDD={st['dd']:.1f} IS={st['IS']:+.0f} OOS={st['OOS']:+.0f}")
    if args.smoke:
        print("  自己検査: --smoke は日付を直近25%に絞るため台帳値とは一致しない（想定通り、スクリプトが"
              "エラー無く走ることの確認のみ）")
    else:
        ok = (st['n'] == 313 and abs(st['win'] - 34.5) < 0.1 and abs(st['pf'] - 1.41) < 0.01)
        print(f"  自己検査: {'PASS' if ok else 'FAIL（要確認・以降の結果は疑ってかかること）'}")

    # ---------------- 6セル本測定 ----------------
    print("\n" + "#" * 110)
    print("1. 6セル(入口3×損切り2) × {全史,2018-23,2024,2025,2026}: n/win%/PF/meanR/totR/maxDD(era内)")
    print("   + セル内 risk幅(ATR比中央値)・実現RR中央値")
    print("#" * 110)

    entries = ["E1", "E2", "E3"]
    entry_names = {"E1": "E1_FVG-CE(mid)指値", "E2": "E2_KZ次足成行", "E3": "E3_0.25リトレース指値"}
    stops = ["S1", "S2"]
    stop_names = {"S1": "S1_buf0.1ATR", "S2": "S2_buf0.5ATR"}

    cell_trades = {}
    cell_tradelog = {}
    for ek in entries:
        for sk in stops:
            trades, tlog = run_cell(df, S_pop, S_pop_e2, atr_by_date, ek, sk, sp, cost, tgt_fn)
            cell_trades[(ek, sk)] = trades
            cell_tradelog[(ek, sk)] = tlog

    for ek in entries:
        for sk in stops:
            label = f"{entry_names[ek]} × {stop_names[sk]}"
            print(f"\n  --- {label} ---")
            trades = cell_trades[(ek, sk)]
            tlog = cell_tradelog[(ek, sk)]
            for lo, hi, elabel in ERAS:
                sub = filter_era(trades, lo, hi)
                cs = cell_stats(sub)
                rr_sum = risk_rr_summary(tlog, atr_by_date, lo, hi)
                extra = ""
                if rr_sum is not None:
                    extra = f"  risk/ATR中央={rr_sum['risk_atr_med']:.2f}  実現RR中央={rr_sum['rr_med']:.2f}"
                print(f"    [{elabel:8s}] " + fmt_cell_row("", cs).strip() + extra)

    # ---------------- 逆選択の直接測定 ----------------
    print("\n" + "#" * 110)
    print("2. 逆選択の直接測定: E1/E3 の 指値約定 vs 非約定 で、その後「KZ成行(E2,S1,PDH-5pip)で持っていたら」のmeanRを比較")
    print("   （非約定=指値サーチで見送りになった本。fill判定は tgt_fn=None・rr=4固定のダミーwalkでfp検出のみに使う）")
    print("#" * 110)
    held_trades = cell_trades[("E2", "S1")]   # 「そのまま持っていたら」の基準セル
    held_by_date = {t[0]: t for t in held_trades}

    for ek, lim_fn_desc, lim_fn in (("E1", "FVG-CE(mid)", EURUSD_LIM_FN), ("E3", "0.25リトレース", None)):
        print(f"\n  --- {ek} ({lim_fn_desc}) の約定判定（ダミーRR4固定・tgt_fn=None） ---")
        fill_probe = walk(df, S_pop, F_CANON, 4.0, S1_BUF, sp, cost, "long", lim_fn=lim_fn)
        filled_dates = {t[0] for t in fill_probe}
        pop_dates = {rec["date"] for rec in S_pop if rec["long"] is not None}
        notfilled_dates = pop_dates - filled_dates
        print(f"  母集団n={len(pop_dates)} 約定n={len(filled_dates)} 非約定n={len(notfilled_dates)}")
        for lo, hi, elabel in ERAS:
            f_r = [held_by_date[d][1] for d in filled_dates
                   if d in held_by_date and lo <= pd.Timestamp(d).year <= hi]
            g_r = [held_by_date[d][2] for d in filled_dates
                   if d in held_by_date and lo <= pd.Timestamp(d).year <= hi]
            nf_r = [held_by_date[d][1] for d in notfilled_dates
                    if d in held_by_date and lo <= pd.Timestamp(d).year <= hi]
            nfg_r = [held_by_date[d][2] for d in notfilled_dates
                     if d in held_by_date and lo <= pd.Timestamp(d).year <= hi]
            f_mean = np.mean(f_r) if f_r else float("nan")
            fg_mean = np.mean(g_r) if g_r else float("nan")
            nf_mean = np.mean(nf_r) if nf_r else float("nan")
            nfg_mean = np.mean(nfg_r) if nfg_r else float("nan")
            print(f"    [{elabel:8s}] 約定本(n={len(f_r):3d}): meanR_net(持てば)={f_mean:+.3f} "
                  f"meanR_gross={fg_mean:+.3f}  |  非約定本(n={len(nf_r):3d}): "
                  f"meanR_net(持てば)={nf_mean:+.3f} meanR_gross={nfg_mean:+.3f}  "
                  f"差(約定-非約定,net)={ (f_mean-nf_mean) if (f_r and nf_r) else float('nan'):+.3f}")

    # ---------------- 一言サマリ ----------------
    print("\n" + "#" * 110)
    print("3. 一言サマリ: 2024-26 合算 totR、6セル横並び")
    print("#" * 110)
    for ek in entries:
        row = []
        for sk in stops:
            sub = filter_era(cell_trades[(ek, sk)], 2024, 2026)
            cs = cell_stats(sub)
            tot = cs['totR'] if cs else float("nan")
            n = cs['n'] if cs else 0
            row.append(f"{entry_names[ek]}×{stop_names[sk]}: totR={tot:+.1f}(n={n})")
        print("  " + "   ".join(row))


if __name__ == "__main__":
    main()
