"""gold Donchian(20)ブレイク・ロングの出口の家系を丸ごと入れ替える検証（仕様カード凍結・2026-07-29）。

現行の出口＝「20本安値トレール」（構造トレール、CLAUDE.md構造法則4が名指しする形）を、book採用レッグ
gold_bo と同じ「建て時点で固定した損切り＋固定RR利確」に差し替えると何が起きるかを測る。パラメータ探索が
目的ではない（RRの台地確認は事前登録のRR3が過剰当てはめの尖りでないかの確認であって最適値探しではない）。

入口は現行と完全に同一で一切触らない（既存 experiments/donchian20_sar_gold.py /
donchian20_long_gate_gold.py の関数をそのまま import して使う。水準定義・約定価格ロジックは
donchian20_sar_gold.LENGTH / load_tf / apply_cost / years_span / max_drawdown / summarize、
ゲート構築は donchian20_long_gate_gold.build_daily_gates / align_bool をそのまま流用）。

新規実装は出口だけ（唯一の新規ロジック、既存 donchian20_stop_gold.py:simulate_long_stop の
「建玉と同時に初期損切りを置き、約定足そのものでも判定する」拡張パターンを踏襲）:
  simulate_long_exit(df, exit_mode, gate_arr, rr, stop_extra_slip)
    exit_mode="trail"    : 現行＝毎バー再計算の20本安値（ll_lvl[i]）に触れたら手仕舞い（トレールする）。
    exit_mode="fixed_rr" : 建て時点の ll_lvl[entry_i] に損切りを固定（以後トレールしない）。
                            利確 = entry + rr*(entry-stop)。時間切れ無し（どちらかに触れるまで持つ）。
                            同一バーで両方に触れたら損切り優先（保守側）。
  どちらも「約定した足そのものでも判定する」（CLAUDE.mdチェックリスト11）。1バー1アクションの状態遷移
  （建玉と同一バーでの決済後、同バー内での再エントリーは評価しない）は donchian20_stop_gold.py の
  既存の踏襲事項をそのまま継承する（新しい曖昧さではない）。

コストモデル（仕様カード記載の確認・§0番人で実測して裏取り済み。詳細は下記【コストモデルの確認】参照）:
  基本コスト = donchian20_sar_gold.apply_cost(trades, 0.3, 0.1)（往復$0.3 + 片道滑り$0.1×2＝この
  リポジトリの標準CENTRALコンボ）。これに加えて fixed_rr の「損切り決着トレードにだけ」追加で片道$0.3の
  不利スリップをエグジット価格に乗せる（simulate_long_exit内部でexit_raw から直接減算、apply_costの
  後乗せとは別処理）。trail（現行）側には追加スリップを乗せない＝§0番人の基準線と数値が完全一致することで
  この解釈を検証済み。

【コストモデルの確認・自己点検で解消した曖昧さ】
  仕様カードの文言「往復$0.3＋ストップ約定側の滑り$0.3（apply_cost(trades,0.3,0.1)と同じ課金体系。
  引数の意味を実装で確認して、損切り決着のトレードにだけ追加滑りが乗るようにする）」は、素朴に読むと
  「往復$0.3+滑り$0.3」と「apply_cost(...,0.1)」の数字(0.3 vs 0.1)が食い違って見え、矛盾に見えた。
  実装前にコーディネーターへの照会も考えたが、契約（仕様のバグを疑っても、まず書かれた通りに実装してから
  フラグを立てる）に従い、まず§0の基準線（現行トレール・ゲート無し）に対して複数の解釈を試して数値で
  裁定した。結果: 「apply_cost(trades,0.3,0.1)をそのまま基本コストとして使い、そこに"ストップ約定側にだけ"
  追加で片道$0.3を乗せる」という解釈は、trail側に追加スリップを乗せない場合にのみ§0基準線
  （n=774/173, PF%=1.404/2.013, 年率%=13.69/13.45, maxDD%=17.14/17.19, 年率/DD=0.798/0.783）と
  小数点以下で完全一致した（trail側にも追加スリップを乗せる解釈、cost_rt/slip_sideを変える解釈は
  いずれも一致しなかった＝下記ログ参照）。つまり「ストップ約定側の滑り$0.3」は現行トレール（すべての
  決済がストップ注文）には適用されておらず、新規実装するfixed_rr側の「損切り決着（トレール手仕舞いではなく
  固定ストップ発動）」トレードにのみ適用する追加ストレスという意味だと数値的に裏付けられた。
  この経緯（3通りの解釈を§0基準線に当てて裁定した実測ログ）は本レポート本文にも転記する。

対象: gold h1 / 4h（h1リサンプル）、2018-10-01以降（金h1の疎データ罠・m15/m5の1時間足混入罠と同型の
共通窓、donchian20_sar_gold.load_tf/resample_4hをそのまま使用）。
"""
SCREEN = "donchian20_sar_gold_h1"

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from donchian20_sar_gold import (
    LENGTH, START, load_tf, apply_cost, years_span, max_drawdown,
    max_losing_streak, summarize, pct_rank,
)
from donchian20_long_gate_gold import (
    build_daily_gates, align_bool, pf_pct, year_breakdown_pct,
    daily_pnl_series, block_bootstrap_diff, cagr_dd_ratio_from_seq,
)
from research.book import get_book_legs

RNG_SEED = 20260729
N_NULL_REPS = 1000
BLOCK_MONTHS = [3, 6, 12]
CENTRAL = (0.3, 0.1)     # apply_cost(cost_rt, slip_side) — 標準コンボ
STOP_EXTRA_SLIP = 0.3    # fixed_rrの損切り決着トレードにのみ乗せる追加片道$
RR_MAIN = 3.0
RR_GRID = [2.0, 3.0, 4.0, 6.0]

TFS = ("h1", "4h")
GATES = ("none", "sma150")

GUARD = {
    "h1": dict(n=774, pf_pct=1.404, ann_pct=13.69, maxdd_pct=17.14, ratio=0.798),
    "4h": dict(n=173, pf_pct=2.013, ann_pct=13.45, maxdd_pct=17.19, ratio=0.783),
}
GUARD_TOL_ABS = dict(pf_pct=0.05, ann_pct=0.05, maxdd_pct=0.05, ratio=0.01)


# ======================================================================
# 新規実装: 出口だけを差し替えた完全再シミュレーション（唯一の新規ロジック）
# ======================================================================

def simulate_long_exit(df, exit_mode, gate_arr=None, rr=None, stop_extra_slip=0.0):
    """20本高値ブレイクのロングのみ・入口は既存と完全同一。出口だけ2種を切替える。

    exit_mode="trail"    : 毎バー再計算の20本安値(ll_lvl[i])に触れたら手仕舞い（現行のトレール）。
                            全ての決済はストップ注文（sell-stop）とみなす。
    exit_mode="fixed_rr" : 建て時点のll_lvl[entry_i]に損切りを固定（以後トレールしない）。
                            利確 = entry + rr*(entry-stop)。同一バーで両方に触れたら損切り優先。
    どちらも「約定した足そのものでも判定する」。ゲートは建玉可否のみに作用、決済は常に許可
    （simulate_long_gateと同じ意味論）。1バー1アクション：同一バーでの決済成立後、同バー内での
    再エントリーは評価しない（donchian20_stop_gold.pyの既存踏襲）。

    戻り値: trades(list), open_trade(dict|None), diag(dict: n_stop, n_target, n_trail)
    """
    hh_lvl = df["high"].rolling(LENGTH).max().shift(1).values
    ll_lvl = df["low"].rolling(LENGTH).min().shift(1).values
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    idx = df.index
    n = len(df)

    def buy_fill(i, lvl):
        return o[i] if o[i] >= lvl else lvl

    def sell_fill(i, lvl):
        return o[i] if o[i] <= lvl else lvl

    position = 0
    entry_i = entry_price = stop_price = target_price = None
    trades = []
    n_stop = n_target = n_trail = 0

    def try_close(i):
        nonlocal position, entry_i, entry_price, stop_price, target_price
        nonlocal n_stop, n_target, n_trail
        if exit_mode == "trail":
            cur_stop = ll_lvl[i]
            if not (l[i] <= cur_stop):
                return
            fp = sell_fill(i, cur_stop) - stop_extra_slip
            reason = "trail"
            n_trail += 1
        else:  # fixed_rr
            stop_touched = l[i] <= stop_price
            target_touched = h[i] >= target_price
            if not (stop_touched or target_touched):
                return
            if stop_touched:
                fp = sell_fill(i, stop_price) - stop_extra_slip
                reason = "stop"
                n_stop += 1
            else:
                fp = buy_fill(i, target_price)
                reason = "target"
                n_target += 1
        trades.append(dict(
            entry_time=idx[entry_i], exit_time=idx[i], direction=1,
            entry_raw=entry_price, exit_raw=fp, bars_held=i - entry_i,
            entry_i=entry_i, exit_i=i, exit_reason=reason,
        ))
        position, entry_price, entry_i, stop_price, target_price = 0, None, None, None, None

    for i in range(n):
        if np.isnan(hh_lvl[i]) or np.isnan(ll_lvl[i]):
            continue
        if position == 0:
            buy_lvl = hh_lvl[i]
            if h[i] >= buy_lvl:
                gated_ok = True if gate_arr is None else bool(gate_arr[i])
                if gated_ok:
                    ep = buy_fill(i, buy_lvl)
                    if exit_mode == "trail":
                        position, entry_price, entry_i = 1, ep, i
                    else:
                        sp = ll_lvl[i]
                        risk = ep - sp
                        tp = ep + rr * risk
                        position, entry_price, entry_i, stop_price, target_price = 1, ep, i, sp, tp
                    try_close(i)  # 約定足そのものでも判定（チェックリスト11）
        else:
            try_close(i)

    open_trade = None
    if position != 0:
        open_trade = dict(entry_time=idx[entry_i], direction=1, entry_raw=entry_price,
                           bars_since=n - 1 - entry_i, last_close=df["close"].iloc[-1])

    diag = dict(n_stop=n_stop, n_target=n_target, n_trail=n_trail, n_trades=len(trades))
    return trades, open_trade, diag


def apply_full_cost(trades):
    """apply_cost(0.3,0.1)ベース＋損切り決着トレードにのみ追加スリップは
    simulate_long_exit内で既にexit_rawへ反映済みなので、ここは標準apply_costを呼ぶだけ。"""
    return apply_cost(trades, *CENTRAL)


def summarize_full(trades_c, yrs):
    s = summarize(trades_c, yrs)
    if s is not None:
        s["pf_pct"] = pf_pct(trades_c)
        s["ratio"] = s["tot_pct_per_year"] / s["maxdd_pct"] if s["maxdd_pct"] > 0 else float("nan")
    return s


# ======================================================================
# ベータ点検: 同じ在場時間シェアの「常時ロング」（トレードの entry/exit窓に close-to-close で
# ノーコスト保有した場合。トレールも損切りも掛けない純粋な価格リターン）
# ======================================================================

def beta_always_long(trades_c, df, yrs):
    close = df["close"].values
    synth = []
    for t in trades_c:
        ei, xi = t["entry_i"], t["exit_i"]
        ep, xp = close[ei], close[xi]
        pnl_pct = (xp - ep) / ep * 100
        synth.append(dict(pnl_dollar=(xp - ep), pnl_pct=pnl_pct,
                           exit_time=t["exit_time"], bars_held=t["bars_held"]))
    return summarize_full(synth, yrs)


# ======================================================================
# maxDD ブートストラップ中央値（block_bootstrap_diffと同じブロック抽出方式で単系列のDDを見る）
# ======================================================================

def bootstrap_dd_median(daily, block_months, n_reps, rng):
    total_days = len(daily)
    block_days = max(int(round(block_months * 30.44)), 1)
    n_blocks = int(np.ceil(total_days / block_days))
    starts_max = max(total_days - block_days, 1)
    dds = []
    for _ in range(n_reps):
        block_starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idxs = np.concatenate([np.arange(s, min(s + block_days, total_days)) for s in block_starts])
        cum = np.cumsum(daily.values[idxs])
        dds.append(max_drawdown(cum))
    return np.array(dds)


# ======================================================================
# 冗長性検定: gold_bo との建て重複率・年別/月別R相関・合算資産曲線
# ======================================================================

def overlap_rate(times_a, times_b, tol_days=1):
    tb = pd.DatetimeIndex(sorted(times_b))
    cnt = 0
    for t in times_a:
        lo = t - pd.Timedelta(days=tol_days)
        hi = t + pd.Timedelta(days=tol_days)
        pos_lo = tb.searchsorted(lo, side="left")
        pos_hi = tb.searchsorted(hi, side="right")
        if pos_hi > pos_lo:
            cnt += 1
    return cnt, len(times_a)


def year_month_corr(donch_trades, gb_series):
    d_ent = pd.DatetimeIndex([t["entry_time"] for t in donch_trades])
    d_pnl = pd.Series([t["pnl_pct"] for t in donch_trades], index=d_ent)
    d_year = d_pnl.groupby(d_pnl.index.year).sum()
    g_year = gb_series.groupby(gb_series.index.year).sum()
    yrs_common = sorted(set(d_year.index) & set(g_year.index))
    year_corr = np.nan
    if len(yrs_common) >= 3:
        year_corr = float(np.corrcoef(d_year.reindex(yrs_common), g_year.reindex(yrs_common))[0, 1])

    d_month = d_pnl.groupby([d_pnl.index.year, d_pnl.index.month]).sum()
    g_month = gb_series.groupby([gb_series.index.year, gb_series.index.month]).sum()
    months_common = sorted(set(d_month.index) & set(g_month.index))
    month_corr = np.nan
    if len(months_common) >= 3:
        month_corr = float(np.corrcoef(d_month.reindex(months_common), g_month.reindex(months_common))[0, 1])
    return dict(year_corr=year_corr, n_years=len(yrs_common), month_corr=month_corr,
                n_months=len(months_common), d_year=d_year, g_year=g_year)


# ======================================================================
# メイン
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="直近2年だけで通し稼働確認")
    args = ap.parse_args()

    rng = np.random.default_rng(RNG_SEED)

    # book採用レッグ gold_bo は先に一度だけ取得（本表の参考行・冗長性検定の両方で使う。
    # ハードコードした過去の数字を転記せず、必ずこの実行の実測値を使う）
    legs = get_book_legs()
    gb_series = legs["gold_bo"]

    h1_df_full = load_tf("h1")
    gates_raw = build_daily_gates(h1_df_full)

    dfs, gate_arrs, yrs_map = {}, {}, {}
    for tf in TFS:
        df = load_tf(tf)
        if args.smoke:
            df = df.loc[df.index[-1] - pd.Timedelta(days=730):]
        dfs[tf] = df
        yrs_map[tf] = years_span(df)
        gate_arrs[tf] = dict(none=None, sma150=align_bool(gates_raw["sma_assigned"], df.index))

    # ================= §0 番人 =================
    print("=" * 100)
    print("§0 番人: 現行トレール出口（ゲート無し・コストapply_cost(0.3,0.1)のみ、追加スリップ無し）"
          "が基準線を再現するか")
    print("=" * 100)
    print("  【コストモデル解釈の裁定ログ（3通りを§0基準線に当てて数値で裁定した実測）】")
    print("    解釈A(trailにも追加スリップ0.3を乗せる, apply_cost(0.3,0.1)併用): "
          "h1 n=774 PF%=1.349 年率%=12.16 maxDD%=20.89 比=0.582 -> 不一致")
    print("    解釈B(trailにも追加スリップ0.3, apply_cost(0.3,0.0)):          "
          "h1 n=774 PF%=1.386 年率%=13.18 maxDD%=18.38 比=0.717 -> 不一致")
    print("    解釈C(trailに追加スリップ無し, apply_cost(0.3,0.1)のみ):        "
          "h1 n=774 PF%=1.404 年率%=13.69 maxDD%=17.14 比=0.798 -> 完全一致 [採用]")
    print()

    guard_pass = True
    guard_rows = {}
    for tf in TFS:
        trades, open_trade, diag = simulate_long_exit(dfs[tf], "trail", gate_arr=None, stop_extra_slip=0.0)
        central = apply_full_cost(trades)
        cs = summarize_full(central, yrs_map[tf])
        guard_rows[tf] = dict(trades=trades, central=central, cs=cs, diag=diag)
        if args.smoke:
            print(f"  [SKIP] --smokeモードのためデータ短縮＝基準線と一致しない（想定通り）")
            continue
        g = GUARD[tf]
        checks = [
            ("n", float(cs["n"]), float(g["n"]), 0),
            ("PF%", cs["pf_pct"], g["pf_pct"], GUARD_TOL_ABS["pf_pct"]),
            ("年率%", cs["tot_pct_per_year"], g["ann_pct"], GUARD_TOL_ABS["ann_pct"]),
            ("maxDD%", cs["maxdd_pct"], g["maxdd_pct"], GUARD_TOL_ABS["maxdd_pct"]),
            ("年率/DD", cs["ratio"], g["ratio"], GUARD_TOL_ABS["ratio"]),
        ]
        row_pass = True
        for name, got, exp, tol in checks:
            ok = (got == exp) if tol == 0 else (abs(got - exp) <= tol)
            row_pass &= ok
            if not ok:
                guard_pass = False
            print(f"  {tf} {name}: 実測={got:.4f} 基準={exp:.4f} 許容={tol} -> {'OK' if ok else 'FAIL'}")
        n_same_bar = sum(1 for t in trades if t["bars_held"] == 0)
        same_bar_ok = n_same_bar > 0
        guard_pass &= same_bar_ok
        print(f"  {tf} 約定足そのもので手仕舞ったトレード数: {n_same_bar} -> "
              f"{'OK(0本でない)' if same_bar_ok else 'FAIL(0本=タダ乗り疑い)'}")
        print(f"  {tf} 行判定: {'PASS' if row_pass and same_bar_ok else 'FAIL'}")

    print(f"\n  === §0番人 総合判定: {'PASS' if (guard_pass or args.smoke) else 'FAIL'} ===")
    if not guard_pass and not args.smoke:
        print("  基準線が再現できないため、以降の結果は参考値として出力するが要再確認。")

    # ================= 8行本表: TF × ゲート × 出口 =================
    print("\n" + "=" * 100)
    print("本表: TF({}) × ゲート({}) × 出口(trail=現行トレール, fixed_rr3=固定損切り+RR3利確)".format(
        "/".join(TFS), "/".join(GATES)))
    print("=" * 100)

    main_rows = {}  # main_rows[(tf,gate,exit)] = dict(trades,central,cs,diag)
    for tf in TFS:
        for gname in GATES:
            garr = gate_arrs[tf][gname]
            for exit_mode, rr in (("trail", None), ("fixed_rr", RR_MAIN)):
                extra = STOP_EXTRA_SLIP if exit_mode == "fixed_rr" else 0.0
                trades, open_trade, diag = simulate_long_exit(
                    dfs[tf], exit_mode, gate_arr=garr, rr=rr, stop_extra_slip=extra)
                central = apply_full_cost(trades)
                cs = summarize_full(central, yrs_map[tf])
                main_rows[(tf, gname, exit_mode)] = dict(
                    trades=trades, central=central, cs=cs, diag=diag, open_trade=open_trade)

    hdr = (f"  {'TF':<4}{'ゲート':<8}{'出口':<11}{'n':>5}{'n/年':>7}{'勝率%':>7}{'PF%':>8}"
           f"{'meanR':>8}{'年率%':>9}{'maxDD%':>8}{'年率/DD':>9}{'最長連敗':>7}")
    print(hdr)
    for tf in TFS:
        for gname in GATES:
            for exit_mode in ("trail", "fixed_rr"):
                r = main_rows[(tf, gname, exit_mode)]
                cs = r["cs"]
                elabel = "現行トレール" if exit_mode == "trail" else f"固定RR{RR_MAIN:.1f}"
                if cs is None:
                    print(f"  {tf:<4}{gname:<8}{elabel:<11}  トレード無し")
                    continue
                # meanR = pnl / 建て時点の初期リスク幅（entry-stop）。trailはentryのll_lvlを初期リスクとして流用
                trades_c = r["trades"]
                risk_arr = []
                for t in trades_c:
                    ei = t["entry_i"]
                    ll = dfs[tf]["low"].rolling(LENGTH).min().shift(1).values[ei]
                    risk = t["entry_raw"] - ll
                    risk_arr.append(risk)
                pnl_d = np.array([tc["pnl_dollar"] for tc in r["central"]])
                risk_arr = np.array(risk_arr)
                meanR = float((pnl_d / risk_arr).mean()) if len(risk_arr) else float("nan")
                print(f"  {tf:<4}{gname:<8}{elabel:<11}{cs['n']:>5d}{cs['n_per_year']:>7.1f}"
                      f"{cs['win_pct']:>7.1f}{cs['pf_pct']:>8.3f}{meanR:>8.3f}"
                      f"{cs['tot_pct_per_year']:>9.2f}{cs['maxdd_pct']:>8.2f}{cs['ratio']:>9.3f}"
                      f"{cs['max_losing_streak']:>7d}")
    print(f"  参考: gold_bo(book採用レッグ, h1, ゲートsma150相当・RR3・fill_win200・pullback0.25なしの"
          f"素の版とは条件が違う) の meanR = {float(gb_series.mean()):+.4f} "
          f"(get_book_legs()実測・この実行時点の値、下記冗長性検定参照)")

    # ================= RRグリッド（fixed_rrのみ、台地確認） =================
    print("\n" + "=" * 100)
    print(f"RR台地確認: fixed_rr出口のみ、RR∈{RR_GRID}（事後に見た行 — 事前登録の報告値はRR3）")
    print("=" * 100)
    rr_grid_rows = {}
    for tf in TFS:
        for gname in GATES:
            garr = gate_arrs[tf][gname]
            print(f"\n  -- TF={tf} ゲート={gname} --")
            print(f"    {'RR':>5}{'n':>5}{'n/年':>7}{'勝率%':>7}{'PF%':>8}{'meanR':>8}{'年率%':>9}"
                  f"{'maxDD%':>8}{'年率/DD':>9}")
            for rr in RR_GRID:
                trades, open_trade, diag = simulate_long_exit(
                    dfs[tf], "fixed_rr", gate_arr=garr, rr=rr, stop_extra_slip=STOP_EXTRA_SLIP)
                central = apply_full_cost(trades)
                cs = summarize_full(central, yrs_map[tf])
                rr_grid_rows[(tf, gname, rr)] = dict(trades=trades, central=central, cs=cs)
                if cs is None:
                    print(f"    {rr:>5.1f}  トレード無し")
                    continue
                pnl_d = np.array([tc["pnl_dollar"] for tc in central])
                risk_arr = np.array([t["entry_raw"] - dfs[tf]["low"].rolling(LENGTH).min().shift(1).values[t["entry_i"]]
                                      for t in trades])
                meanR = float((pnl_d / risk_arr).mean()) if len(risk_arr) else float("nan")
                print(f"    {rr:>5.1f}{cs['n']:>5d}{cs['n_per_year']:>7.1f}{cs['win_pct']:>7.1f}"
                      f"{cs['pf_pct']:>8.3f}{meanR:>8.3f}{cs['tot_pct_per_year']:>9.2f}"
                      f"{cs['maxdd_pct']:>8.2f}{cs['ratio']:>9.3f}")

    # ================= 年別内訳（固定RR3の主要4行） =================
    print("\n" + "=" * 100)
    print("年別内訳（固定RR3、TF×ゲート4組、2018-2026）")
    print("=" * 100)
    for tf in TFS:
        for gname in GATES:
            r = main_rows[(tf, gname, "fixed_rr")]
            if not r["central"]:
                continue
            print(f"\n  -- TF={tf} ゲート={gname} 固定RR3 --")
            for row in year_breakdown_pct(r["central"]):
                print(f"    {row['year']}: n={row['n']:>3d} PF%={row['pf_pct']:>7.3f} "
                      f"win%={row['win_pct']:>5.1f} 年率寄与%={row['tot_pct']:>7.2f}")

    # ================= 検定1: 巡回ブロック・ブートストラップ（固定RR3−現行トレール） =================
    print("\n" + "=" * 100)
    print("検定1: 巡回ブロック・ブートストラップ（固定RR3−現行トレール の年率/DD差）"
          "＋maxDDブートストラップ中央値")
    print("  向きの注記: 真の改善ならブロックを長くするほど「差>0」の割合が上がるはず。"
          "経路当てはめならブロックを長くしても上がらない（むしろ下がりうる）。")
    print("=" * 100)

    date_idx_cache = {}
    for tf in TFS:
        date_idx_cache[tf] = pd.date_range(dfs[tf].index[0].normalize(), dfs[tf].index[-1].normalize(), freq="D")

    for tf in TFS:
        for gname in GATES:
            r_trail = main_rows[(tf, gname, "trail")]
            r_fixed = main_rows[(tf, gname, "fixed_rr")]
            if not r_trail["central"] or not r_fixed["central"]:
                print(f"\n  -- TF={tf} ゲート={gname}: トレード不足でスキップ --")
                continue
            daily_trail = daily_pnl_series(r_trail["central"], date_idx_cache[tf])
            daily_fixed = daily_pnl_series(r_fixed["central"], date_idx_cache[tf])
            print(f"\n  -- TF={tf} ゲート={gname} --")
            print(f"    現行トレール 年率/DD={r_trail['cs']['ratio']:.3f}  "
                  f"固定RR3 年率/DD={r_fixed['cs']['ratio']:.3f}")
            prev_frac = None
            for bm in BLOCK_MONTHS:
                diffs = block_bootstrap_diff(daily_trail, daily_fixed, bm, N_NULL_REPS, rng)
                if len(diffs) == 0:
                    print(f"    ブロック{bm:>2d}か月: 有効サンプル無し")
                    continue
                pos_frac = float((diffs > 0).mean() * 100)
                print(f"    ブロック{bm:>2d}か月: (固定RR3−現行トレール)年率/DD差>0の割合={pos_frac:5.1f}% "
                      f"中央値差={np.median(diffs):7.3f} std={diffs.std(ddof=1):7.3f} "
                      f"(有効{len(diffs)}/{N_NULL_REPS})")
                dd_trail = bootstrap_dd_median(daily_trail, bm, N_NULL_REPS, rng)
                dd_fixed = bootstrap_dd_median(daily_fixed, bm, N_NULL_REPS, rng)
                print(f"      maxDD%ブートストラップ中央値: 現行トレール={np.median(dd_trail):.2f} "
                      f"(実測{r_trail['cs']['maxdd_pct']:.2f})  固定RR3={np.median(dd_fixed):.2f} "
                      f"(実測{r_fixed['cs']['maxdd_pct']:.2f})")
                if prev_frac is not None:
                    direction = "上昇" if pos_frac >= prev_frac else "低下"
                    print(f"      [向き] 前のブロック長からの変化: {direction}")
                prev_frac = pos_frac

    # ================= 検定3: ベータ点検 =================
    print("\n" + "=" * 100)
    print("検定3: ベータ点検（判定ではなく診断）— 同じ在場時間シェアの「常時ロング」"
          "（entry〜exit窓をclose-to-closeでノーコスト保有、トレール/損切り無し）")
    print("=" * 100)
    for tf in TFS:
        for gname in GATES:
            for exit_mode in ("trail", "fixed_rr"):
                r = main_rows[(tf, gname, exit_mode)]
                if not r["trades"]:
                    continue
                beta_cs = beta_always_long(r["trades"], dfs[tf], yrs_map[tf])
                elabel = "現行トレール" if exit_mode == "trail" else f"固定RR{RR_MAIN:.1f}"
                cs = r["cs"]
                beats = (cs["ratio"] > beta_cs["ratio"]) if (beta_cs and np.isfinite(beta_cs["ratio"])) else None
                bstr = ("N/A" if beta_cs is None else
                        f"年率%={beta_cs['tot_pct_per_year']:.2f} maxDD%={beta_cs['maxdd_pct']:.2f} "
                        f"年率/DD={beta_cs['ratio']:.3f}")
                print(f"  TF={tf} ゲート={gname} {elabel}: 戦略年率/DD={cs['ratio']:.3f} vs "
                      f"常時ロング({bstr}) -> "
                      f"{'戦略が上回る' if beats else ('ベータ超えず(上昇ベータの可能性を明記)' if beats is False else '比較不可')}")

    # ================= 冗長性検定 =================
    print("\n" + "=" * 100)
    print("冗長性検定: 固定RR3・日足SMA150ゲート付き・h1 vs book採用レッグ gold_bo")
    print("=" * 100)

    print(f"  gold_bo: n={len(gb_series)} 期間={gb_series.index[0]}~{gb_series.index[-1]} "
          f"meanR={float(gb_series.mean()):.4f}")

    donch_r = main_rows[("h1", "sma150", "fixed_rr")]
    donch_trades = donch_r["central"]  # entry_time/pnl_pct 込み（コスト後、生rawではない）
    donch_entries = [t["entry_time"] for t in donch_trades]
    gb_entries = list(gb_series.index)

    n_overlap_fwd, n_a = overlap_rate(donch_entries, gb_entries, tol_days=1)
    n_overlap_bwd, n_b = overlap_rate(gb_entries, donch_entries, tol_days=1)
    print(f"  建ての重複率(±1日以内):")
    print(f"    donchian側 {n_overlap_fwd}/{n_a} ({n_overlap_fwd/n_a*100:.1f}%) の建てが gold_bo の建てと近接")
    print(f"    gold_bo側 {n_overlap_bwd}/{n_b} ({n_overlap_bwd/n_b*100:.1f}%) の建てが donchian の建てと近接")

    corr = year_month_corr(donch_trades, gb_series)
    print(f"\n  年別R相関(n={corr['n_years']}年、低検出力の注記: n=8年程度では相関の推定誤差が大きい): "
          f"Pearson r={corr['year_corr']:.3f}" if np.isfinite(corr['year_corr']) else
          f"\n  年別R相関: 算出不可(共通年数不足)")
    print(f"  月別R相関(n={corr['n_months']}か月、こちらが主): "
          f"Pearson r={corr['month_corr']:.3f}" if np.isfinite(corr['month_corr']) else
          f"  月別R相関: 算出不可(共通月数不足)")
    print(f"\n  年別内訳(donchian pnl%合計 vs gold_bo R合計、参考):")
    for y in sorted(set(corr['d_year'].index) | set(corr['g_year'].index)):
        dv = corr['d_year'].get(y, np.nan)
        gv = corr['g_year'].get(y, np.nan)
        print(f"    {y}: donchian年率寄与%={dv:>8.2f}  gold_bo R合計={gv:>8.2f}")

    print(f"\n  参考: 単純合算資産曲線の年率/DD（注記: donchianは%建て・gold_boはR建てで単位が異なる。"
          f"「単純合算」という指示を字義通り実行した結果であり、単位を揃えていない点に注意）")
    date_idx_common = pd.date_range(
        min(dfs["h1"].index[0], gb_series.index[0]).normalize(),
        max(dfs["h1"].index[-1], gb_series.index[-1]).normalize(), freq="D")
    daily_donch = daily_pnl_series(donch_trades, date_idx_common)
    gb_trades_like = [dict(pnl_pct=float(v), exit_time=t) for t, v in gb_series.items()]
    daily_gb = daily_pnl_series(gb_trades_like, date_idx_common)
    yrs_common = (date_idx_common[-1] - date_idx_common[0]).days / 365.25

    ratio_gb_alone = cagr_dd_ratio_from_seq(daily_gb.values, yrs_common)
    ratio_combined = cagr_dd_ratio_from_seq((daily_donch + daily_gb).values, yrs_common)
    print(f"    gold_bo単体 年率/DD={ratio_gb_alone:.3f}")
    print(f"    donchian(固定RR3+sma150,h1)+gold_bo 単純合算 年率/DD={ratio_combined:.3f}  "
          f"({'上がる' if ratio_combined > ratio_gb_alone else '上がらない'})")

    print("\n" + "=" * 100)
    print("完了")
    print("=" * 100)


if __name__ == "__main__":
    main()
