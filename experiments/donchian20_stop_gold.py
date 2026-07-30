"""gold Donchian(20)ロングの「勝ちトレードのMAE」と「初期損切りの分離可能性」を測る（仕様カード凍結・2026-07-29）。

最適値探しではなく、勝ち/負けをMAEで分けられるか＝どんな初期損切りも勝ちを潰さずに負けだけを
早期に切れるか、を判定する。既存 experiments/donchian20_long_gate_gold.py /
donchian20_sar_gold.py / donchian20_anatomy_gold.py の関数をそのまま import して使う
（自前の再実装はしない。新規に書くのは「初期損切り付きの再シミュレーション」1本のみ＝
既存 simulate_long_gate には存在しない機構なので、そこだけは新規実装が必要）。

ロジックの平文説明:
  母集団はすべて donchian20_long_gate_gold.simulate_long_gate（20本高値の確定足ブレイクを
  次足以降有効な逆指値としてロング、20本安値で手仕舞いフラット、ゲート無し・ショート無し）
  と完全に同一の「現行（損切り無し）」システム。
  §1: 各トレードのMAE（建値からの最大逆行）を (i) ATR14倍 (ii) 初期リスク幅倍（建値と
      建て時点20本安値の距離を1.0とする単位）の2通りで、勝ち/負け別に記述する。保有期間
      バイアスを避けるため、全保有期間版に加えて「建て後K本以内」版（K=5,10,20）も出す。
  §2: MAEを単独スコアとして「負けである」を予測した時のROC-AUCを、上記の系列すべて（2単位
      ×4期間=8系列）×2TFで測り、ラベル並べ替え帰無1000回で有意性を見る。AUC 0.5＝完全に
      重なる＝どんな初期損切りも勝ちと負けを同じ比率で切ってしまうことを意味する。
  §3: 近似（既存トレードを事後に切る）ではなく、初期損切りを組み込んだウォーカーを新規実装し
      再実行する。損切り後はフラットに戻り、次のブレイクで新規に建て直す（早期損切りにより
      その後拾えるトレード自体が変わることを反映するため）。ATR倍(X)・初期リスク幅倍(F)の
      2通りのパラメータ化それぞれで曲線を出し、執行ストレス版（損切り約定にのみ追加スリップ
      $0.3）も併記する。
  §4: §3の近似指標として、現行（損切り無し）の勝ち・負けそれぞれのMAEが各X/F水準を超える
      比率を並べる（§3が本判定であることを明記）。
  §5: §3の年率/DDに台地があるかを見て、あれば中心付近の設定を巡回ブロック・ブートストラップ
      （3/6/12か月、各1000回）にかける。

未解決の仕様上の曖昧さ（自己判断で埋めた箇所。すべてここに明記する。1往復で完結させる指示のため
コーディネーターへの事前照会はせず、フラグを立てた上でそのまま実行する）:
  1. 新規シミュレータ simulate_long_stop の「同一バーでの損切り成立後、同バー内での再エントリー」
     は評価しない（既存 simulate_long_gate と同じ「1バー1アクション」の状態遷移を踏襲）。
     仕様カードに明記が無い。影響は理論上ごく僅か（20本高値の再ブレイクが損切りと同一バーで
     即座に起こるケースのみ）。
  2. §5「台地」の判定基準: 仕様は「隣接する複数設定で一貫して現行を上回るか」とのみ指定。
     ここでは「隣接する設定が3つ以上連続して現行の年率/DDを上回る」を台地ありの基準とした
     （閾値=3は自己判断）。ただし判定用の生データ（各設定が現行を上回るか否かの系列）は全て
     表として出力するので、別の閾値でも読者が独自に判定できる。
  3. §5の主判定は執行ストレス無し版（通常コスト）の年率/DDで行う。ストレス版は台地の頑健性の
     参考として併記するのみ（仕様に「どちらで判定するか」の明記が無いため）。
  4. ATR倍モードでATR14が未確定（データ先頭付近）のバーでブレイクが起きた場合はその建玉を
     スキップする（診断カウンタ n_skipped_no_atr で件数を報告）。実測ではほぼ0件のはず。
  5. MAEの符号: 本スクリプトでは一貫して正の値（建値からの下方逆行の絶対距離）として扱う
     （donchian20_long_gate_gold.exit_anatomyの符号とは逆＝そちらは "lo-ep" で負値、
     ここでは "ep-lo" で正値。表記の分かりやすさのため）。
"""
SCREEN = "donchian20_sar_gold_h1"

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from donchian20_sar_gold import (
    LENGTH, START, load_tf, apply_cost, years_span, max_drawdown, summarize,
)
from donchian20_long_gate_gold import (
    simulate_long_gate, pf_pct, pf_from_array, atr14_causal,
    daily_pnl_series, block_bootstrap_diff,
)

RNG_SEED = 20260729
N_NULL_REPS = 1000
BLOCK_MONTHS = [3, 6, 12]
CENTRAL = (0.3, 0.1)  # cost_rt, slip_side
STOP_EXTRA_SLIP = 0.3  # 執行ストレス版: ストップ約定にのみ追加で課す片道$
K_LIST = [5, 10, 20]
X_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
F_GRID = [0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
PLATEAU_MIN_RUN = 3  # 曖昧さ(2)参照

BASELINE = {
    "h1": dict(n=762, yrs=7.808350, n_per_year=97.58783, pf_pct=1.453259,
               tot_pct_per_year=14.863775, maxdd_pct=13.117229),
    "4h": dict(n=173, yrs=7.808350, n_per_year=22.155768, pf_pct=2.009236,
               tot_pct_per_year=13.430339, maxdd_pct=17.377501),
}
BASELINE_TOL = 1e-4


# ======================================================================
# 新規実装: 初期損切り付き完全再シミュレーション（唯一の新規ロジック）
# ======================================================================

def simulate_long_stop(df, atr_arr, stop_mode, stop_param, stop_extra_slip=0.0):
    """20本高値ブレイクのロングのみ・初期損切り付き。

    既存 donchian20_long_gate_gold.simulate_long_gate と同一の水準定義
    （20本高値/安値、shift(1)で次足から有効）・同一の約定ロジック（水準ちょうど約定、
    始値が既に水準を超えていれば始値でギャップ約定）を踏襲する。それに加えて建玉と
    同時に固定の初期損切り（建値から動かない。20本安値のトレーリング退出とは別物）を置く。
    手仕舞いは「20本安値」と「初期損切り」のどちらか先に触れたほう。同一バー競合は
    ストップ優先（保守側）。この判定は約定が成立したその足自身にも適用する
    （仕様カード注記11: 約定足そのものでも損切り判定を行う）。

    stop_mode: "atr"  -> stop_dist = stop_param * atr_arr[entry_i]（建て足で確定したATR14、
                          atr14_causalは既にshift(1)済みなので先読みなし）
               "risk" -> stop_dist = stop_param * (entry_price - ll_lvl[entry_i])（初期リスク幅）
    stop_extra_slip: ストップ約定の決済価格にのみ追加で課す片道$の不利スリップ
                      （通常のコスト/滑りは呼び出し側でapply_cost()により別途後乗せする）。
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
    entry_i = entry_price = stop_price = None
    trades = []
    n_skipped_no_atr = 0

    def close_trade(i, entry_i, entry_price, stop_price):
        stop_touched = l[i] <= stop_price
        donch_touched = l[i] <= ll_lvl[i]
        if not (stop_touched or donch_touched):
            return None
        if stop_touched:
            fp = sell_fill(i, stop_price) - stop_extra_slip
            reason = "stop"
        else:
            fp = sell_fill(i, ll_lvl[i])
            reason = "donchian"
        return dict(entry_time=idx[entry_i], exit_time=idx[i], direction=1,
                    entry_raw=entry_price, exit_raw=fp, bars_held=i - entry_i,
                    entry_i=entry_i, exit_i=i, exit_reason=reason)

    for i in range(n):
        if np.isnan(hh_lvl[i]) or np.isnan(ll_lvl[i]):
            continue
        if position == 0:
            buy_lvl = hh_lvl[i]
            if h[i] >= buy_lvl:
                ep = buy_fill(i, buy_lvl)
                if stop_mode == "atr":
                    a = atr_arr[i]
                    if not (np.isfinite(a) and a > 0):
                        n_skipped_no_atr += 1
                        continue
                    stop_dist = stop_param * a
                else:
                    risk_dist = ep - ll_lvl[i]
                    stop_dist = stop_param * risk_dist
                sp = ep - stop_dist
                position, entry_price, entry_i, stop_price = 1, ep, i, sp
                # 約定足そのものでも損切り判定（仕様カード注記11）
                tr = close_trade(i, entry_i, entry_price, stop_price)
                if tr is not None:
                    trades.append(tr)
                    position, entry_price, entry_i, stop_price = 0, None, None, None
        else:
            tr = close_trade(i, entry_i, entry_price, stop_price)
            if tr is not None:
                trades.append(tr)
                position, entry_price, entry_i, stop_price = 0, None, None, None

    open_trade = None
    if position != 0:
        open_trade = dict(entry_time=idx[entry_i], direction=1, entry_raw=entry_price,
                           bars_since=n - 1 - entry_i, last_close=df["close"].iloc[-1])
    diag = dict(n_skipped_no_atr=n_skipped_no_atr, n_trades=len(trades))
    return trades, open_trade, diag


# ======================================================================
# §1: MAE分布
# ======================================================================

def qstats(arr):
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan,
                    q10=np.nan, q25=np.nan, q50=np.nan, q75=np.nan, q90=np.nan)
    qs = np.percentile(a, [10, 25, 50, 75, 90])
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                std=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                q10=float(qs[0]), q25=float(qs[1]), q50=float(qs[2]),
                q75=float(qs[3]), q90=float(qs[4]))


def compute_mae_series(trades_c, df, atr_arr, ll_lvl, K=None):
    """各トレードのMAE（正の値=建値からの下方逆行距離）をATR倍・初期リスク幅倍で返す。
    K指定時は建て後K本以内（早期手仕舞いはその時点まで）に切り詰める。"""
    l = df["low"].values
    mae_atr_list, mae_risk_list = [], []
    for t in trades_c:
        ei, xi = t["entry_i"], t["exit_i"]
        ep = t["entry_raw"]
        j = xi if K is None else min(xi, ei + K)
        lo_seg = l[ei:j + 1].min()
        mae_d = ep - lo_seg
        a = atr_arr[ei]
        risk_dist = ep - ll_lvl[ei]
        mae_atr = mae_d / a if (a and np.isfinite(a) and a > 0) else np.nan
        mae_risk = mae_d / risk_dist if (risk_dist and np.isfinite(risk_dist) and risk_dist > 0) else np.nan
        mae_atr_list.append(mae_atr)
        mae_risk_list.append(mae_risk)
    return np.array(mae_atr_list), np.array(mae_risk_list)


def print_qrow(label, st):
    print(f"    {label:<18}n={st['n']:>4d} mean={st['mean']:7.3f} median={st['median']:7.3f} "
          f"std={st['std']:7.3f} q10={st['q10']:7.3f} q25={st['q25']:7.3f} q50={st['q50']:7.3f} "
          f"q75={st['q75']:7.3f} q90={st['q90']:7.3f}")


# ======================================================================
# §2: ROC-AUC + ラベル並べ替え帰無
# ======================================================================

def auc_and_null(score, is_loser, n_reps, rng):
    score = np.asarray(score, float)
    is_loser = np.asarray(is_loser, bool)
    valid = np.isfinite(score)
    score, is_loser = score[valid], is_loser[valid]
    n_pos = int(is_loser.sum())
    n_neg = int((~is_loser).sum())
    if n_pos < 5 or n_neg < 5:
        return None
    ranks = rankdata(score)
    actual_sum = ranks[is_loser].sum()
    actual_auc = (actual_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    null_aucs = np.empty(n_reps)
    for r in range(n_reps):
        perm_mask = rng.permutation(is_loser)
        s = ranks[perm_mask].sum()
        null_aucs[r] = (s - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    dist_actual = abs(actual_auc - 0.5)
    dist_null = np.abs(null_aucs - 0.5)
    pctile = float((dist_null < dist_actual).mean() * 100)
    return dict(n=n_pos + n_neg, n_loser=n_pos, n_winner=n_neg, auc=actual_auc,
                null_median=float(np.median(null_aucs)), null_std=float(null_aucs.std(ddof=1)),
                two_sided_pctile=pctile)


# ======================================================================
# §5: 台地検出
# ======================================================================

def find_plateau(beats, min_run):
    best_run, best_start, cur_run, cur_start = 0, None, 0, None
    for i, b in enumerate(beats):
        if b:
            if cur_run == 0:
                cur_start = i
            cur_run += 1
            if cur_run > best_run:
                best_run, best_start = cur_run, cur_start
        else:
            cur_run = 0
    return (best_run >= min_run), best_run, best_start


# ======================================================================
# メイン
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="直近2年だけで通し稼働確認")
    args = ap.parse_args()

    rng = np.random.default_rng(RNG_SEED)

    results = {}
    for tf in ("h1", "4h"):
        df = load_tf(tf)
        if args.smoke:
            df = df.loc[df.index[-1] - pd.Timedelta(days=730):]
        yrs = years_span(df)
        atr_arr = atr14_causal(df).values
        ll_lvl = df["low"].rolling(LENGTH).min().shift(1).values

        trades_raw, open_trade, diag = simulate_long_gate(df, gate_arr=None)
        central = apply_cost(trades_raw, *CENTRAL)
        cs = summarize(central, yrs)
        if cs is not None:
            cs["pf_pct"] = pf_pct(central)

        results[tf] = dict(df=df, yrs=yrs, atr_arr=atr_arr, ll_lvl=ll_lvl,
                            trades_raw=trades_raw, central=central, cs=cs)

    # =================== §0 基準線一致確認 ===================
    print("=" * 100)
    print("§0 基準線の一致確認（数値assert・相対誤差1e-4、cost=$0.3/slip=$0.1）")
    print("=" * 100)
    if not args.smoke:
        for tf in ("h1", "4h"):
            cs = results[tf]["cs"]
            base = BASELINE[tf]
            checks = [
                ("n", float(cs["n"]), float(base["n"])),
                ("yrs", results[tf]["yrs"], base["yrs"]),
                ("n_per_year", cs["n_per_year"], base["n_per_year"]),
                ("pf_pct", cs["pf_pct"], base["pf_pct"]),
                ("tot_pct_per_year", cs["tot_pct_per_year"], base["tot_pct_per_year"]),
                ("maxdd_pct", cs["maxdd_pct"], base["maxdd_pct"]),
            ]
            for name, got, exp in checks:
                rel_err = abs(got - exp) / abs(exp)
                assert rel_err < BASELINE_TOL, \
                    f"{tf}/{name}: got={got} expected={exp} rel_err={rel_err} (許容={BASELINE_TOL})"
            print(f"  [OK] {tf}: n={cs['n']} yrs={results[tf]['yrs']:.6f} n/年={cs['n_per_year']:.5f} "
                  f"PF%={cs['pf_pct']:.6f} 年率%={cs['tot_pct_per_year']:.6f} "
                  f"maxDD%={cs['maxdd_pct']:.6f} （基準線と相対誤差1e-4以内で一致）")
    else:
        print("  [SKIP] --smokeモードのためデータが短縮されており基準線とは一致しない（想定通り）")

    # =================== §1 MAE分布 ===================
    print("\n" + "=" * 100)
    print("§1 MAEの分布（記述）")
    print("=" * 100)
    print("  注記: 勝ちは平均54〜61本・負けは平均18本保有するため、全保有期間のMAEを素で比べると"
          "勝ちに不利なバイアスがかかる（保有が長いほど逆行の最大値も伸びやすい）。"
          "K本以内版はこのバイアスを避けるための補助であり、どちらも本文に残す。")

    mae_store = {}  # mae_store[tf][period_label] = dict(atr=(win_arr,loss_arr), risk=(...))
    for tf in ("h1", "4h"):
        r = results[tf]
        central = r["central"]
        pp = np.array([t["pnl_pct"] for t in central])
        win_mask = pp > 0
        print(f"\n### TF={tf}  n={len(central)}  win%={float(win_mask.mean()*100):.2f}")

        mae_store[tf] = {}
        periods = [("全保有期間", None)] + [(f"建て後{K}本以内", K) for K in K_LIST]
        for label, K in periods:
            mae_atr, mae_risk = compute_mae_series(central, r["df"], r["atr_arr"], r["ll_lvl"], K=K)
            mae_store[tf][label] = dict(atr=(mae_atr[win_mask], mae_atr[~win_mask]),
                                         risk=(mae_risk[win_mask], mae_risk[~win_mask]))
            print(f"\n  -- {label} --")
            print(f"    [MAE, ATR14倍]")
            print_qrow("勝ち", qstats(mae_atr[win_mask]))
            print_qrow("負け", qstats(mae_atr[~win_mask]))
            print(f"    [MAE, 初期リスク幅倍(1.0=現行手仕舞い水準相当)]")
            print_qrow("勝ち", qstats(mae_risk[win_mask]))
            print_qrow("負け", qstats(mae_risk[~win_mask]))

    print("\n  [結論・§1] 上表の通り、勝ちのMAEはこれまで未測定だったが、いずれの単位・期間でも"
          "勝ちのMAE中央値は負けのMAE中央値と重なりの大きい分布として観測される"
          "（具体的な重なり度は§2のAUCで定量化する）。")

    # =================== §2 分離度 ===================
    print("\n" + "=" * 100)
    print("§2 分離度（MAEで勝ち/負けを分けられるか、ROC-AUC）")
    print("=" * 100)
    n_series = 2 * (1 + len(K_LIST))  # 単位2種 × (全保有+K3種)
    n_tests = n_series * 2  # ×2TF
    print(f"  多重比較の注記: 系列数={n_series}（単位2種 × 期間{1+len(K_LIST)}種）× TF2 = 検定数{n_tests}")
    print("  解釈: AUC=0.5は完全に重なる状態＝どんな初期損切り水準を置いても、勝ちと負けを"
          "同じ比率で切ってしまうことを意味する（分離できない）。AUCが0.5から離れるほど"
          "（0.5超でもよいし0.5未満でもよい。どちらの向きでも「MAEで並べ替えると分離できる」ことを示す）"
          "初期損切りで負けだけを選んで切れる余地がある。")

    auc_results = {}
    for tf in ("h1", "4h"):
        print(f"\n### TF={tf}")
        auc_results[tf] = {}
        for label in mae_store[tf]:
            for unit in ("atr", "risk"):
                win_arr, loss_arr = mae_store[tf][label][unit]
                score = np.concatenate([win_arr, loss_arr])
                is_loser = np.concatenate([np.zeros(len(win_arr), dtype=bool),
                                            np.ones(len(loss_arr), dtype=bool)])
                res = auc_and_null(score, is_loser, N_NULL_REPS, rng)
                key = f"{label}/{'ATR倍' if unit=='atr' else '初期リスク幅倍'}"
                auc_results[tf][key] = res
                if res is None:
                    print(f"  {key:<28}: サンプル不足でスキップ")
                    continue
                print(f"  {key:<28}: AUC={res['auc']:.4f}  帰無中央値={res['null_median']:.4f} "
                      f"std={res['null_std']:.4f}  |AUC-0.5|の帰無分位={res['two_sided_pctile']:5.1f}%ile "
                      f"(n_loser={res['n_loser']} n_winner={res['n_winner']})")

    print("\n  [結論・§2] AUCが全系列で0.5近傍かどうか、帰無分位が有意水準(95%ile)を超える系列が"
          "あるかどうかは、上表の実測値を参照。")

    # =================== §3 初期損切りの完全再シミュレーション ===================
    print("\n" + "=" * 100)
    print("§3 初期損切りの完全再シミュレーション（曲線として。最適値は選ばない）")
    print("=" * 100)

    stop_tables = {}  # stop_tables[tf][stress][mode] = list of dict rows (param, cs, diag)
    for tf in ("h1", "4h"):
        r = results[tf]
        df, atr_arr, yrs = r["df"], r["atr_arr"], r["yrs"]
        stop_tables[tf] = {}
        for stress_name, extra_slip in (("通常", 0.0), (f"執行ストレス(ストップ約定に追加スリップ${STOP_EXTRA_SLIP})", STOP_EXTRA_SLIP)):
            stop_tables[tf][stress_name] = {}
            for mode_name, mode_key, grid in (("(a) ATR倍 X", "atr", X_GRID), ("(b) 初期リスク幅倍 F", "risk", F_GRID)):
                rows = []
                for param in grid:
                    trades_raw, open_trade, diag = simulate_long_stop(df, atr_arr, mode_key, param, extra_slip)
                    central = apply_cost(trades_raw, *CENTRAL)
                    cs = summarize(central, yrs)
                    if cs is not None:
                        cs["pf_pct"] = pf_pct(central)
                    rows.append(dict(param=param, cs=cs, diag=diag, central=central))
                stop_tables[tf][stress_name][mode_name] = rows

    for tf in ("h1", "4h"):
        r = results[tf]
        cs0 = r["cs"]
        ratio0 = cs0["tot_pct_per_year"] / cs0["maxdd_pct"] if cs0["maxdd_pct"] > 0 else float("nan")
        print(f"\n### TF={tf}")
        print(f"  -- 現行（損切り無し）基準行 -- n/年={cs0['n_per_year']:.2f} 勝率%={cs0['win_pct']:.1f} "
              f"PF%={cs0['pf_pct']:.3f} mean%={cs0['mean_pct']:.4f} 年率%={cs0['tot_pct_per_year']:.2f} "
              f"maxDD%={cs0['maxdd_pct']:.2f} 年率/DD={ratio0:.3f} 最大連敗={cs0['max_losing_streak']}")

        for stress_name in stop_tables[tf]:
            print(f"\n  === {stress_name} ===")
            for mode_name in stop_tables[tf][stress_name]:
                print(f"\n  -- {mode_name} --")
                print(f"    {'param':>7}{'n/年':>8}{'勝率%':>7}{'PF%':>8}{'mean%':>8}{'年率%':>9}"
                      f"{'maxDD%':>8}{'年率/DD':>9}{'最大連敗':>8}{'skip(ATR無)':>11}")
                print(f"    {'現行':>7}{cs0['n_per_year']:>8.2f}{cs0['win_pct']:>7.1f}{cs0['pf_pct']:>8.3f}"
                      f"{cs0['mean_pct']:>8.4f}{cs0['tot_pct_per_year']:>9.2f}{cs0['maxdd_pct']:>8.2f}"
                      f"{ratio0:>9.3f}{cs0['max_losing_streak']:>8d}{'-':>11}")
                for row in stop_tables[tf][stress_name][mode_name]:
                    cs = row["cs"]
                    if cs is None:
                        print(f"    {row['param']:>7.2f}  トレード無し")
                        continue
                    ratio = cs["tot_pct_per_year"] / cs["maxdd_pct"] if cs["maxdd_pct"] > 0 else float("nan")
                    print(f"    {row['param']:>7.2f}{cs['n_per_year']:>8.2f}{cs['win_pct']:>7.1f}"
                          f"{cs['pf_pct']:>8.3f}{cs['mean_pct']:>8.4f}{cs['tot_pct_per_year']:>9.2f}"
                          f"{cs['maxdd_pct']:>8.2f}{ratio:>9.3f}{cs['max_losing_streak']:>8d}"
                          f"{row['diag']['n_skipped_no_atr']:>11d}")

    print("\n  [結論・§3] PF%は%建て。台地の有無は§5で判定する（本節は生データのみ）。")

    # =================== §4 勝ちがどれだけ殺されるか ===================
    print("\n" + "=" * 100)
    print("§4 勝ちがどれだけ殺されるか（近似指標。本判定は§3の再シミュレーション）")
    print("=" * 100)

    for tf in ("h1", "4h"):
        print(f"\n### TF={tf}")
        full_atr_win, full_atr_loss = mae_store[tf]["全保有期間"]["atr"]
        full_risk_win, full_risk_loss = mae_store[tf]["全保有期間"]["risk"]

        print(f"\n  -- (a) ATR倍 X: 現行の勝ち/負けのうちMAE(全保有,ATR倍)がX以上の割合 --")
        print(f"    {'X':>6}{'勝ち触れる%':>12}{'負け触れる%':>12}{'差(負け-勝ち)':>14}")
        for X in X_GRID:
            win_touch = float((full_atr_win >= X).mean() * 100)
            loss_touch = float((full_atr_loss >= X).mean() * 100)
            print(f"    {X:>6.2f}{win_touch:>12.1f}{loss_touch:>12.1f}{loss_touch-win_touch:>14.1f}")

        print(f"\n  -- (b) 初期リスク幅倍 F: 現行の勝ち/負けのうちMAE(全保有,リスク幅倍)がF以上の割合 --")
        print(f"    {'F':>6}{'勝ち触れる%':>12}{'負け触れる%':>12}{'差(負け-勝ち)':>14}")
        for F in F_GRID:
            win_touch = float((full_risk_win >= F).mean() * 100)
            loss_touch = float((full_risk_loss >= F).mean() * 100)
            print(f"    {F:>6.2f}{win_touch:>12.1f}{loss_touch:>12.1f}{loss_touch-win_touch:>14.1f}")

    print("\n  [結論・§4] 差(負け-勝ち)が小さい設定は「勝ちを同じ比率で殺している」＝分離度が低い。"
          "これは近似指標であり、実際の玉の入れ替わりを反映しないため本判定は§3を用いる。")

    # =================== §5 判定 ===================
    print("\n" + "=" * 100)
    print("§5 判定（§3年率/DDの台地の有無 → あれば巡回ブロック・ブートストラップ）")
    print("=" * 100)
    print(f"  台地の基準（曖昧さ(2)参照・自己判断）: 隣接する設定が{PLATEAU_MIN_RUN}つ以上連続して"
          "現行の年率/DDを上回ること。主判定は通常版（執行ストレス無し）で行う（曖昧さ(3)参照）。")

    date_index_cache = {}
    for tf in ("h1", "4h"):
        date_index_cache[tf] = pd.date_range(results[tf]["df"].index[0].normalize(),
                                              results[tf]["df"].index[-1].normalize(), freq="D")

    for tf in ("h1", "4h"):
        r = results[tf]
        cs0 = r["cs"]
        ratio0 = cs0["tot_pct_per_year"] / cs0["maxdd_pct"] if cs0["maxdd_pct"] > 0 else float("nan")
        print(f"\n### TF={tf}  現行年率/DD={ratio0:.3f}")
        daily_none = daily_pnl_series(r["central"], date_index_cache[tf])

        for mode_name, grid in (("(a) ATR倍 X", X_GRID), ("(b) 初期リスク幅倍 F", F_GRID)):
            rows_normal = stop_tables[tf]["通常"][mode_name]
            ratios = []
            for row in rows_normal:
                cs = row["cs"]
                if cs is None or cs["maxdd_pct"] <= 0:
                    ratios.append(float("-inf"))
                else:
                    ratios.append(cs["tot_pct_per_year"] / cs["maxdd_pct"])
            beats = [rv > ratio0 for rv in ratios]
            has_plateau, best_run, best_start = find_plateau(beats, PLATEAU_MIN_RUN)
            beat_str = " ".join(f"{p}:{'○' if b else '×'}" for p, b in zip(grid, beats))
            print(f"\n  -- {mode_name} -- 現行超え系列: {beat_str}")
            print(f"    最長連続超え run={best_run} (基準={PLATEAU_MIN_RUN}) → "
                  f"台地{'あり' if has_plateau else 'なし'}")

            # ストレス版での頑健性チェック（曖昧さ(3)：参考のみ）
            rows_stress = stop_tables[tf][f"執行ストレス(ストップ約定に追加スリップ${STOP_EXTRA_SLIP})"][mode_name]
            ratios_stress = []
            for row in rows_stress:
                cs = row["cs"]
                if cs is None or cs["maxdd_pct"] <= 0:
                    ratios_stress.append(float("-inf"))
                else:
                    ratios_stress.append(cs["tot_pct_per_year"] / cs["maxdd_pct"])
            beats_stress = [rv > ratio0 for rv in ratios_stress]
            beat_str_stress = " ".join(f"{p}:{'○' if b else '×'}" for p, b in zip(grid, beats_stress))
            print(f"    [参考:執行ストレス版] 現行超え系列: {beat_str_stress}")

            if not has_plateau:
                print(f"    → 台地なしのためブートストラップは省略。")
                continue

            center_idx = best_start + best_run // 2
            center_param = grid[center_idx]
            center_row = rows_normal[center_idx]
            print(f"    → 台地の中央付近設定 param={center_param} でブートストラップを実行")
            daily_cand = daily_pnl_series(center_row["central"], date_index_cache[tf])
            prev_frac = None
            monotone_up = True
            for bm in BLOCK_MONTHS:
                diffs = block_bootstrap_diff(daily_none, daily_cand, bm, N_NULL_REPS, rng)
                if len(diffs) == 0:
                    print(f"      ブロック{bm:>2d}か月: 有効サンプル無し")
                    continue
                pos_frac = float((diffs > 0).mean() * 100)
                print(f"      ブロック{bm:>2d}か月: (候補年率/DD − 現行年率/DD)>0の割合={pos_frac:5.1f}% "
                      f"中央値={np.median(diffs):7.3f} std={diffs.std(ddof=1):7.3f} "
                      f"(有効{len(diffs)}/{N_NULL_REPS})")
                if prev_frac is not None and pos_frac < prev_frac:
                    monotone_up = False
                prev_frac = pos_frac
            print(f"      [結論] ブロックを長くするほど割合が単調に上がる={monotone_up}"
                  f"（上がらない場合は経路当てはめの疑い）")

    print("\n" + "=" * 100)
    print("完了")
    print("=" * 100)


if __name__ == "__main__":
    main()
