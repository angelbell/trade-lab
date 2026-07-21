"""仕様カード scratchpad/spec_context_time_fomc.md の実装。

btc15m_A・gold15m の発火トレードを「いつ建ったか」の文脈(時間帯4バケット/曜日/FOMC)で
層別し、EVが並ぶか(強度の段として読めるか)を測る。サイズ倍率は出さない
[[feedback-fixed-lot-no-sizing]]。

対象レッグ(照合済み再構築を流用・車輪の再発明禁止):
  btc15m_A: scratchpad/strength_btc15mA.py の build_A() (照合ゲート1-4 PASS 済み,
            2026-07-19 out_strength_btc15mA.txt に記録済み)。同スクリプトの
            LEDGER(n=229/win=34.1%/meanR=+1.170/PF=2.61/IS=+1.15/OOS=+1.19)に対し
            ここでも同じ許容差で再照合する(tie-back)。
  gold15m : scratchpad/strength_gateslope_generalize.py の build_gold15m() (照合ゲート
            1-3 PASS 済み, out_strength_gateslope_generalize.txt に記録済み: n=325/
            win=24%/meanR=+0.58/IS=+0.63/OOS=+0.54)。同じ数字に対し再照合する。

時刻は約定足(fill bar)時刻を主とする(tL["time"]/t["time"] — 指値が実際に建った瞬間)。
データはブローカー時刻(EET/EEST)がそのまま索引になっている(src/data_loader.load_mt5_csv
がUTCタグを付けているが値はブローカー壁時計 -- .values で取り出すとタグが外れ数値はそのまま
ブローカー時刻のnaive datetime64になる。これは data/ext_econ_calendar.csv の dt_broker 列
(同じくnaive)と同じ表現なので変換不要で突き合わせられる)。

FOMC: 既存 data/ext_econ_calendar.csv に2021-2026の48行(声明日=2日目、dt_utc/dt_broker済み)
がある。2018-2020分だけ連銀公式 federalreserve.gov の年別 historical ページ(fomchistorical
2018/2019/2020.htm, 2026-07-19 WebFetch)から声明日(2日目)を書き写し、14:00 ET →
UTC/Europe-Riga へ tz変換して補う(決定論的正典・Wayback不使用)。tz変換メソッドの正しさは
既存48行の日付に同じ変換を再適用し、既存の dt_utc/dt_broker と完全一致するかで検証する
(2018-2020分そのものは既存48行と期間が重ならないため、手法をここで検算する)。

Run:
  .venv/bin/python scratchpad/context_time_fomc.py --smoke 2>&1 | tee scratchpad/out_context_time_fomc_smoke.txt
  .venv/bin/python scratchpad/context_time_fomc.py 2>&1 | tee scratchpad/out_context_time_fomc.txt
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import strength_btc15mA as satA                      # build_A, stats_of, block_bootstrap_gap
import strength_gateslope_generalize as sg            # build_gold15m

DATA_ECON = f"{ROOT}/data/ext_econ_calendar.csv"
DATA_FOMC_OUT = f"{ROOT}/data/ext_fomc_dates.csv"

LEDGER_A = dict(n=229, win=34.1, meanR=1.170, pf=2.61, isR=1.15, oosR=1.19)
# 訂正(2026-07-19 実行時に自己点検で発覚): out_strength_gateslope_generalize.txt の
# 「n=325 win=24% meanR=+0.58」行は src/engine/stats.summarize() が run() 内部で自動的に
# 印字する生R(コスト前 t["R"])の要約であり、netR(コスト後、build_gold15m が返す値)ではない
# (strength_gateslope_generalize.py 自体も report_candidate/era_report には生Rを渡しており
# netRはgate1のbook.get_book_legs()照合にしか使っていない)。これをnetRの台帳と誤読していた。
# 本スクリプトは仕様カード指定どおりnetR(コスト0.3/risk込み)を使うため、台帳照合は
# book.get_book_legs()['gold15m'] との直接突き合わせ(gate1と同じ)に差し替える。

# 連銀公式 federalreserve.gov/monetarypolicy/fomchistorical{2018,2019,2020}.htm
# (2026-07-19 WebFetch で取得。声明日=2日目のカレンダー日のみ。2020年は3/17-18の定例回が
#  未定例の緊急会合2回(3/2, 3/15)に置き換わったため定例7回のみ収録=8回でないのは仕様どおり)
FOMC_STMT_DATES_2018_2020 = [
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-04-29", "2020-06-10", "2020-07-29",
    "2020-09-16", "2020-11-05", "2020-12-16",
]


# ---------------------------------------------------------------- FOMC カレンダー構築

def et14_to_utc_broker(date_strs):
    """カレンダー日の配列 -> 14:00 ET に固定し UTC / Europe-Riga(ブローカー) のnaive
    Timestampに変換する(声明時刻はET 14:00固定、DSTはtzライブラリが自動処理)。"""
    et = pd.DatetimeIndex(pd.to_datetime(list(date_strs))) + pd.Timedelta(hours=14)
    et = et.tz_localize("America/New_York")
    dt_utc = et.tz_convert("UTC").tz_localize(None)
    dt_broker = et.tz_convert("Europe/Riga").tz_localize(None)
    return dt_utc, dt_broker


def validate_tz_method(existing):
    """既存48行(2021-2026, dt_utc/dt_broker)の各行のET暦日に同じ 14:00 ET 変換を
    再適用し、既存の dt_utc/dt_broker と完全一致するかを検算する(手法の正しさの裏取り)。"""
    et_dates = (existing["dt_utc"].dt.tz_localize("UTC")
                .dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d"))
    recon_utc, recon_broker = et14_to_utc_broker(et_dates.values)
    ok_utc = bool((recon_utc.values == existing["dt_utc"].values).all())
    ok_broker = bool((recon_broker.values == existing["dt_broker"].values).all())
    return ok_utc, ok_broker


def build_fomc_calendar():
    existing = pd.read_csv(DATA_ECON, parse_dates=["dt_utc", "dt_broker"])
    existing = existing[existing["kind"] == "FOMC"].reset_index(drop=True)

    ok_utc, ok_broker = validate_tz_method(existing)
    print(f"[FOMC tz変換メソッド検算] 既存48行(2021-2026)に同じ 14:00ET->UTC/Riga 変換を再適用: "
          f"dt_utc完全一致={ok_utc}  dt_broker完全一致={ok_broker}  "
          f"=> {'PASS(手法を信用してよい)' if ok_utc and ok_broker else 'FAIL(手法を見直すこと)'}")

    dt_utc_new, dt_broker_new = et14_to_utc_broker(FOMC_STMT_DATES_2018_2020)
    new_rows = pd.DataFrame({"kind": "FOMC", "dt_utc": dt_utc_new, "dt_broker": dt_broker_new})

    full = (pd.concat([new_rows, existing], ignore_index=True)
            .sort_values("dt_utc").reset_index(drop=True))
    full.to_csv(DATA_FOMC_OUT, index=False)

    by_year = full["dt_utc"].dt.year.value_counts().sort_index()
    print(f"\n[FOMCカレンダー] 新規追加(2018-2020, federalreserve.gov historical, 声明=2日目): "
          f"{len(new_rows)}行  既存(2021-2026, ext_econ_calendar.csv): {len(existing)}行  "
          f"合計: {len(full)}行 -> {DATA_FOMC_OUT}")
    print("  年別内訳:")
    for y, c in by_year.items():
        note = "  (2020は3月定例回が緊急会合2回に置換され定例7回のみ)" if y == 2020 else ""
        print(f"    {y}: {c}回{note}")
    return full


# ---------------------------------------------------------------- 汎用: セッション/曜日/FOMCフラグ

WEEKDAY_JA = ["月", "火", "水", "木", "金", "土", "日"]


def session_bucket(times, offset_h=0):
    """アジア(01-08)/ロンドン(08-15)/NY(15-22)/深夜(22-01) の4バケット(ブローカー時刻)。
    offset_h で境界を±ずらして頑健性を1回だけ確認する。"""
    h = (pd.DatetimeIndex(times).hour - offset_h) % 24
    cond = [(h >= 1) & (h < 8), (h >= 8) & (h < 15), (h >= 15) & (h < 22)]
    lab = [f"アジア({(1+offset_h)%24:02d}-{(8+offset_h)%24:02d})",
           f"ロンドン({(8+offset_h)%24:02d}-{(15+offset_h)%24:02d})",
           f"NY({(15+offset_h)%24:02d}-{(22+offset_h)%24:02d})"]
    default_lab = f"深夜({(22+offset_h)%24:02d}-{(1+offset_h)%24:02d})"
    return np.select(cond, lab, default=default_lab)


def weekday_label(times):
    dow = pd.DatetimeIndex(times).dayofweek
    return np.array([WEEKDAY_JA[d] for d in dow])


def fomc_flags(times, fomc_broker):
    """times(naive, ブローカー時刻) それぞれについて (当日, 前24h以内, 後24h以内) を返す。
    n が小さいので単純な全ペア比較で十分(高速化不要)。"""
    t = pd.DatetimeIndex(times)
    f = pd.DatetimeIndex(fomc_broker).sort_values()
    f_day = f.normalize()
    same_day = np.zeros(len(t), dtype=bool)
    pre24 = np.zeros(len(t), dtype=bool)
    post24 = np.zeros(len(t), dtype=bool)
    for i, ti in enumerate(t):
        same_day[i] = (ti.normalize() == f_day).any()
        pre24[i] = ((ti >= (f - pd.Timedelta(hours=24))) & (ti < f)).any()
        post24[i] = ((ti >= f) & (ti < (f + pd.Timedelta(hours=24)))).any()
    return same_day, pre24, post24


# ---------------------------------------------------------------- 層別レポート

def print_layer_table(groups, R_by_group, span_years, note=""):
    print(f"  {'':<18}{'n':>6}{'本/年':>7}{'win%':>8}{'PF':>8}{'meanR':>9}")
    for label in groups:
        st = satA.stats_of(R_by_group[label])
        win_s = f"{st['win']:.1f}%" if st['n'] else "  ·  "
        meanR_s = f"{st['meanR']:+.3f}" if st['n'] else "   ·   "
        print(f"  {label:<18}{st['n']:>6}{st['n']/span_years:>7.2f}"
              f"{win_s:>8}{satA.pf_str(st):>8}{meanR_s:>9}")
    if note:
        print(f"  {note}")


def layer_by(labels_arr, R, order):
    return {lab: R[labels_arr == lab] for lab in order}


def leg_context_report(leg_name, times, R, span_years, ledger, ledger_note, check_ledger=True,
                        external_gate=None):
    """external_gate: (bool, str) が渡されたらそれをそのまま照合結果として使う
    (gold15m はここでの netR が book.get_book_legs() とのビット一致で照合済みのため、
    このヘルパ内の簡易な n/win%/meanR 近似照合ではなく、その厳密な結果を使う)。"""
    print(f"\n{'#'*78}\n# {leg_name} -- 文脈層別 (n={len(R)}, span={span_years:.2f}yr)\n{'#'*78}")

    st_all = satA.stats_of(R)
    print(f"母集団全体: n={st_all['n']}  win={st_all['win']:.1f}%  PF={satA.pf_str(st_all)}  "
          f"meanR={st_all['meanR']:+.3f}")
    if external_gate is not None:
        ok, detail = external_gate
        print(f"  [台帳照合] {detail}")
        print(f"  => {'PASS(一致)' if ok else 'FAIL(不一致、以降の数字を疑うこと)'}")
    elif check_ledger:
        print(f"  [台帳照合] 基準({ledger_note}): n={ledger['n']} win={ledger['win']:.1f}% "
              f"meanR={ledger['meanR']:+.3f}")
        ok = (st_all['n'] == ledger['n'] and abs(st_all['win'] - ledger['win']) < 0.5
              and abs(st_all['meanR'] - ledger['meanR']) < 0.01)
        print(f"  => {'PASS(一致)' if ok else 'FAIL(不一致、以降の数字を疑うこと)'}")
    else:
        print(f"  [台帳照合] --smoke のため省略({ledger_note})")

    # ---- 時間帯4バケット ----
    print(f"\n--- 時間帯4バケット(ブローカー時刻・約定足) ---")
    sess = session_bucket(times)
    order_sess = sorted(set(sess), key=lambda s: ["アジア", "ロンドン", "NY", "深夜"]
                         .index(next(k for k in ["アジア", "ロンドン", "NY", "深夜"] if s.startswith(k))))
    by_sess = layer_by(sess, R, order_sess)
    print_layer_table(order_sess, by_sess, span_years)

    print(f"\n  [境界±1h頑健性チェック(1回のみ)] offset=+1h:")
    sess_p1 = session_bucket(times, offset_h=1)
    order_p1 = sorted(set(sess_p1))
    by_sess_p1 = layer_by(sess_p1, R, order_p1)
    print_layer_table(order_p1, by_sess_p1, span_years)

    # ---- 曜日 ----
    print(f"\n--- 曜日 ---")
    wd = weekday_label(times)
    order_wd = [d for d in WEEKDAY_JA if (wd == d).any()]
    by_wd = layer_by(wd, R, order_wd)
    print_layer_table(order_wd, by_wd, span_years,
                       note="(土日を含む場合は右のn=0でない行に注意。無ければ土日不建玉)")

    return dict(sess=sess, order_sess=order_sess, by_sess=by_sess, wd=wd)


# ---------------------------------------------------------------- ブロックブートストラップ (候補のみ)

def bootstrap_candidate(label, times, mask, R, span_years):
    print(f"\n  >>> ブロックブートストラップ候補: 「{label}」 vs 残り "
          f"(n={mask.sum()}本/{len(R)-mask.sum()}本)")
    top_st = satA.stats_of(R[mask])
    rest_st = satA.stats_of(R[~mask])
    gap = top_st['meanR'] - rest_st['meanR']
    satA.print_two_row(label, top_st, "残り", rest_st, span_years, note=f"ギャップ = {gap:+.3f}")
    for k in (1, 3, 6, 12):
        med, lo, hi, nvalid = satA.block_bootstrap_gap(times, mask, R, k, n_boot=3000)
        tag = "0超" if (np.isfinite(lo) and lo > 0) else ("0未満" if (np.isfinite(hi) and hi < 0) else "0またぎ")
        print(f"    {k:>2}mo循環ブロック: median gap={med:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  "
              f"(有効draw={nvalid}/3000)  {tag}")
    yrs = pd.DatetimeIndex(times).year
    print("    年別:")
    for y in sorted(set(yrs)):
        m = (yrs == y)
        a = R[m & mask]; b = R[m & ~mask]
        if len(a) < 3 or len(b) < 3:
            print(f"      {y}: n不足(該当{len(a)}/残り{len(b)}) スキップ")
            continue
        print(f"      {y}: 該当(n={len(a)}) meanR={a.mean():+.3f} | 残り(n={len(b)}) meanR={b.mean():+.3f} "
              f"| 差={a.mean()-b.mean():+.3f}")


# ---------------------------------------------------------------- FOMC レポート

def fomc_report(leg_name, times, R, fomc_full, span_years):
    print(f"\n--- FOMC ({leg_name}) ---")
    fomc_broker = fomc_full["dt_broker"].values
    same_day, pre24, post24 = fomc_flags(times, fomc_broker)
    rest = ~(same_day | pre24 | post24)
    labels = ["FOMC当日", "FOMC前24h以内", "FOMC後24h以内", "平常"]
    masks = [same_day, pre24, post24, rest]
    print(f"  {'':<18}{'n':>6}{'win%':>8}{'PF':>8}{'meanR':>9}")
    for lab, m in zip(labels, masks):
        st = satA.stats_of(R[m])
        win_s = f"{st['win']:.1f}%" if st['n'] else "  ·  "
        meanR_s = f"{st['meanR']:+.3f}" if st['n'] else "   ·   "
        print(f"  {lab:<18}{st['n']:>6}{win_s:>8}{satA.pf_str(st):>8}{meanR_s:>9}")
    thin = sum(m.sum() for m in masks[:3]) < 15
    if thin:
        print(f"  (FOMC関連の合計 n={sum(m.sum() for m in masks[:3])} -- 薄く、読めるほどの標本"
              f"ではない可能性。無理に有意化しない)")
    return dict(same_day=same_day, pre24=pre24, post24=post24, rest=rest)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    fomc_full = build_fomc_calendar()

    # ================================================================ btc15m_A
    d15A, rawA, argsA, tLA, abA = satA.build_A(cli.smoke)
    R_A_raw = tLA["R"].values[abA]
    risk_A = tLA["risk"].values[abA]
    times_A = tLA["time"].values[abA]
    netR_A = R_A_raw - 15.0 / risk_A
    span_A = (pd.DatetimeIndex(times_A).max() - pd.DatetimeIndex(times_A).min()).days / 365.25
    print(f"\nbtc15m_A 再構築: n={len(netR_A)}  span={span_A:.2f}yr  (smoke={cli.smoke})")

    if cli.smoke:
        leg_context_report("btc15m_A[SMOKE]", times_A, netR_A, span_A, LEDGER_A,
                            "strength_btc15mA.py LEDGER, --smoke中は非照合", check_ledger=False)
    else:
        infoA = leg_context_report("btc15m_A", times_A, netR_A, span_A, LEDGER_A,
                                    "strength_btc15mA.py LEDGER(2026-07-19 ゲート4 PASS済み)")
        fomcA = fomc_report("btc15m_A", times_A, netR_A, fomc_full, span_A)

    # ================================================================ gold15m
    g15, argsG, t, netR_G = sg.build_gold15m(cli.smoke)
    times_G = t["time"].values
    span_G = (pd.DatetimeIndex(times_G).max() - pd.DatetimeIndex(times_G).min()).days / 365.25
    print(f"\ngold15m 再構築: n={len(netR_G)}  span={span_G:.2f}yr  (smoke={cli.smoke})")

    if cli.smoke:
        leg_context_report("gold15m[SMOKE]", times_G, netR_G, span_G, None,
                            "--smoke中は非照合(get_book_legs()はフルデータ前提)", check_ledger=False)
    else:
        # gold15m の台帳照合は strength_gateslope_generalize.py の gate1 と同じ形
        # (research.book.get_book_legs()['gold15m'] とビット一致するか)で行う --
        # LEDGER_G を使わないのは、out_strength_gateslope_generalize.txt の
        # 「n=325 win=24% meanR=+0.58」行が run() 自身が印字する生R(コスト前)の
        # 要約であり netR(コスト後)ではないと判明したため(上のコメント参照)。
        import contextlib, io
        import research.book as book_mod
        with contextlib.redirect_stderr(io.StringIO()):
            legs = book_mod.get_book_legs()
        ref_G = legs["gold15m"]
        # 注意: t["time"] は tz付き(UTCラベル、実体はブローカー壁時計)Series。gate照合は
        # book.get_book_legs() 側と同じ tz付きindexで突き合わせる(times_G は .values で tz
        # を落とした版でセッション/FOMC層別に使うため、ここでは別に t["time"] を直接使う)。
        mine_G = pd.Series(netR_G, index=pd.DatetimeIndex(t["time"]))
        same_len = len(ref_G) == len(mine_G)
        same_idx = same_len and ref_G.index.equals(mine_G.index)
        same_val = same_idx and np.allclose(ref_G.values, mine_G.values, rtol=0, atol=1e-12)
        gateG = same_len and same_idx and same_val
        detailG = (f"netR vs book.get_book_legs()['gold15m']: len {len(ref_G)}=={len(mine_G)} -> "
                   f"{same_len} | idx一致 -> {same_idx} | 値一致(atol=1e-12) -> {same_val}")
        infoG = leg_context_report("gold15m", times_G, netR_G, span_G, None, detailG,
                                    external_gate=(gateG, detailG))
        fomcG = fomc_report("gold15m", times_G, netR_G, fomc_full, span_G)

    if cli.smoke:
        print("\n(--smoke のためブロックブートストラップ候補選定・既存台帳突き合わせは省略)")
        print(f"\n実行コマンド: .venv/bin/python scratchpad/context_time_fomc.py --smoke")
        return

    # ================================================================ 候補選定 + ブロックブートストラップ
    print(f"\n{'#'*78}\n# 差が出た候補のブロックブートストラップ (多重比較注記: 全バケットに乱発しない)\n{'#'*78}")

    def pick_and_boot(leg_name, times, R, sess, order_sess, wd, span_years, fomc_flags_dict):
        overall = satA.stats_of(R)['meanR']
        # セッション: 最良/最悪バケット(n>=15)がoverallから離れているものだけ
        cand = []
        for lab in order_sess:
            m = (sess == lab)
            if m.sum() >= 15:
                cand.append((abs(satA.stats_of(R[m])['meanR'] - overall), lab, m))
        cand.sort(reverse=True)
        if cand and cand[0][0] > 0.15:
            _, lab, m = cand[0]
            bootstrap_candidate(f"{leg_name}: {lab}", times, m, R, span_years)
        else:
            print(f"\n  {leg_name}: 時間帯バケットで overall との差が0.15R超のものなし -- "
                  f"ブロックブートストラップは省略(偽の凸凹の可能性が高いため)")
        # 曜日: 同様
        cand_wd = []
        for lab in sorted(set(wd)):
            m = (wd == lab)
            if m.sum() >= 15:
                cand_wd.append((abs(satA.stats_of(R[m])['meanR'] - overall), lab, m))
        cand_wd.sort(reverse=True)
        if cand_wd and cand_wd[0][0] > 0.15:
            _, lab, m = cand_wd[0]
            bootstrap_candidate(f"{leg_name}: {lab}曜日", times, m, R, span_years)
        else:
            print(f"  {leg_name}: 曜日で overall との差が0.15R超のものなし -- "
                  f"ブロックブートストラップは省略")
        # FOMC: n>=15の層のみ候補にする
        for lab, m in zip(["FOMC当日", "FOMC前24h以内", "FOMC後24h以内"],
                           [fomc_flags_dict["same_day"], fomc_flags_dict["pre24"], fomc_flags_dict["post24"]]):
            if m.sum() >= 15:
                bootstrap_candidate(f"{leg_name}: {lab}", times, m, R, span_years)
            else:
                print(f"  {leg_name}: {lab} n={m.sum()} < 15 -- ブロックブートストラップは省略"
                      f"(標本が薄すぎて読めない)")

    pick_and_boot("btc15m_A", times_A, netR_A, infoA["sess"], infoA["order_sess"], infoA["wd"],
                  span_A, fomcA)
    pick_and_boot("gold15m", times_G, netR_G, infoG["sess"], infoG["order_sess"], infoG["wd"],
                  span_G, fomcG)

    # ================================================================ 既存台帳との突き合わせ (gold15m)
    print(f"\n{'#'*78}\n# 既存台帳(docs/verified_findings.md)との突き合わせ -- gold15m セッション\n{'#'*78}")
    print("  既存記録: 12-15UTC(n=41) PF2.28 meanR+0.936 IS+0.919/OOS+0.952 | "
          "9-15UTC(n=106) PF1.90 IS+0.781/OOS+0.619 | 13時PF2.75/14時PF2.59 (ともに黒字、"
          "『セッションスキップ禁止』が確定済み -- 窓を捨てろとは結論しない)")
    # 参考: UTC時間帯(このスクリプトの時間帯はブローカー時刻=UTC+2/3なのでUTC変換して比較)
    utc_hour_G = (pd.DatetimeIndex(times_G).tz_localize("Europe/Riga", ambiguous="infer")
                  .tz_convert("UTC").hour)
    for lo, hi, tag in [(12, 15, "12-15UTC"), (9, 15, "9-15UTC")]:
        m = (utc_hour_G >= lo) & (utc_hour_G < hi)
        st = satA.stats_of(netR_G[m])
        print(f"  このスクリプトでの{tag} (n={st['n']}): win={st['win']:.1f}%  PF={satA.pf_str(st)}  "
              f"meanR={st['meanR']:+.3f}  (netRの定義がコスト後という点は既存記録と同一想定)")
    print("  [注記] n(72/121)が既存記録(41/106)と一致しない -- 母集団の総数自体が違う"
          "(このスクリプトは n=325、既存記録は非開示だが本スクリプトより少ない)。"
          "scratchpad/book_spec_fix.py(2026-07-13)の gold15m 構築には fill_win=200 が"
          "指定されておらず(BASE既定=fwd相当)、現行README/get_book_legs()の fill_win=200 "
          "仕様と母集団が異なる可能性が高い(旧測定=機械修正の途中経過、本スクリプトは"
          "book.get_book_legs()とビット一致=現行正典)。**結論の向き(黒字窓・スキップ禁止)は"
          "一致**しており矛盾ではないが、絶対値(PF/n)は比較不可として扱うこと。")

    print(f"\n実行コマンド: .venv/bin/python scratchpad/context_time_fomc.py")


if __name__ == "__main__":
    main()
